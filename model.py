from transformers import SiglipForImageClassification,SiglipPreTrainedModel,SiglipVisionModel
import torch
from typing import Optional, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs  import ImageClassifierOutput 
from transformers.models.siglip.configuration_siglip  import SiglipConfig

class SiglipForImageClassificationWithSelfDistillation(SiglipForImageClassification):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__(config)
        self.hidden_size = config.vision_config.hidden_size
        #定义额外的分类器
        self.classifier_layer3 = nn.Linear(self.hidden_size, self.num_labels)
        self.classifier_layer6 = nn.Linear(self.hidden_size, self.num_labels)
        self.classifier_layer9  = nn.Linear(self.hidden_size,  self.num_labels)    # 第 9 层 

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        self_distillation: bool = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        self_distillation (`bool`, *optional*, defaults to `False`):
            If `True`, enables self-distillation mode, returning logits and features from multiple layers.
            If `False`, behaves like the original SiglipForImageClassification.

        Returns:
            ImageClassifierOutput with:
            - loss: Optional loss if labels are provided.
            - logits: List of logits from all classifiers [logits_deep, logits_mid, logits_shallow] if self_distillation=True,
                     otherwise a single logits tensor.
            - hidden_states: List of features from selected layers [deep_feature, mid_feature, shallow_feature] if self_distillation=True,
                            otherwise the original hidden states.
            - attentions: Optional attention weights.
        """
        if not self_distillation:
            # 直接调用父类的 forward 方法，保持原始行为
            return super().forward(
                pixel_values=pixel_values,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        # 自蒸馏模式
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True  # 强制输出中间层
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        hidden_states = outputs.hidden_states
        layer3_feature = torch.mean(hidden_states[3], dim=1)  # 第 3 层
        layer6_feature = torch.mean(hidden_states[6], dim=1)      # 第 6 层
        layer9_feature = torch.mean(hidden_states[9], dim=1)
        deep_feature = torch.mean(hidden_states[-1], dim=1)    # 最后一层

        logits_deep = self.classifier(deep_feature)
        logits_layer9 = self.classifier_layer9(layer9_feature)
        logits_layer6 = self.classifier_layer6(layer6_feature)
        logits_layer3 = self.classifier_layer3(layer3_feature)

        all_logits = [logits_deep, logits_layer9, logits_layer6, logits_layer3]
        all_features = [deep_feature, layer9_feature, layer6_feature, layer3_feature]

        loss = None
        if labels is not None:
            labels = labels.to(logits_deep.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_deep.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (all_logits, all_features) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=all_logits,
            hidden_states=all_features,
            attentions=outputs.attentions,
        )
    



class SiglipStudentForImageClassification(SiglipPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipConfig, shallow_layers: int = 3) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.shallow_layers = shallow_layers  # 浅层网络的层数（例如 3）

        # 创建视觉编码器，但只保留浅层部分
        vision_model = SiglipVisionModel._from_config(config.vision_config)
        self.vision_model = vision_model.vision_model

        # 只保留前 shallow_layers 层
        self.vision_model.encoder.layers = nn.ModuleList(
            self.vision_model.encoder.layers[:self.shallow_layers]
        )

        # 浅层分类头
        self.classifier = nn.Linear(config.vision_config.hidden_size, self.num_labels)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]
        sequence_output = torch.mean(sequence_output, dim=1)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )