import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from model import SiglipForImageClassificationWithSelfDistillation, SiglipStudentForImageClassification
from data import get_data_loaders
import argparse
import os
from transformers import SiglipConfig
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CrossEntropy(outputs, targets, temperature):
    log_softmax_outputs = F.log_softmax(outputs/temperature, dim=1)
    softmax_targets = F.softmax(targets/temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def train_self_distillation(model, trainloader, testloader, args):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
    # 添加余弦学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)
    best_acc = 0

    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(args.epoch):
        model.train()
        sum_loss, total = 0.0, 0.0
        correct = [0 for _ in range(5)]  # Deep, Layer9, Layer6, Layer3, Ensemble
        predicted = [0 for _ in range(5)]

        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{args.epoch}", unit="batch", leave=True) as pbar:
            for i, (inputs, labels) in enumerate(trainloader):
                if inputs is None:
                    pbar.update(1)
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(pixel_values=inputs, self_distillation=True)
                logits_list = outputs.logits
                features_list = outputs.hidden_states

                ensemble = sum(logits_list) / len(logits_list)
                ensemble.detach_()

                loss = torch.FloatTensor([0.]).to(device)
                loss += criterion(logits_list[0], labels)

                teacher_output = logits_list[0].detach()
                teacher_feature = features_list[0].detach()

                for index in range(1, len(logits_list)):
                    loss += CrossEntropy(logits_list[index], teacher_output, args.temperature) * args.loss_coefficient
                    loss += criterion(logits_list[index], labels) * (1 - args.loss_coefficient)
                    if index != len(logits_list) - 1:
                        loss += torch.dist(features_list[index], teacher_feature) * args.feature_loss_coefficient

                sum_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(labels.size(0))

                logits_list.append(ensemble)
                for classifier_index in range(len(logits_list)):
                    _, predicted[classifier_index] = torch.max(logits_list[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())

                if i % 200 == 199:
                    acc_str = '[epoch:%d, iter:%d] Loss: %.03f | Acc: Deep: %.2f%% Layer9: %.2f%% Layer6: %.2f%% Layer3: %.2f%% Ensemble: %.2f%%' % (
                        epoch + 1, i + 1, sum_loss / (i + 1),
                        100 * correct[0] / total, 100 * correct[1] / total,
                        100 * correct[2] / total, 100 * correct[3] / total,
                        100 * correct[4] / total)
                    tqdm.write(acc_str)
                pbar.update(1)

        # 测试阶段
        tqdm.write("Waiting Test!")
        with torch.no_grad():
            correct = [0 for _ in range(5)]
            total = 0.0
            for data in testloader:
                if data[0] is None:
                    continue
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images, self_distillation=True)
                logits_list = outputs.logits
                ensemble = sum(logits_list) / len(logits_list)
                logits_list.append(ensemble)
                for classifier_index in range(len(logits_list)):
                    _, predicted = torch.max(logits_list[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            acc_str = 'Test Set Accuracy: Deep: %.4f%% Layer9: %.4f%% Layer6: %.4f%% Layer3: %.4f%% Ensemble: %.4f%%' % (
                100 * correct[0] / total, 100 * correct[1] / total,
                100 * correct[2] / total, 100 * correct[3] / total,
                100 * correct[4] / total)
            tqdm.write(acc_str)
            if correct[-1] / total > best_acc:
                best_acc = correct[-1] / total
                tqdm.write(f"Best Accuracy Updated: {best_acc * 100:.1f}")
                save_file(model.state_dict(), "./checkpoints/teacher_best_siglip_self_distillation.safetensors")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"Learning rate updated to: {current_lr:.8f}")

    # 保存最终教师模型
    save_file(model.state_dict(), "./checkpoints/teacher_final_siglip_self_distillation.safetensors")
    tqdm.write(f"Training Finished, TotalEPOCH={args.epoch}, Best Accuracy={best_acc:.3f}")
    tqdm.write("Final teacher model saved to ./checkpoints/teacher_final_siglip_self_distillation.safetensors")
    return model

def extract_student_model(teacher_model, config, best_save_path="./checkpoints/student_best_model.safetensors", final_save_path="./checkpoints/student_final_model.safetensors"):
    student = SiglipStudentForImageClassification(config, shallow_layers=3)
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = {}
    for key, value in teacher_state_dict.items():
        if (key.startswith("vision_model.embeddings") or
            key.startswith("vision_model.encoder.layers.0") or
            key.startswith("vision_model.encoder.layers.1") or
            key.startswith("vision_model.encoder.layers.2")):
            student_state_dict[key] = value
        elif key == "classifier_layer3.weight" or key == "classifier_layer3.bias":
            student_key = key.replace("classifier_layer3", "classifier")
            student_state_dict[student_key] = value

    student.load_state_dict(student_state_dict, strict=False)
    student.to(device)
    
    # 保存学生模型（最终状态）
    save_file(student_state_dict, final_save_path)
    tqdm.write(f"Student model (final) saved to {final_save_path}")

    # 加载最佳教师模型并保存对应的学生模型
    if os.path.exists("./checkpoints/teacher_best_siglip_self_distillation.safetensors"):
        best_teacher_state_dict = load_file("./checkpoints/teacher_best_siglip_self_distillation.safetensors")
        best_student_state_dict = {}
        for key, value in best_teacher_state_dict.items():
            if (key.startswith("vision_model.embeddings") or
                key.startswith("vision_model.encoder.layers.0") or
                key.startswith("vision_model.encoder.layers.1") or
                key.startswith("vision_model.encoder.layers.2")):
                best_student_state_dict[key] = value
            elif key == "classifier_layer3.weight" or key == "classifier_layer3.bias":
                student_key = key.replace("classifier_layer3", "classifier")
                best_student_state_dict[student_key] = value
        save_file(best_student_state_dict, best_save_path)
        tqdm.write(f"Student model (best) saved to {best_save_path}")
    else:
        tqdm.write("No best teacher model found, skipping best student model save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Distillation Training for SigLIP')
    parser.add_argument('--epoch', default=5, type=int, help="training epochs")
    parser.add_argument('--loss_coefficient', default=0.3, type=float)
    parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
    parser.add_argument('--dataset_path', default="AI-vs-Deepfake-vs-Real", type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--init_lr', default=0.000001, type=float)
    args = parser.parse_args()

    trainloader, testloader = get_data_loaders(args.dataset_path, args.batchsize)

    config = SiglipConfig.from_json_file("AI-vs-Deepfake-vs-Real-v2.0/config.json")
    state_dict = load_file("AI-vs-Deepfake-vs-Real-v2.0/model.safetensors")
    model = SiglipForImageClassificationWithSelfDistillation(config)
    model.load_state_dict(state_dict, strict=False)

    trained_model = train_self_distillation(model, trainloader, testloader, args)
    extract_student_model(trained_model, config)