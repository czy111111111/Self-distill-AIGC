---
license: apache-2.0
task_categories:
- image-classification
language:
- en
tags:
- deepfake
- ai
- real
size_categories:
- 1K<n<10K
---
# **AI vs Deepfake vs Real**  

**AI vs Deepfake vs Real** is a dataset designed for image classification, distinguishing between artificial, deepfake, and real images. This dataset includes a diverse collection of high-quality images to enhance classification accuracy and improve the model’s overall efficiency. By providing a well-balanced dataset, it aims to support the development of more robust AI-generated and deepfake detection models.  

# **Label Mappings**  
- **Mapping of IDs to Labels:** `{0: 'Artificial', 1: 'Deepfake', 2: 'Real'}`  
- **Mapping of Labels to IDs:** `{'Artificial': 0, 'Deepfake': 1, 'Real': 2}`  

This dataset serves as a valuable resource for training, evaluating, and benchmarking AI models in the field of deepfake and AI-generated image detection.  

# **Dataset Composition**  

The **AI vs Deepfake vs Real** dataset is composed of modular subsets derived from the following datasets:  

- [open-image-preferences-v1](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1)
- [Deepfakes-QA-Patch1](https://huggingface.co/datasets/prithivMLmods/Deepfakes-QA-Patch1)  
- [Deepfakes-QA-Patch2](https://huggingface.co/datasets/prithivMLmods/Deepfakes-QA-Patch2)  

The dataset is evenly distributed across three categories:  
- **Artificial** (33.3%)  
- **Deepfake** (33.3%)  
- **Real** (33.3%)  

With a total of **9,999 entries**, this balanced distribution ensures better generalization and improved robustness in distinguishing between AI-generated, deepfake, and real images.