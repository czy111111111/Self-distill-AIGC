import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
from io import BytesIO
import os
from transformers import AutoImageProcessor
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "./AI-vs-Deepfake-vs-Real-v2.0"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)  

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        初始化数据集。
        :param dataframe: pandas.DataFrame，包含 'image' 和 'label' 列。
        :param transform: 图像预处理函数。
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        :param idx: 样本的索引。
        :return: 图像张量和标签张量。
        """
        image_info = self.dataframe.iloc[idx]["image"]
        label = self.dataframe.iloc[idx]["label"]

        # 从二进制数据加载图像
        image_bytes = image_info["bytes"]
        if isinstance(image_bytes, bytes):
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.fromarray(image_bytes).convert("RGB")

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        # 假设 label 已为整数 (0, 1, 2)，转换为张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def get_data_loaders(dataset_path, batch_size=32):
    """
    获取训练和测试数据加载器。
    :param dataset_path: 数据集根目录。
    :param batch_size: 每个批次的大小。
    :return: 训练和测试数据加载器。
    """
    # 加载数据集
    file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet")]
    if not file_paths:
        raise ValueError(f"No Parquet files found in {dataset_path}")

    dataset = load_dataset(file_paths)
    dataset = dataset.reset_index(drop=True)

    # 定义训练和测试的 transform
    train_transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0]  # 移除无效参数
    test_transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0]

    # 创建数据集
    train_dataset = CustomDataset(dataset, transform=train_transform)
    test_dataset = CustomDataset(dataset, transform=test_transform)

    # 划分训练和测试集
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # # 调试输出
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     print(f"Batch images shape: {images.shape}")
    #     print(f"Batch labels: {labels}")
    #     break

    return train_loader, test_loader

def load_dataset(file_paths):
    """
    从多个 Parquet 文件加载数据集。
    :param file_paths: Parquet 文件路径列表。
    :return: 合并后的 pandas.DataFrame。
    """
    data = []
    for path in tqdm(file_paths, desc="Loading Parquet Files"):
        df = pd.read_parquet(path)
        data.append(df)
    return pd.concat(data, ignore_index=True)

if __name__ == "__main__":
    dataset_path = "test-data"
    train_loader, test_loader = get_data_loaders(dataset_path, batch_size=32)
    print("数据加载器创建完成！")