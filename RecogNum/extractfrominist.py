import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pickle
import gzip
import os
from urllib.request import urlretrieve

def download_mnist():
    """下载MNIST数据集"""
    url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    
    if not os.path.exists(filename):
        print("正在下载MNIST数据集...")
        urlretrieve(url, filename)
        print("下载完成!")
    return filename

def load_mnist():
    """加载MNIST数据集"""
    filename = download_mnist()
    
    with gzip.open(filename, 'rb') as f:
        # 对于Python 3.x需要指定encoding
        data = pickle.load(f, encoding='latin1')
    
    # 数据集包含训练集、验证集、测试集
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def extract_mnist_samples(num_samples=10, dataset='train', random_seed=None):
    """
    从MNIST数据集中提取手写数字样本
    
    参数:
    num_samples: 要提取的样本数量
    dataset: 'train', 'val', 或 'test'
    random_seed: 随机种子，用于可重复性
    
    返回:
    images: 提取的数字图像数组 (28x28)
    labels: 对应的标签数组
    """
    # 加载数据集
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    
    # 选择数据集
    if dataset == 'train':
        images, labels = x_train, y_train
    elif dataset == 'val':
        images, labels = x_val, y_val
    else:
        images, labels = x_test, y_test
    
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 随机选择样本
    total_samples = len(images)
    indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # 提取选中的样本并重塑为28x28图像
    selected_images = images[indices].reshape(-1, 28, 28)
    selected_labels = labels[indices]
    
    return selected_images, selected_labels

def extract_specific_digits(digits_to_extract, num_samples_per_digit=5, dataset='train', random_seed=None):
    """
    提取特定数字的样本
    
    参数:
    digits_to_extract: 要提取的数字列表，如 [0, 1, 2]
    num_samples_per_digit: 每个数字提取的样本数量
    dataset: 'train', 'val', 或 'test'
    random_seed: 随机种子
    
    返回:
    images: 提取的图像数组 (28x28)
    labels: 对应的标签数组
    """
    # 加载数据集
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    
    # 选择数据集
    if dataset == 'train':
        images, labels = x_train, y_train
    elif dataset == 'val':
        images, labels = x_val, y_val
    else:
        images, labels = x_test, y_test
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    selected_images = []
    selected_labels = []
    
    for digit in digits_to_extract:
        # 找到该数字的所有索引
        digit_indices = np.where(labels == digit)[0]
        
        # 随机选择指定数量的样本
        if len(digit_indices) > 0:
            chosen_indices = np.random.choice(
                digit_indices, 
                min(num_samples_per_digit, len(digit_indices)), 
                replace=False
            )
            
            selected_images.extend(images[chosen_indices])
            selected_labels.extend(labels[chosen_indices])
    
    # 重塑为28x28图像
    selected_images = np.array(selected_images).reshape(-1, 28, 28)
    selected_labels = np.array(selected_labels)
    
    return selected_images, selected_labels

def display_mnist_samples(images, labels, num_cols=5, figsize=(12, 8)):
    """
    显示提取的MNIST样本
    
    参数:
    images: 图像数组
    labels: 标签数组
    num_cols: 每行显示的图像数量
    figsize: 图像大小
    """
    num_samples = len(images)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    plt.figure(figsize=figsize)
    for i in range(num_samples):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_mnist_samples(images, labels, filename_prefix='mnist_samples'):
    """
    保存提取的MNIST样本为图像文件
    
    参数:
    images: 图像数组
    labels: 标签数组
    filename_prefix: 文件名前缀
    """
    os.makedirs('mnist_samples', exist_ok=True)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        filename = f'mnist_samples/{filename_prefix}_{i}_label_{label}.png'
        plt.imsave(filename, image, cmap='gray')
        print(f"已保存: {filename}")

# 使用示例
if __name__ == "__main__":
    print("MNIST手写数字提取工具")
    print("=" * 50)
    
    # 示例1: 提取随机样本
    print("1. 提取10个随机样本:")
    images, labels = extract_mnist_samples(num_samples=10, random_seed=42)
    display_mnist_samples(images, labels)
    
    print(f"提取了 {len(images)} 个样本")
    print(f"图像形状: {images[0].shape}")
    print(f"标签: {labels}")
    print()
    
    # 示例2: 提取特定数字
    print("2. 提取数字 3, 7, 9 的各3个样本:")
    specific_images, specific_labels = extract_specific_digits(
        digits_to_extract=[3, 7, 9],
        num_samples_per_digit=3,
        random_seed=42
    )
    display_mnist_samples(specific_images, specific_labels, num_cols=3)
    
    print(f"提取的特定数字样本标签: {specific_labels}")
    print()
    
    # 示例3: 保存样本到文件
    print("3. 保存样本到文件...")
    save_mnist_samples(images, labels, 'random_samples')
    print("样本已保存到 mnist_samples/ 目录")