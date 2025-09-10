import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Tuple
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 创建模型权重文件夹
os.makedirs('model_weight', exist_ok=True)

class SimpleNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # 使用Xavier/Glorot初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        # 数值稳定性改进
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # 使用交叉熵损失
        m = y_true.shape[0]
        # 添加小值防止log(0)
        log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        m = X.shape[0]
        
        # 将y转换为one-hot编码
        y_one_hot = np.eye(self.output_size)[y]
        
        # 输出层误差
        dz2 = self.a2 - y_one_hot
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 数据预处理函数
def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载MNIST数据"""
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(np.int32)
    
    # 分割训练集和测试集 (MNIST标准分割: 前60000训练，后10000测试)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 检查类别分布
    print("训练集类别分布:")
    for i in range(10):
        print(f"数字 {i}: {np.sum(y_train == i)} 样本")
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X: np.ndarray) -> np.ndarray:
    """数据预处理"""
    # 归一化到0-1
    X = X.astype(np.float32) / 255.0
    X = X + 0.01 * np.random.randn(*X.shape)
    # 确保值在0-1范围内
    X = np.clip(X, 0, 1)
    return X

def save_model_and_weights(model: SimpleNN, model_path: str, weights_path: str):
    """
    保存模型结构和权重
    """
    # 保存模型结构信息
    model_info = {
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'output_size': model.output_size,
        'activation': 'relu'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    # 保存权重
    weights = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2
    }
    
    np.savez(weights_path, **weights)
    print(f"模型结构已保存到: {model_path}")
    print(f"模型权重已保存到: {weights_path}")

def load_model_and_weights(model_path: str, weights_path: str) -> SimpleNN:
    """
    加载模型结构和权重
    """
    # 加载模型结构
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    # 创建模型实例
    model = SimpleNN(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        output_size=model_info['output_size']
    )
    
    # 加载权重
    weights = np.load(weights_path)
    model.W1 = weights['W1']
    model.b1 = weights['b1']
    model.W2 = weights['W2']
    model.b2 = weights['b2']
    
    print("模型和权重加载成功!")
    return model

# 训练函数
def train_model():
    """训练模型"""
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist()
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # 创建模型
    model = SimpleNN(input_size=784, hidden_size=256, output_size=10)  # 增加隐藏层大小
    epochs = 20 
    batch_size = 128  
    learning_rate = 0.01  
    decay_rate = 0.95
    decay_step = 10
    
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    train_losses = []
    test_accuracies = []
    
    print("开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(n_batches):
            # 获取批次数据
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            

            y_pred = model.forward(X_batch)
            
           
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            
           
            model.backward(X_batch, y_batch, learning_rate)
        
        
        if (epoch + 1) % decay_step == 0:
            learning_rate *= decay_rate
            print(f"Epoch {epoch+1}: 学习率衰减至 {learning_rate:.6f}")
        
        # 计算平均损失和测试准确率
        avg_loss = epoch_loss / n_batches
        test_acc = model.accuracy(X_test, y_test)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)
        
        # 检查预测分布
        if epoch == 0 or (epoch + 1) % 10 == 0:
            predictions = model.predict(X_test[:1000])
            pred_counts = np.bincount(predictions, minlength=10)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
            print(f"预测分布: {pred_counts}")
    
    return model, train_losses, test_accuracies

# 可视化训练过程


# 使用示例
if __name__ == "__main__":
    # 训练模型
    model, train_losses, test_accuracies = train_model()
    
  
    final_acc = test_accuracies[-1]
    print(f"最终测试准确率: {final_acc:.4f}")
    
    # 保存模型和权重
    model_path = 'model_weight/model_structure.pkl'
    weights_path = 'model_weight/model_weights.npz'
    save_model_and_weights(model, model_path, weights_path)
    
    # 示例：如何加载模型和权重
    print("\n示例：加载保存的模型和权重...")
    loaded_model = load_model_and_weights(model_path, weights_path)
    
    # 验证加载的模型
    X_test, y_test = load_mnist()[2:]  # 获取测试数据
    X_test = preprocess_data(X_test)
    loaded_acc = loaded_model.accuracy(X_test, y_test)
    print(f"加载模型的测试准确率: {loaded_acc:.4f}")
    
    # 检查预测分布
    predictions = loaded_model.predict(X_test)
    pred_counts = np.bincount(predictions, minlength=10)
    print(f"测试集预测分布: {pred_counts}")
    
    # 检查每个类别的准确率
    print("\n每个类别的准确率:")
    for i in range(10):
        idx = y_test == i
        if np.any(idx):
            acc = np.mean(predictions[idx] == y_test[idx])
            print(f"数字 {i}: {acc:.4f}")