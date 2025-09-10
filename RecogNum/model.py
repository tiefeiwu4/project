import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class SimpleNN:
    """简单的手写数字识别神经网络"""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        """
        初始化网络
        input_size: 28x28=784
        hidden_size: 隐藏层神经元数量
        output_size: 10个数字（0-9）
        """
        # 初始化权重和偏置
        self.input_size = input_size  # 添加这一行
        self.hidden_size = hidden_size  # 添加这一行
        self.output_size = output_size  # 添加这一行
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # 缓存用于反向传播
        self.cache = {}
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU的导数"""
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 第一层: 全连接 + ReLU
        self.cache['z1'] = np.dot(X, self.W1) + self.b1
        self.cache['a1'] = self.relu(self.cache['z1'])
        
        # 第二层: 全连接 + Softmax
        self.cache['z2'] = np.dot(self.cache['a1'], self.W2) + self.b2
        self.cache['a2'] = self.softmax(self.cache['z2'])
        
        return self.cache['a2']
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算交叉熵损失"""
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_probs) / m
        return loss
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, learning_rate: float = 0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 计算输出层的梯度
        dz2 = self.cache['a2'].copy()
        dz2[range(m), y_true] -= 1
        dz2 /= m
        
        # 第二层梯度
        dW2 = np.dot(self.cache['a1'].T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # 第一层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.cache['z1'])
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


