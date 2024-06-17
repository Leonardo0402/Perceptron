import numpy as np
import sys
from sklearn.model_selection import KFold

sys.stdout.reconfigure(encoding='utf-8')

class SimplePerceptron:
    def __init__(self, input_dim, l_rate=0.01, max_iter=1000, batch_size=10, lambda_reg=0.01, decay=0.99, early_stopping_rounds=10):
        self.b = [0.0] * input_dim  # 初始化权值向量
        self.bias = 0.0  # 初始化偏置
        self.l_rate = l_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.lambda_reg = lambda_reg  # 正则化参数
        self.decay = decay  # 学习率衰减
        self.early_stopping_rounds = early_stopping_rounds  # 早停机制的轮数
    
    def activation(self, x):
        weighted_sum = sum(b_i * x_i for b_i, x_i in zip(self.b, x)) + self.bias
        return 1 if weighted_sum > 0 else -1
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        best_b = self.b[:]
        best_bias = self.bias
        best_error = float('inf')
        no_improvement_count = 0
        
        for epoch in range(self.max_iter):
            no_errors = True
            batch_start = 0
            while batch_start < len(X_train):
                batch_end = min(batch_start + self.batch_size, len(X_train))
                batch_X = X_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]
                
                gradient_b = [0.0] * len(self.b)
                gradient_bias = 0.0
                
                for x_i, y_i in zip(batch_X, batch_y):
                    prediction = self.activation(x_i)
                    if prediction != y_i:
                        no_errors = False
                        for j in range(len(self.b)):
                            gradient_b[j] += y_i * x_i[j]
                        gradient_bias += y_i
                
                for j in range(len(self.b)):
                    self.b[j] = (1 - self.l_rate * self.lambda_reg) * self.b[j] + self.l_rate * gradient_b[j]
                self.bias += self.l_rate * gradient_bias
                
                batch_start += self.batch_size
            
            # 动态调整学习率
            self.l_rate *= self.decay
            
            # 早停机制
            if X_val is not None and y_val is not None:
                val_error = sum(1 for x_i, y_i in zip(X_val, y_val) if self.activation(x_i) != y_i)
                if val_error < best_error:
                    best_error = val_error
                    best_b = self.b[:]
                    best_bias = self.bias
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.early_stopping_rounds:
                        print(f"在第 {epoch} 轮时早停")
                        self.b = best_b
                        self.bias = best_bias
                        return "感知机模型已训练完成并早停！"
            
            if no_errors:
                break
        
        self.b = best_b
        self.bias = best_bias
        return "感知机模型已训练完成！"
    
    def predict(self, X):
        return [self.activation(x_i) for x_i in X]

    @staticmethod
    def standardize(X):
        mean = [sum(feature) / len(feature) for feature in zip(*X)]
        std = [((sum((x_i - m) ** 2 for x_i in feature) / len(feature)) ** 0.5) for feature, m in zip(zip(*X), mean)]
        return [[(x_i - m) / s for x_i, m, s in zip(x, mean, std)] for x in X]

    @staticmethod
    def normalize(X):
        min_val = [min(feature) for feature in zip(*X)]
        max_val = [max(feature) for feature in zip(*X)]
        return [[(x_i - min_v) / (max_v - min_v) for x_i, min_v, max_v in zip(x, min_val, max_val)] for x in X]

def cross_validate(model_class, X, y, k=5):
    kf = KFold(n_splits=k)
    accuracies = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = np.array(X)[train_index], np.array(X)[val_index]
        y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]
        
        model = model_class(input_dim=len(X[0]))
        model.fit(X_train.tolist(), y_train.tolist(), X_val.tolist(), y_val.tolist())
        
        predictions = model.predict(X_val.tolist())
        accuracy = sum(p == t for p, t in zip(predictions, y_val)) / len(y_val)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)

# 示例训练集（标准化处理）
X = [[1, 2], [2, 3], [1.5, 2.5], [-1, -2], [-2, -3], [-1.5, -2.5]]
y = [1, 1, 1, -1, -1, -1]

# 标准化训练数据
X_standardized = SimplePerceptron.standardize(X)

# 使用交叉验证评估模型性能
mean_accuracy, std_accuracy = cross_validate(SimplePerceptron, X_standardized, y)
print(f"平均准确率: {mean_accuracy:.2f}, 标准差: {std_accuracy:.2f}")
