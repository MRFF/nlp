# 导入数据集
from sklearn.datasets import load_boston
import random

data = load_boston() # 波士顿房价数据集
x, y = data['data'], data['target']
x_rm = x[:,5]

def price(rm, k, b):
    """f(x) = kx + b"""
    return k * rm + b

# 以方差计算损失
def loss_by_variance(y, y_hat):
    return sum((y_i - y_hat_i) ** 2 for y_i, y_hat_i in zip(list(y), list(y_hat))) / len(list(y))

# 以绝对值计算损失
def loss_by_abs(y, y_hat):
    return sum(abs(y_i - y_hat_i) for y_i, y_hat_i in zip(list(y), list(y_hat))) / len(list(y))

# k的偏导
def partial_k(x, y, y_hat):
    n = len(y)
    return  (-2 / n) * sum( (y_i - y_hat_i) * x_i for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)))

# b的偏导
def partial_b(x, y, y_hat):
    n = len(y)
    return (-2 / n) * sum( (y_i - y_hat_i) for y_i, y_hat_i in zip(list(y), list(y_hat)) )


min_loss = float('inf') # 正无穷
trying_times = 10000
current_k = random.random() * 200 - 100
current_b = random.random() * 200 - 100

# 学习速率
learning_rate = 0.001

for i in range(trying_times):
    
    # 当前k和b计算而来的y值，即y_hat
    price_by_kb = [price(r, current_k, current_b) for r in x_rm]
    # 当前函数的损失值
    current_loss = loss_by_variance(y, price_by_kb)
    
    if current_loss < min_loss:
        min_loss = current_loss
        print('Get best k {} and b {} and the loss is {} when trying {} times'.format(current_k, current_b, min_loss, i+1)) 
   
    # 注意这里的第一个参数是x_rm，
    k_direction, b_direction = partial_k(x_rm, y, price_by_kb), partial_b(x_rm,y, price_by_kb)    
    # 为什么要乘以一个很小的系数
    # 考虑这样的情况，如果函数陡峭，某一点的导数可能很大，向反方向变化，有可能仍然和最小点距离很远，所以就要一点点变
    # 这就是学习速率
    # 其实还有优化空间：让学习速率本身也是可变化的，导数大的时候，速率就小些；等导数小了，学习速率再变大些
    current_k = current_k + (-1 * k_direction) * learning_rate
    current_b = current_b + (-1 * b_direction) * learning_rate