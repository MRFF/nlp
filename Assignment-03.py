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
    # current_loss = loss_by_variance(y, price_by_kb)
    urrent_loss = loss_by_abs(y, price_by_kb)
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
    
    “”“
    Output Example:
    Get best k 18.194797252558786 and b -43.65048566637728 and the loss is 48.20222667554089 when trying 1 times
    Get best k 17.58044666240098 and b -43.74681437936495 and the loss is 44.266406122937894 when trying 2 times
    Get best k 17.016441784062025 and b -43.835228497237054 and the loss is 40.65789027035569 when trying 3 times
    Get best k 16.498656554844505 and b -43.916376657967334 and the loss is 37.348978820462925 when trying 4 times
    Get best k 16.023303061877076 and b -43.990854340662544 and the loss is 34.31124804884696 when trying 5 times
    Get best k 15.586903829180072 and b -44.05920822217626 and the loss is 31.532300123782623 when trying 6 times
    Get best k 15.186266375933595 and b -44.12194017667811 and the loss is 28.981629887942045 when trying 7 times
    Get best k 14.818459859812647 and b -44.17951094743989 and the loss is 26.648172739991107 when trying 8 times
    Get best k 14.480793634508409 and b -44.23234351770193 and the loss is 24.50695440418095 when trying 9 times
    Get best k 14.170797564559347 and b -44.28082620528146 and the loss is 22.54121891439085 when trying 10 times
    Get best k 13.886203953472492 and b -44.32531550356353 and the loss is 20.745232936414038 when trying 11 times
    ......
    Get best k 10.357819743490278 and b -42.658345121517215 and the loss is 4.453917286565011 when trying 9997 times
    Get best k 10.35778954712934 and b -42.65815303890182 and the loss is 4.453915949509992 when trying 9998 times
    Get best k 10.35775935149454 and b -42.657960960905484 and the loss is 4.453914612487127 when trying 9999 times
    Get best k 10.357729156585862 and b -42.657768887528086 and the loss is 4.453913275496415 when trying 10000 times
    ”“”
