
"""
线性回归：y=w*x+b
误差函数：loss=累加(w*x+b-y)^2
找到w,b使loss最小
->计算梯度每次更新w,b的值，会越来越接近最低点
反复1000次
"""

import numpy as np

#1.loss函数的计算
def loss_function(points, b, w):
    loss = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        loss += ((w * x + b) - y) ** 2
    return loss/float(len(points))

#2.更新b,w的值
def update_bw(points, b_current, w_current, learning_rate):
    N = float(len(points))
    b_gradient = 0
    w_gradient = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    new_b = b_current - learning_rate * b_gradient
    new_w = w_current - learning_rate * w_gradient
    return [new_b, new_w]


#3.每次带入更新的b,w，逼近理想值

def loop_bw(points, o_b, o_w, learning_rate, number):
    b = o_b
    w = o_w
    for i in range(number):
        [b, w] = update_bw(points, b, w, learning_rate)
    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    b = 0
    w = 0
    learning_rate = 0.0001
    [new_b, new_w] = loop_bw(points, b, w, learning_rate, 1000)
    print("[b, w] = ", new_b, new_w)
    print("1000次后loss函数：", loss_function(points, new_b, new_w))

if __name__ == '__main__':
    run()
