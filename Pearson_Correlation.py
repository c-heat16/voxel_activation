import math
import numpy as np
import numpy.random as random


def findMean(x, y):
    Xsum = 0
    Ysum = 0
    for i in range(len(x)):
        Xsum += x[i]
    Xmean = Xsum / float(len(x))
    for i in range(len(y)):
        Ysum += x[i]
    Ymean = Ysum / float(len(y))
    return Xmean, Ymean


def nem(x, y, n):
    product = 1
    list = []
    productSum = 0
    Xmean, Ymean = findMean(x, y)
    # if n <=0 or len(x) <=0 or len(y) <=0:
    # return "error"
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            product = (x[i, j] - Xmean[j]) * (y[i, j] - Ymean[i])
            list.append(product)
    for i in range(len(list)):
        productSum += list[i]
    return productSum


def dem(x, y, n):
    Xmean, Ymean = findMean(x, y)
    XSS = np.zeros((x.shape[1],), dtype=np.float32)
    for i in range(n):
        for j in range(x.shape[1]):
            XSS[j] += (x[i, j] - Xmean[j]) * (x[i, j] - Xmean[j])
    YSS = np.zeros((x.shape[1],), dtype=np.float32)
    for i in range(n):
        for j in range(x.shape[1]):
            YSS[j] += (y[i, j] - Ymean[i]) * (y[i, j] - Ymean[i])
    squareProduct = np.zeros((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            squareProduct[i, j] = np.sqrt(XSS[i]) * np.sqrt(YSS[j])
    # squareProduct = np.sqrt(XSS) * np.sqrt(YSS)
    return squareProduct


def calc(x, y, n):
    # print("test")
    productSum = nem(x, y, n)
    squareProduct = dem(x, y, n)
    r = productSum / squareProduct  # upper is a summation and lower is square root
    print(productSum, squareProduct)
    return r


x = np.random.rand(100, 10000)
y = np.random.rand(100, 10000)
n = 100
# print(nem(x,y,n),dem(x,y,n))
print(findMean(x, y))
print(calc(x, y, n))