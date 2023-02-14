#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import scipy.interpolate
from datetime import datetime
import numpy as np
from numpy import linalg as LA
import pandas as pd
pd.options.mode.chained_assignment = None


# In[2]:


# loading data
df = pd.read_csv("A1.csv").drop(['Name'], axis=1).rename(columns={'16': '2023-01-16',
                                                                  '17': '2023-01-17',
                                                                  '18': '2023-01-18',
                                                                  '19': '2023-01-19',
                                                                  '20': '2023-01-20',
                                                                  '23': '2023-01-23',
                                                                  '24': '2023-01-24',
                                                                  '25': '2023-01-25',
                                                                  '26': '2023-01-26',
                                                                  '27': '2023-01-27'})
df['maturity date'] = pd.to_datetime(df['maturity date'], dayfirst=True)
time = ['2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
        '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27']


# In[3]:

# # generate data
data16 = df.iloc[:, 0:5]
data17 = df.iloc[:, [0, 1, 2, 3, 5]]
data18 = df.iloc[:, [0, 1, 2, 3, 6]]
data19 = df.iloc[:, [0, 1, 2, 3, 7]]
data20 = df.iloc[:, [0, 1, 2, 3, 8]]
data23 = df.iloc[:, [0, 1, 2, 3, 9]]
data24 = df.iloc[:, [0, 1, 2, 3, 10]]
data25 = df.iloc[:, [0, 1, 2, 3, 11]]
data26 = df.iloc[:, [0, 1, 2, 3, 12]]
data27 = df.iloc[:, [0, 1, 2, 3, 13]]
data = [data16, data17, data18, data19, data20,
        data23, data24, data25, data26, data27]


for d in data:
    now = d.columns.values[-1]
    now = datetime.fromisoformat(now)
    d["time to maturity"] = [(x - now).days for x in d["maturity date"]]
    lt1, lt2, lt3, lt4 = [], [], [], []
    for i, b in d.iterrows():
        lt1.append((182-b["time to maturity"] % 182) *
                   float(b["coupon"]) / 365)
    d["accrued interest"] = lt1
    for i, b in d.iterrows():
        lt2.append(b[-3] + b["accrued interest"])
    d["dirty price"] = lt2
    for i, b in d.iterrows():
        lt4.append(b["time to maturity"] / 365)
        day = np.asarray([(b["time to maturity"] % 182) / 182 +
                          n for n in range(0, int(b["time to maturity"] / 182) + 1)])
        pay = np.asarray([float(b["coupon"]) / 2] * int(
            b["time to maturity"] / 182) + [float(b["coupon"]) / 2 + 100])

        def t(y): return np.dot(pay, (1 + y / 2) ** (-day)) - b["dirty price"]
        y = optimize.fsolve(t, .05)[0]
        lt3.append(y)
    d["yield"] = lt3
    d["x"] = lt4

# In[4]:
# uninterpolated yield curve
labels = ['2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19',
          '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27']
for i in range(len(data)):
    plt.plot(data[i]["x"], data[i]["yield"], label=labels[i])
    plt.legend(loc='best')
    plt.xlim(0, 5)
    plt.xlabel('Time to Maturity')
    plt.ylabel('Yield')
    plt.title('5years Uninterpolated Yield Curve')
plt.show()


# In[5]:
# calculate interpolation
def inter(a, b):
    x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    ls = []
    inter = interp1d(a, b, bounds_error=False)
    for i in x:
        value = float(inter(i))
        ls.append(value)
    return np.asarray(x), np.asarray(ls)


def inter(a, b):
    x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    z = np.polyfit(a, b, 2)
    ls = np.poly1d(z)(x)
    return np.asarray(x), np.asarray(ls)


# interpolated yield curve
for i in range(len(data)):
    inter_res = inter(data[i]["x"], data[i]["yield"])
    plt.plot(inter_res[0], inter_res[1].squeeze(), label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Yield')
    plt.title('5years Interpolated Yield Curve')
plt.show()

# In[6]:


# calculate spot rate
def spot(df):
    npl = np.empty([1, 10])
    for i, bond in df.iterrows():
        if i == 0:
            npl[0, i] = -np.log(bond["dirty price"] /
                                (float(bond["coupon"]) / 2 + 100)) / bond["x"]
        else:
            t = np.asarray([float(bond["coupon"]) / 2]
                           * i + [float(bond["coupon"]) / 2 + 100])

            def s(y): return np.dot(t[:-1], np.exp(-(np.multiply(npl[0, :i], df["x"][:i])))
                                    ) + t[i] * np.exp(-y * bond["x"]) - bond["dirty price"]
            npl[0, i] = optimize.fsolve(s, .05)
    npl[0, 8] = (npl[0, 7] + npl[0, 9]) / 2
    return npl


# uninterpolated spot curve
for i in range(len(data)):
    plt.plot(data[i]["x"], spot(data[i]).squeeze(), label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Spot Rate')
    plt.xlim(0, 5)
    plt.title('5years Uninterpolated Spot Curve')
plt.show()

# In[7 ]
# calculate the forward rate


def forward(df):
    x, y = inter(df["x"], spot(df).squeeze())
    f1 = (y[3] * 2 - y[1] * 1)/(2-1)
    f2 = (y[5] * 3 - y[1] * 1)/(3-1)
    f3 = (y[7] * 4 - y[1] * 1)/(4-1)
    f4 = (y[9] * 5 - y[1] * 1)/(5-1)
    f = [f1, f2, f3, f4]
    return f


# forward curve
for i in range(len(data)):
    plt.plot(['1yr', '2yr', '3yr', '4yr'], forward(data[i]), label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Forward Rate')
    plt.title('1year Forward Rate Curve')
plt.show()


# In[8]:


# calculate covariance matrix
def cov(df):
    log = np.empty([5, 9])
    npl = np.empty([5, 10])
    for i in range(len(df)):
        x, y = inter(df[i]["x"], df[i]["yield"])
        npl[0, i] = y[1]
        npl[1, i] = y[3]
        npl[2, i] = y[5]
        npl[3, i] = y[7]
        npl[4, i] = y[9]
    for i in range(0, 9):
        log[0, i] = np.log(npl[0, i+1]/npl[0, i])
        log[1, i] = np.log(npl[1, i+1]/npl[1, i])
        log[2, i] = np.log(npl[2, i+1]/npl[2, i])
        log[3, i] = np.log(npl[3, i+1]/npl[3, i])
        log[4, i] = np.log(npl[4, i+1]/npl[4, i])
    return np.cov(log), log


print('covariance matrix')
print(cov(data)[0])


# In[9]:


def matrix(df):
    npl = np.empty([4, 10])
    for i in range(len(df)):
        npl[:, i] = forward(df[i])
    return npl


print("covariance matrix: ", np.cov(matrix(data)))
print('-----------------------------------------------------------------------------------------------')
w1, v1 = LA.eig(np.cov(cov(data)[1]))
print("eigenvalue of the matrix :", w1)
print('------------------------------------------------------------------------------------------------')
print("eigenvector of the matrix is: ", v1)
print('-------------------------------------------------------------------------------------------------')
w2, v2 = LA.eig(np.cov(matrix(data)))
print("eigenvalue of the matrix :", w2)
print('-------------------------------------------------------------------------------------------------')
print("eigenvector of the matrix is: ", v2)
