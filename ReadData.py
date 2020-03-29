#coding:utf-8

## 2.3.1 载入各种数据科学以及可视化库
## 导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


## 2.3.2 载入数据
## 1) 载入训练集和测试集；
path = 'C:/Users/Huang Qiang/Desktop/task1/'
Train_data = pd.read_csv(path+'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(path+'used_car_testA_20200313.csv', sep=' ')

## 2) 简略观察数据(head()+shape)
# print(Train_data.head().append(Train_data.tail()))    #append
# print(Train_data.shape)

# print(Test_data.head().append(Train_data.tail()))
# print(Test_data.shape)


## 2.3.3 总览数据概况
## 1) 通过describe()来熟悉数据的相关统计量
# print(Train_data.describe())
# print(Test_data.describe())

## 2) 通过info()来熟悉数据类型
# print(Train_data.info())
# print(Test_data.info())


## 2.3.4 判断数据缺失和异常
## 1) 查看每列的存在nan情况,数据缺失为nan
# print(Train_data.isnull().sum())
# print(Test_data.isnull().sum())

# nan可视化
# missing = Train_data.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.show()

# 可视化看下缺省值
# msno.matrix(Train_data.sample(250))
# msno.bar(Train_data.sample(1000))
# plt.show()

# msno.matrix(Test_data.sample(250))
# msno.bar(Test_data.sample(1000))
# plt.show()

## 2) 查看异常值检测
# Train_data.info()
# print(Train_data['notRepairedDamage'].value_counts())

# '-'也为空缺值，因为很多模型对nan有直接的处理，这里我们先不做处理，先替换成nan
# Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
# print(Train_data['notRepairedDamage'].value_counts())

# Train_data.isnull().sum()
# Test_data['notRepairedDamage'].value_counts()
# Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

# Train_data.isnull().sum()

# Test_data['notRepairedDamage'].value_counts()
# Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

# 以下两个类别特征严重倾斜，一般不会对预测有什么帮助，故这边先删掉
# 当然也可以继续挖掘，但是一般意义不大
# Train_data["seller"].value_counts()
# Train_data["offerType"].value_counts()
# del Train_data["seller"]
# del Train_data["offerType"]
# del Test_data["seller"]
# del Test_data["offerType"]


## 2.3.5 了解预测值的分布
# print(Train_data['price'])
# print(Train_data['price'].value_counts())

## 1) 总体分布概况（无界约翰逊分布等）
import scipy.stats as st
# y = Train_data['price']
# plt.figure(1); plt.title('Johnson SU')
# sns.distplot(y, kde=False, fit=st.johnsonsu)
# plt.figure(2); plt.title('Normal')
# sns.distplot(y, kde=False, fit=st.norm)
# plt.figure(3); plt.title('Log Normal')
# sns.distplot(y, kde=False, fit=st.lognorm)
# plt.show()

## 2) 查看skewness and kurtosis
# sns.distplot(Train_data['price']);
# print("Skewness: %f" % Train_data['price'].skew())
# print("Kurtosis: %f" % Train_data['price'].kurt())
# plt.show()

# print(Train_data.skew(), Train_data.kurt())

# skew、kurt说明参考https://www.cnblogs.com/wyy1480/p/10474046.html
# sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
# plt.show()

# sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')
# plt.show()

## 3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()

# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red') 
plt.show()