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
path = 'C:/Users/Huang Qiang/Desktop/task2/'
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
## 1) 查看每列的存在nan情况
# print(Train_data.isnull().sum())
# print(Test_data.isnull().sum())

# nan可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()