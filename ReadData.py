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
# plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
# plt.show()

# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
# plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red') 
# plt.show()


## 2.3.6 特征分为类别特征和数字特征，并对类别特征查看unique分布
# 分离label即预测值
Y_train = Train_data['price']

### 这个区别方式适用于没有直接label coding的数据
### 这里不适用，需要人为根据实际含义来区分
### 数字特征
### numeric_features = Train_data.select_dtypes(include=[np.number])
### numeric_features.columns
### # 类型特征
### categorical_features = Train_data.select_dtypes(include=[np.object])
### categorical_features.columns

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

# categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

# 特征nunique分布
#for cat_fea in categorical_features:
#    print(cat_fea + "的特征分布如下：")
#    print("{}特征有个{}不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
#    print(Train_data[cat_fea].value_counts())

# 特征nunique分布
#for cat_fea in categorical_features:
#    print(cat_fea + "的特征分布如下：")
#    print("{}特征有个{}不同的值".format(cat_fea, Test_data[cat_fea].nunique()))
#    print(Test_data[cat_fea].value_counts())


## 2.3.7 数字特征分析
# numeric_features.append('price')
# print(numeric_features)

## 1) 相关性分析
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
# print(correlation['price'].sort_values(ascending = False),'\n')

# f , ax = plt.subplots(figsize = (7, 7))

# plt.title('Correlation of Numeric Features with Price',y=1,size=16)

# sns.heatmap(correlation,square = True,  vmax=0.8)
# plt.show()

# del price_numeric['price']

## 2) 查看几个特征得 偏度和峰值
# for col in numeric_features:
#    print('{:15}'.format(col), 
#          'Skewness: {:05.2f}'.format(Train_data[col].skew()) , 
#          '   ' ,
#          'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())  
#         )


## 3) 每个数字特征得分布可视化
# f = pd.melt(Train_data, value_vars=numeric_features)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
# g = g.map(sns.distplot, "value")
# plt.show()


## 4) 数字特征相互之间的关系可视化
# sns.set()
# columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
# sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
# plt.show()

# print(Train_data.columns)
# print(Y_train)

## 5) 多变量互相回归关系可视化
#fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']

#v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
#sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

#v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
#sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

#v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
#sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

#power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
#sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

#v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
#sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

#v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
#sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

#v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
#sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

#v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
#sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

#v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
#sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

#v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
#sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
#plt.show()