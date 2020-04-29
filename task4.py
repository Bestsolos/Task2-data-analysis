<<<<<<< HEAD
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]


## 4.4.2 线性回归 & 五折交叉验证 & 模拟真实业务情况
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']

# 4.4.2 - 1 简单建模
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
# 查看训练的线性回归模型的截距（intercept）与权重(coef)
# print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)

from matplotlib import pyplot as plt

subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
# plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
# plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
# plt.xlabel('v_9')
# plt.ylabel('price')
# plt.legend(['True Price','Predicted Price'],loc='upper right')
# print('The predicted price is obvious different from true price')
# plt.show()

import seaborn as sns
# print('It is clear to see the price shows a typical exponential distribution')
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.distplot(train_y)
# plt.subplot(1,2,2)
# sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
# plt.show()

# 如果使用train_y_ln值，在预测完结果后需要逆变换成train_y
train_y_ln = np.log(train_y + 1)
# print('The transformed price seems like normal distribution')
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.distplot(train_y_ln)
# plt.subplot(1,2,2)
# sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
# plt.show()

model = model.fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)

# plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
# plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
# plt.xlabel('v_9')
# plt.ylabel('price')
# plt.legend(['True Price','Predicted Price'],loc='upper right')
# print('The predicted price seems normal after np.log transforming')
# plt.show()

# 4.4.2 - 2 五折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer

def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper

scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))
print('AVG:', np.mean(scores))

# scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
# print('AVG:', np.mean(scores))

# scores = pd.DataFrame(scores.reshape(1,-1))
# scores.columns = ['cv' + str(x) for x in range(1, 6)]
# scores.index = ['MAE']


# 4.4.2 - 3 模拟真实业务情况
import datetime

sample_feature = sample_feature.reset_index(drop=True)
# 选用靠前时间的4/5样本当作训练集，靠后时间的1/5当作验证集
split_point = len(sample_feature) // 5 * 4
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)

model = model.fit(train_X, train_y_ln)
print(mean_absolute_error(val_y_ln, model.predict(val_X)))

# 4.4.2 - 4 绘制学习率曲线与验证曲线
from sklearn.model_selection import learning_curve, validation_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size,
                scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean,'o-',color="g", label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt

# plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)
# plt.show()


## 4.4.3 多种模型对比
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)

# 4.4.3 - 1 线性模型 & 嵌入式特征选择
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

models = [LinearRegression(), Ridge(), Lasso()]
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')

# result = pd.DataFrame(result)
# result.index = ['cv' + str(x) for x in range(1, 6)]

# model = LinearRegression().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)
# plt.show()

# model = Ridge().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)
# plt.show()

# model = Lasso().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)
# plt.show()

# 4.4.3 - 2 非线性模型
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100), 
          XGBRegressor(n_estimators = 100, objective='reg:squarederror'), 
          LGBMRegressor(n_estimators = 100)]

for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')

# result = pd.DataFrame(result)
# result.index = ['cv' + str(x) for x in range(1, 6)]


## 4.4.4 模型调参

## LGB的参数集合：
objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []

# 4.4.4 - 1 贪心调参
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], 
            y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
# plt.show()

# 4.4.4 - 2 Grid Search 调参
from sklearn.model_selection import GridSearchCV

parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
# print(clf.best_params_)   # {'max_depth': 15, 'num_leaves': 55, 'objective': 'regression'}
model = LGBMRegressor(objective='regression', num_leaves=55, max_depth=15)

np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))  # 0.13626164479243302

# 4.4.4 - 3 贝叶斯调参
from bayes_opt import BayesianOptimization

def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val

rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
print(rf_bo.maximize())
print(1 - rf_bo.max['target'])  # 0.1296693644053145

# 总结
plt.figure(figsize=(13,5))
sns.lineplot(x=['0_origin','1_log_transfer','2_L1_&_L2','3_change_model','4_parameter_turning'], y=[1.36 ,0.19, 0.19, 0.14, 0.13])
# plt.show()
=======
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]


## 4.4.2 线性回归 & 五折交叉验证 & 模拟真实业务情况
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']

# 4.4.2 - 1 简单建模
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
# 查看训练的线性回归模型的截距（intercept）与权重(coef)
# print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)

from matplotlib import pyplot as plt

subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
# plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
# plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
# plt.xlabel('v_9')
# plt.ylabel('price')
# plt.legend(['True Price','Predicted Price'],loc='upper right')
# print('The predicted price is obvious different from true price')
# plt.show()

import seaborn as sns
# print('It is clear to see the price shows a typical exponential distribution')
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.distplot(train_y)
# plt.subplot(1,2,2)
# sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
# plt.show()

train_y_ln = np.log(train_y + 1)
# print('The transformed price seems like normal distribution')
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.distplot(train_y_ln)
# plt.subplot(1,2,2)
# sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
# plt.show()

model = model.fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)

# plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
# plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
# plt.xlabel('v_9')
# plt.ylabel('price')
# plt.legend(['True Price','Predicted Price'],loc='upper right')
# print('The predicted price seems normal after np.log transforming')
# plt.show()

# 4.4.2 - 2 五折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer

def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper

# scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))
# print('AVG:', np.mean(scores))

scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
# print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']


# 4.4.2 - 3 模拟真实业务情况
import datetime

sample_feature = sample_feature.reset_index(drop=True)
split_point = len(sample_feature) // 5 * 4

train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)

model = model.fit(train_X, train_y_ln)
print(mean_absolute_error(val_y_ln, model.predict(val_X)))

# 4.4.2 - 4 绘制学习率曲线与验证曲线
from sklearn.model_selection import learning_curve, validation_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt

# plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)
>>>>>>> 6f71a354596e06e7b8729dfffb5d5649f18e4a32
