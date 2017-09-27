#
# xgboost调参详解
# 第一步：确定学习速率和tree_based 参数调优的估计器数目

from xgboost import XGBClassifier

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble

def load_data():
	print('load_data ~')
	mat_data = sio.loadmat('finaldata.mat')

	X = mat_data['X']
	X_normed = X/X.max(axis=0)
	print('X.shape:')
	print(X.shape)
	Y = mat_data['Y']
	print('Y.shape:')
	print(Y.shape)

	print('end load_data ~')
	return cross_validation.train_test_split(X_normed, Y,test_size=0.25,
		random_state=0,stratify=Y)


## main
X_train,X_test,y_train,y_test = load_data()
XTrain = pd.DataFrame(X_train)
yTrain = pd.Series(y_train.ravel())
XTest = pd.DataFrame(X_test)
yTest = pd.Series(y_test.ravel())
xgb1 = XGBClassifier(
	learning_rate=0.1,
	n_estimators=1000,
	max_depth=5,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27)

# 第二步： max_depth 和 min_weight 参数调优
# grid_search参考 :
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
# http://blog.csdn.net/abcjennifer/article/details/23884761

# 网格搜索scoring=’roc_auc’只支持二分类，多分类需要修改scoring(默认支持多分类)

param_test1 = {
	'max_depth': list(range(3, 10, 2)),
	'min_child_weight': list(range(1, 6, 2))
}
param_test2 = {
'max_depth': [4, 5, 6],
'min_child_weight': [4, 5, 6]
}
from sklearn import svm, grid_search, datasets
from sklearn import grid_search

gsearch1 = grid_search.GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=140, max_depth=5,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test1,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)
gsearch1.fit(XTrain, yTrain)
print('gsearch1.grid_scores_')
print(gsearch1.grid_scores_)
print('gsearch1.best_params_')
print(gsearch1.best_params_)
print('gsearch1.best_score_')
print(gsearch1.best_score_)
print('gsearch1.test_score_')
print(gsearch1.score(XTest,yTest))

# 第三步：gamma参数调优

param_test3 = {
'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch3 = grid_search.GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=140,
	max_depth=4,
	min_child_weight=6,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test3,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)
gsearch3.fit(XTrain, yTrain)
print('gsearch1.grid_scores_')
print(gsearch3.grid_scores_)
print('gsearch1.best_params_')
print(gsearch3.best_params_)
print('gsearch1.best_score_')
print(gsearch3.best_score_)
print('gsearch1.test_score_')
print(gsearch3.score(XTest,yTest))

# 第四步：调整subsample 和 colsample_bytree 参数
# 取0.6,0.7,0.8,0.9作为起始值

param_test4 = {
'subsample': [i / 10.0 for i in range(6, 10)],
'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}

gsearch6 = grid_search.GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=177,
	max_depth=3,
	min_child_weight=4,
	gamma=0.1,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test4,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)

gsearch6.fit(XTrain, yTrain)
print('gsearch6.grid_scores_')
print(gsearch3.grid_scores_)
print('gsearch6.best_params_')
print(gsearch3.best_params_)
print('gsearch6.best_score_')
print(gsearch3.best_score_)
print('gsearch6.test_score_')
print(gsearch3.score(XTest,yTest))

# 第六步：降低学习速率

xgb4 = XGBClassifier(
learning_rate = 0.01,
n_estimators = 5000,
max_depth = 4,
min_child_weight = 6,
gamma = 0,
subsample = 0.8,
colsample_bytree = 0.8,
reg_alpha = 0.005,
objective = 'binary:logistic',
nthread = 4,
scale_pos_weight = 1,
seed = 27)
gsearch6.fit(XTrain, yTrain)
print('gsearch6.grid_scores_')
print(gsearch3.grid_scores_)
print('gsearch6.best_params_')
print(gsearch3.best_params_)
print('gsearch6.best_score_')
print(gsearch3.best_score_)
print('gsearch6.test_score_')
print(gsearch3.score(XTest,yTest))

# 第六步：正则化参数调优
param_test6 = {
'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=177,
	max_depth=4,
	min_child_weight=6,
	gamma=0.1,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test6,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)
gsearch6.fit(XTrain, yTrain)
print('gsearch6.grid_scores_')
print(gsearch3.grid_scores_)
print('gsearch6.best_params_')
print(gsearch3.best_params_)
print('gsearch6.best_score_')
print(gsearch3.best_score_)
print('gsearch6.test_score_')
print(gsearch3.score(XTest,yTest))
# python示例
import xgboost as xgb
import pandas as pd
# 获取数据
from sklearn import cross_validation
from sklearn.datasets import load_iris

iris = load_iris()
# 切分数据集
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.33,
																	 random_state=42)
# 设置参数
m_class = xgb.XGBClassifier(
learning_rate = 0.1,
n_estimators = 1000,
max_depth = 5,
gamma = 0,
subsample = 0.8,
colsample_bytree = 0.8,
objective = 'binary:logistic',
nthread = 4,
seed = 27)
# 训练
m_class.fit(X_train, y_train)
test_21 = m_class.predict(X_test)
print("Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21))
# 预测概率
# test_2 = m_class.predict_proba(X_test)
# 查看AUC评价标准
from sklearn import metrics

print("Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21))
##必须二分类才能计算
##print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)
# 查看重要程度
feat_imp = pd.Series(m_class.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
import matplotlib.pyplot as plt

plt.show()
# 回归
# m_regress = xgb.XGBRegressor(n_estimators=1000,seed=0)
# m_regress.fit(X_train, y_train)
# test_1 = m_regress.predict(X_test)