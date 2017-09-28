import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from sklearn.metrics import roc_curve,roc_auc_score
import sqlite3
import pandas as pd
def loadData(misRate=None,method=None):			# load from sqlite given misRate and method
	'''
		misRate:(number) missing rate of X, 0 means origin X
		method:(string) 'bpca','ppca','mean'
	'''
	print('load_data ~')
	con = sqlite3.connect(r'accDetect.db')
	cur = con.cursor()
	if misRate is None:	# return nonmissing X
		cur.execute('SELECT X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,Y')
		result = cur.fetchall()
		print('len(realX):')
		print(len(result))
		con.close()
		return result
	else:
		cur.execute('SELECT X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,Y  FROM fullX WHERE method=? and misRate=?',(method,misRate))
		result = cur.fetchall()
		# result = list(map(lambda x:x[:-2],result))	# delete misRate and method col
		print('len(fullX):')
		print(len(result))
		con.close()
		return result

## main
if __name__=='__main__':
	misRate = 0.1
	method = 'bpca'
	dataList = loadData(misRate=misRate,method=method)
	dataF = pd.DataFrame(dataList,columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9',
											'X10','X11','X12','X13','X14','X15','X16','X17',
											'X18','X19','X20','X21','X22','X23','X24','Y'])
	XDf = dataF.ix[:,0:24]	# from 0 to 23 ,slice as list 
	yDf = dataF.ix[:,24]
	X_train,X_test,y_train,y_test=train_test_split(XDf, yDf,test_size=0.25,
		random_state=0,stratify=yDf)	# all result is Df or Series
# svm roc
gammas=range(1,20)
train_scores=[]
test_scores=[]
for gamma in gammas:
	cls=svm.SVC(kernel='rbf',gamma=gamma)
	cls.fit(X_train,y_train)
	train_scores.append(cls.score(X_train,y_train))
	test_scores.append(cls.score(X_test, y_test))
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(gammas,train_scores,label="Training score ",marker='+' )
ax.plot(gammas,test_scores,label= " Testing  score ",marker='o' )
ax.set_title( "SVC_rbf")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel("score")
ax.set_ylim(0,1.05)
ax.legend(loc="best",framealpha=0.5)
plt.show()

gamma = 5
cls = svm.SVC(kernel='rbf',gamma=gamma,probability=True)
cls.fit(X_train,y_train.ravel())
y_score = cls.predict_proba(X_test) # 2 cols
# draw roc curve for svm
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
roc = roc_auc_score(y_test,y_score[:,1])
print('roc:')
print(roc)	# 0.796
ax.plot([0,1],[0,1],'k--')
ax.plot(fpr,tpr,'k--')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC')
ax.legend(loc='best')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
plt.show()

# rf roc
nTree = 100
clf=ensemble.RandomForestClassifier(n_estimators=nTree)
clf.fit(X_train,y_train.ravel())
y_score = clf.predict_proba(X_test) # 2 cols
# draw roc curve for rf
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
roc = roc_auc_score(y_test,y_score[:,1])
print('roc:')
print(roc)	# 0.803
ax.plot([0,1],[0,1],'k--')
ax.plot(fpr,tpr,'k--')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC')
ax.legend(loc='best')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
plt.show()

# draw roc curve for xgboost
from xgboost import XGBClassifier
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
gsearch1.fit(X_train, y_train.ravel())
y_score = gsearch1.predict_proba(X_test) # 2 cols
# draw roc curve for rf
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
roc = roc_auc_score(y_test,y_score[:,1])
print('roc:')
print(roc)	# 0.788
ax.plot([0,1],[0,1],'k--')
ax.plot(fpr,tpr,'k--')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC')
ax.legend(loc='best')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
plt.show()
print('gsearch1.grid_scores_')
print(gsearch1.grid_scores_)
print('gsearch1.best_params_')
print(gsearch1.best_params_)
print('gsearch1.best_score_')
print(gsearch1.best_score_)
print('gsearch1.test_score_')
print(gsearch1.score(X_test,y_test.ravel()))