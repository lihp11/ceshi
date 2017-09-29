import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
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
		cur.execute('SELECT X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,Y FROM realX')
		result = cur.fetchall()
		print('len(realX):')
		print(len(result))
		con.close()
		realDataF = pd.DataFrame(result,columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9',
										'X10','X11','X12','X13','X14','X15','X16','X17',
										'X18','X19','X20','X21','X22','X23','X24','Y'])
		realXDf = realDataF.iloc[:,0:24]	# from 0 to 23 ,slice as list
		realyDf = realDataF.iloc[:,24]
		X_train,X_test,y_train,y_test=train_test_split(realXDf, realyDf,test_size=0.25,
			random_state=0,stratify=realyDf)
		return X_train,X_test,y_train,y_test
	else:
		cur.execute('SELECT X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,Y  FROM fullX WHERE method=? and abs(misRate-?)<0.0001',(method,misRate))
		result = cur.fetchall()
		# result = list(map(lambda x:x[:-2],result))	# delete misRate and method col
		print('len(fullX):')
		print(len(result))
		con.close()
		dataF = pd.DataFrame(result,columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9',
												'X10','X11','X12','X13','X14','X15','X16','X17',
												'X18','X19','X20','X21','X22','X23','X24','Y'])
		XDf = dataF.iloc[:,0:24]	# from 0 to 23 ,slice as list
		yDf = dataF.iloc[:,24]
		X_train,X_test,y_train,y_test=train_test_split(XDf, yDf,test_size=0.25,
			random_state=0,stratify=yDf)	# all result is Df or Series
		return X_train,X_test,y_train,y_test


###################################################
# global var and add mlMethodRoc: list of dicts with 7 fields
	# (mlMethod,
	# compMethod,
	# misRate,
	# maxRoc,
	# paramDict(dict),
	# fpr(list),
	# tpr(list))
methods = [None,'bpca','ppca','mean']
misRates = np.linspace(0.05,0.6,12)	#np.array with 12 elems
resultList = []	#list of result dict containing 7 fields
#############
# svc roc
resultList = []	#list of result dict containing 7 fields
# gammas= np.logspace(-2,4,num=100)
gammas= [3]
mlMethod = 'svc'
rocs = []
fprs = []	# list of fpr list
tprs = []	# list of tpr list
for method in methods:
	if method == None:
		X_train,X_test,y_train,y_test=loadData()
		for gamma in gammas:
			clf = svm.SVC(kernel='linear',gamma=gamma,probability=True)
			clf.fit(X_train,y_train)
			y_score = clf.predict_proba(X_test) # 2 cols

			fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
			roc = roc_auc_score(y_test,y_score[:,1])
			fprs.append(fpr)
			tprs.append(tpr)
			rocs.append(roc)
		maxRoc = max(rocs)
		indMaxRoc = rocs.index(maxRoc)	# index respect to max roc in rocs list
		paramDict = {'gamma':gammas[indMaxRoc]}
		fpr = fprs[indMaxRoc]	# list 
		tpr = tprs[indMaxRoc]	# list

		resultItem = {'mlMethod':mlMethod,'compMethod':'origin','misRate':0,
						'maxRoc':maxRoc,'paramDict':paramDict,'fpr':fpr,'tpr':tpr}
		resultList.append(resultItem)
		rocs = []
	else:
		for misRate in misRates:
			X_train,X_test,y_train,y_test=loadData(misRate=misRate,method=method)
			print('compMethod:',method,'misRate:',misRate,'mlMethod: svc')
			for gamma in gammas:
				clf = svm.SVC(kernel='linear',gamma=gamma,probability=True)
				clf.fit(X_train,y_train)
				y_score = clf.predict_proba(X_test) # 2 cols

				fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
				roc = roc_auc_score(y_test,y_score[:,1])
				fprs.append(fpr)
				tprs.append(tpr)
				rocs.append(roc)
			maxRoc = max(rocs)
			indMaxRoc = rocs.index(maxRoc)	# index respect to max roc in rocs list
			paramDict = {'gamma':gammas[indMaxRoc]}
			fpr = fprs[indMaxRoc]	# list 
			tpr = tprs[indMaxRoc]	# list
			resultItem = {'mlMethod':mlMethod,'compMethod':method,'misRate':misRate,
							'maxRoc':maxRoc,'paramDict':paramDict,'fpr':fpr,'tpr':tpr}
			resultList.append(resultItem)
			rocs = []
## for teminal output
resultDf=pd.DataFrame(resultList)
resultDf[['compMethod','misRate','maxRoc']]

		
# print('rocs:')
# print(rocs)	# 0.82 max

# draw roc curve for svm
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
print('roc:')
print(roc)	# 0.796
ax.plot([0,1],[0,1],'k--')
ax.plot(fpr,tpr,'k--')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC of SVC')
ax.legend(loc='best')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
plt.show()
#####################################################
# rf roc
nTree = 100
clf=ensemble.RandomForestClassifier(n_estimators=nTree)
clf.fit(X_train,y_train)
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
ax.set_title('ROC of Random Forest')
ax.legend(loc='best')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
plt.show()


#####################################################
# draw roc curve for logistic regressions
Cs=np.logspace(-2,4,num=100)
rocs=[]
for C in Cs:
	clf = linear_model.LogisticRegression(C=C)
	clf.fit(X_train, y_train)
	y_score = clf.predict_proba(X_test)
	roc = roc_auc_score(y_test,y_score[:,1])
	rocs.append(roc)
print('rocs:')
print(rocs)	# 0.76
#####################################################
# draw roc curve for dt
maxdepth=20
depths=np.arange(1,maxdepth)
rocs=[]
for depth in depths:
	clf = DecisionTreeClassifier(max_depth=depth)
	clf.fit(X_train, y_train)
	y_score = clf.predict_proba(X_test)
	roc = roc_auc_score(y_test,y_score[:,1])
	rocs.append(roc)
print('rocs:')
print(rocs)	# 0.73

#####################################################
# draw roc curve for knn
Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
Ps=[1,2,10]
rocs=[]
for P in Ps:
	for K in Ks:
		clf=neighbors.KNeighborsClassifier(p=P,n_neighbors=K)
		clf.fit(X_train,y_train)
		y_score = clf.predict_proba(X_test)
		roc = roc_auc_score(y_test,y_score[:,1])
		rocs.append(roc)
print('rocs:')
print(rocs)	# 0.802

#####################################################
# draw roc curve for adaboost with dt
clf=ensemble.AdaBoostClassifier(learning_rate=0.05)
clf.fit(X_train,y_train)
y_score = clf.predict_proba(X_test)
roc = roc_auc_score(y_test,y_score[:,1])
print('roc:')
print(roc)	# 0.75
#####################################################
# draw roc curve for mlp
hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] # 候选的 hidden_layer_sizes 参数值组成的数组
rocs = []
for itx,size in enumerate(hidden_layer_sizes):
	clf=MLPClassifier(activation='logistic',max_iter=10000,hidden_layer_sizes=size)
	clf.fit(X_train,y_train)
	y_score = clf.predict_proba(X_test)
	roc = roc_auc_score(y_test,y_score[:,1])
	rocs.append(roc)
print('rocs:')
print(rocs)	# 0.75

#####################################################
# draw roc curve for xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
## tune max_depth and min_weight
param_test1 = {
	'max_depth': list(range(3, 10, 2)),
	'min_child_weight': list(range(1, 6, 2))
}
clf1 = GridSearchCV(
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
	iid = False,
	cv = 5)
clf1.fit(X_train, y_train)
y_score = clf.predict_proba(X_test) # 2 cols
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
print('gseaclf1rch1.best_params_')
print(clf1.best_params_)	# maxdepth:7 minweight:1


## tune gamma
param_test2 = {
'gamma': [i / 10.0 for i in range(0, 5)]
}
clf2 = GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=140,
	max_depth=7,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test2,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)
clf2.fit(X_train, y_train)
y_score = clf.predict_proba(X_test) # 2 cols
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
print('clf2.best_params_')	# 0.4
print(clf2.best_params_)

## tune subsample and colsample_bytree
param_test3 = {
'subsample': [i / 10.0 for i in range(6, 10)],
'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
clf3 = GridSearchCV(
estimator = XGBClassifier(
	learning_rate=0.1,
	n_estimators=177,
	max_depth=7,
	min_child_weight=1,
	gamma=0.4,
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
clf3.fit(X_train, y_train)
y_score = clf.predict_proba(X_test) # 2 cols
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
print('clf3.best_params_')	# 0.8 0.8
print(clf3.best_params_)

## tune learning rate and reg param
param_test4 = {
'learning_rate': [0.01,0.025,0.05,0.075,0.1],
}
clf4 = GridSearchCV(
estimator = XGBClassifier(
	n_estimators=177,
	max_depth=7,
	min_child_weight=1,
	gamma=0.4,
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
clf4.fit(X_train, y_train)
y_score = clf.predict_proba(X_test) # 2 cols
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
print('clf4.best_params_')	# 0.01
print(clf4.best_params_)

## tune reg param
param_test5 = {
'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
clf5 = GridSearchCV(
estimator = XGBClassifier(
	n_estimators=177,
	learning_rate=0.01,
	max_depth=7,
	min_child_weight=1,
	gamma=0.4,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27),
param_grid = param_test5,
scoring = 'roc_auc',
n_jobs = 4,
iid = False,
cv = 5)
clf5.fit(X_train, y_train)
y_score = clf.predict_proba(X_test) # 2 cols
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
print('clf5.best_params_')	# 0.01
print(clf5.best_params_)

