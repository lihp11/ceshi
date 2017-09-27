import scipy.io as sio
import numpy as np
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

## lr_c
def lr_c(*data):
	'''
	测试 LogisticRegression 的预测性能随  C  参数的影响

	:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
	Cs=np.logspace(-2,4,num=100)
	scores=[]
	for C in Cs:
		regr = linear_model.LogisticRegression(C=C)
		regr.fit(X_train, y_train)
		scores.append(regr.score(X_test, y_test))
	## 绘图
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.plot(Cs,scores)
	ax.set_xlabel(r"C")
	ax.set_ylabel(r"score")
	ax.set_xscale('log')
	ax.set_title("LogisticRegression")
	ax.set_ylim([0.6,0.8])
	plt.show()
## dt
def dt(*data,maxdepth):
	'''
	测试 DecisionTreeClassifier 的预测性能随 max_depth 参数的影响

	:param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:param maxdepth: 一个整数，用于 DecisionTreeClassifier 的 max_depth 参数
	:return:  None
	'''
	X_train,X_test,y_train,y_test=data
	depths=np.arange(1,maxdepth)
	training_scores=[]
	testing_scores=[]
	for depth in depths:
		clf = DecisionTreeClassifier(max_depth=depth)
		clf.fit(X_train, y_train)
		training_scores.append(clf.score(X_train,y_train))
		testing_scores.append(clf.score(X_test,y_test))

	## 绘图
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.plot(depths,training_scores,label="traing score",marker='o')
	ax.plot(depths,testing_scores,label="testing score",marker='*')
	ax.set_xlabel("maxdepth")
	ax.set_ylabel("score")
	ax.set_title("Decision Tree Classification")
	ax.legend(framealpha=0.5,loc='best')
	plt.show()
## GaussianNB
def GaussianNB(*data):
	'''
	测试 GaussianNB 的用法

	:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
	cls=naive_bayes.GaussianNB()
	cls.fit(X_train,y_train)
	print('cls.theta_')
	print(cls.theta_)
	print('cls.sigma_')
	print(cls.sigma_)
	print('Training Score: %.2f' % cls.score(X_train,y_train))
	print('Testing Score: %.2f' % cls.score(X_test, y_test))
## knn
def knn_kp(*data):
	'''
	测试 KNeighborsClassifier 中 n_neighbors 和 p 参数的影响

	:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
	Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
	Ps=[1,2,10]

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	### 绘制不同 p 下， 预测得分随 n_neighbors 的曲线
	for P in Ps:
		training_scores=[]
		testing_scores=[]
		for K in Ks:
			clf=neighbors.KNeighborsClassifier(p=P,n_neighbors=K)
			clf.fit(X_train,y_train)
			testing_scores.append(clf.score(X_test,y_test))
			training_scores.append(clf.score(X_train,y_train))
		ax.plot(Ks,testing_scores,label="testing score:p=%d"%P)
		ax.plot(Ks,training_scores,label="training score:p=%d"%P)
	ax.legend(loc='best')
	ax.set_xlabel("K")
	ax.set_ylabel("score")
	ax.set_ylim(0,1.05)
	ax.set_title("KNeighborsClassifier")
	plt.show()
## svm
def svc_rbf(*data):
	'''
	测试 高斯核的 SVC 的预测性能随 gamma 参数的影响

	:param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
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
## mlp
def mlp_size(train_x,test_x,train_y,test_y):
	fig=plt.figure()
	hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] # 候选的 hidden_layer_sizes 参数值组成的数组
	train_scores = []
	test_scores = []
	for itx,size in enumerate(hidden_layer_sizes):
		ax=fig.add_subplot(2,3,itx+1)
		classifier=MLPClassifier(activation='logistic',max_iter=10000,hidden_layer_sizes=size)
		classifier.fit(train_x,train_y)
		train_score=classifier.score(train_x,train_y)
		train_scores.append(train_score)
		test_score=classifier.score(test_x,test_y)
		test_scores.append(test_score)
	print('train_scores')
	print(train_scores)
	print('test_scores')
	print(test_scores)
	plt.show()

## rf
def rf_ntree(*data):
	'''
	测试 RandomForestClassifier 的预测性能随 n_estimators 参数的影响

	:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
	nums=np.arange(1,100,step=2)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	testing_scores=[]
	training_scores=[]
	for num in nums:
		clf=ensemble.RandomForestClassifier(n_estimators=num)
		clf.fit(X_train,y_train)
		training_scores.append(clf.score(X_train,y_train))
		testing_scores.append(clf.score(X_test,y_test))
	ax.plot(nums,training_scores,label="Training Score")
	ax.plot(nums,testing_scores,label="Testing Score")
	ax.set_xlabel("estimator num")
	ax.set_ylabel("score")
	ax.legend(loc="lower right")
	ax.set_ylim(0,1.05)
	plt.suptitle("RandomForestClassifier")
	plt.show()


## adboost
def adboost_base(*data):
	'''
	测试  AdaBoostClassifier 的预测性能随基础分类器数量和基础分类器的类型的影响

	:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return:  None
	'''
	from sklearn.naive_bayes import GaussianNB
	X_train,X_test,y_train,y_test=data
	fig=plt.figure()
	ax=fig.add_subplot(1,2,1)
	########### 默认的个体分类器 #############
	clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
	clf.fit(X_train,y_train)
	## 绘图
	estimators_num=len(clf.estimators_)
	X=range(1,estimators_num+1)
	ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
	ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
	ax.set_xlabel("estimator num")
	ax.set_ylabel("score")
	ax.legend(loc="lower right")
	ax.set_ylim(0,1)
	ax.set_title("AdaBoostClassifier with Decision Tree")
	####### Gaussian Naive Bayes 个体分类器 ########
	ax=fig.add_subplot(1,2,2)
	clf=ensemble.AdaBoostClassifier(learning_rate=0.1,base_estimator=GaussianNB())
	clf.fit(X_train,y_train)
	## 绘图
	estimators_num=len(clf.estimators_)
	X=range(1,estimators_num+1)
	ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
	ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
	ax.set_xlabel("estimator num")
	ax.set_ylabel("score")
	ax.legend(loc="lower right")
	ax.set_ylim(0,1)
	ax.set_title("AdaBoostClassifier with Gaussian Naive Bayes")
	plt.show()

## main
if __name__=='__main__':
	X_train,X_test,y_train,y_test = load_data()
	# lr_c(X_train,X_test,y_train,y_test)	# test C
	# dt(X_train,X_test,y_train,y_test,maxdepth=10)	# dt maxdepth
	# GaussianNB(X_train,X_test,y_train,y_test)	# GaussianNB
	# knn_kp(X_train,X_test,y_train,y_test)
	# svc_rbf(X_train,X_test,y_train,y_test)
	# mlp_size(X_train,X_test,y_train,y_test)
	# rf_ntree(X_train,X_test,y_train,y_test)
	# adboost_base(X_train,X_test,y_train,y_test)

