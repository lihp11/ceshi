# read data from mat file
setwd('e:\\research\\acc_detect\\paper')
library(pcaMethods)
library('R.matlab')
data <- readMat('finaldata.mat')
X <- data$X # matrix
Y <- data$Y # matrix
nData <- nrow(X)
maxXCol = apply(X,2,max)	# vec length=24
rowDiv <- function(row,denom){
  result <- row/denom
	return (result)}
X = t(apply(X,1,rowDiv,denom=maxXCol))
	# x = X
	# dim(x) = c(length(X),1)	# like matlab
# create rmse dataframe and mape dataframe
rmseDf <- data.frame(rate=numeric(),avg=numeric(),bpca=numeric(),ppca=numeric())
mapeDf <- data.frame(rate=numeric(),avg=numeric(),bpca=numeric(),ppca=numeric())
fullXDf <- data.frame(X1=numeric(),X2=numeric(),X3=numeric(),X4=numeric(),
							X5=numeric(), X6=numeric(), X7=numeric(), X8=numeric(),
							X9=numeric(), X10=numeric(),X11=numeric(),X12=numeric(),
							X13=numeric(),X14=numeric(),X15=numeric(),X16=numeric(),
							X7=numeric(), X18=numeric(),X19=numeric(),X20=numeric(),
							X21=numeric(),X22=numeric(),X23=numeric(),X24=numeric(),
							misRate=numeric(),method=character())
# make mat with missing
misRates <- seq(0.05,0.6,0.05)
for (misRate in misRates){
	misNum <- round(misRate*length(X))	# length(X):total number in matrix
	set.seed(1235)
	misInd <- sample(1:length(X),misNum)
	misX <- X
	dim(misX) <- c(length(X),1)	# to vec
	misX[misInd] <- NA
	dim(misX) <- dim(X)
	avgX <- misX
	for(i in 1:ncol(avgX)){
	  avgX[is.na(avgX[,i]) & 1:(nData/2) , i] <- mean(avgX[1:(nData/2),i] , na.rm = TRUE)
	  avgX[is.na(avgX[,i]) & (nData/2+1):nData , i] <- mean(avgX[(nData/2+1):nData,i] , na.rm = TRUE)
	}
	# methodVec = rep('mean',dim(avgX)[1])
	# rateVec = rep(misRate,dim(avgX)[1])
	method = 'mean'
	avgXDf = cbind(data.frame(avgX),method,misRate)
	fullXDf = rbind(fullXDf,avgXDf)
  cat('dim(fullXDf) is ',dim(fullXDf))
	  # call pca package to complete misX
		# listPcaMethods()
		#  [1] "svd"          "nipals"       "rnipals"     
		#  [4] "bpca"         "ppca"         "svdImpute"   
		#  [7] "robustPca"    "nlpca"        "llsImpute"   
		# [10] "llsImputeAll"
	cat('Missing Rate is',misRate,'.\n')
	cat('Missing Num is',misNum,'.\n')
	bpcaX <- pca(misX,nPcs=23,method='bpca',maxSteps=500)
	bpcaX <- completeObs(bpcaX)
	method = 'bpca'
	bpcaDf = cbind(data.frame(bpcaX),method,misRate)
	fullXDf = rbind(fullXDf,bpcaDf)
	cat('dim(fullXDf) is ',dim(fullXDf))
	

	ppcaX <- pca(misX,nPcs=10,method='ppca', maxIterations = 1000)
	ppcaX <- completeObs(ppcaX)
	method = 'ppca'
	ppcaDf = cbind(data.frame(bpcaX),method,misRate)
	fullXDf = rbind(fullXDf,ppcaDf)
	cat('dim(fullXDf) is ',dim(fullXDf))
	# bpcaXDf <- cbind(bpcaX,matrix(1,nrow=dim(bpcaX)[1],ncol=1))
	# bpcaXDf <- data.frame(bpcaXDf)

	# calc MAPE and RMSE
	mapeAvgX <- sum(abs(avgX[is.na(misX)]-X[is.na(misX)])/X[is.na(misX)])/misNum
	mapeBpcaX <- sum(abs(bpcaX[is.na(misX)]-X[is.na(misX)])/X[is.na(misX)])/misNum
	mapePpcaX <- sum(abs(ppcaX[is.na(misX)]-X[is.na(misX)])/X[is.na(misX)])/misNum
	rmseAvgX <- sqrt(sum((avgX[is.na(misX)]-X[is.na(misX)])^2)/misNum)
	rmseBpcaX <- sqrt(sum((bpcaX[is.na(misX)]-X[is.na(misX)])^2)/misNum)
	rmsePpcaX <- sqrt(sum((ppcaX[is.na(misX)]-X[is.na(misX)])^2)/misNum)

	rmseDf[nrow(rmseDf) + 1,] = c(misRate,rmseAvgX,rmseBpcaX,rmsePpcaX)
	mapeDf[nrow(mapeDf) + 1,] = c(misRate,mapeAvgX,mapeBpcaX,mapePpcaX)
}

rmseDf
mapeDf

###################################
# export fullXDf,rmseDf,and mapeDf to sqlite
library(DBI)
library(RSQLite)
con <- dbConnect(RSQLite::SQLite(), 'accDetect.db')
if (dbExistsTable(con,'fullX')){
  dbExecute(con, 'DROP TABLE fullX')
}
dbWriteTable(con, 'fullX', fullXDf )

if (dbExistsTable(con,'rmse')){
  dbExecute(con, 'DROP TABLE rmse')
}
dbWriteTable(con, 'rmse', rmseDf )

if (dbExistsTable(con,'mape')){
  dbExecute(con, 'DROP TABLE mape')
}
dbWriteTable(con, 'mape', mapeDf )
if (dbExistsTable(con,'realX')){
  dbExecute(con, 'DROP TABLE realX')
}
dbWriteTable(con, 'realX', data.frame(X) )
dbDisconnect(con)







