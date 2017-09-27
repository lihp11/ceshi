Perform principal component analysis

Description

Perform PCA on a numeric matrix for visualisation, information extraction and missing value imputation.

Usage

pca(object, method, nPcs = 2, scale = c("none", "pareto", "vector", "uv"),
  center = TRUE, completeObs = TRUE, subset = NULL, cv = c("none",
  "q2"), ...)
Arguments

object	
Numerical matrix with (or an object coercible to such) with samples in rows and variables as columns. Also takes ExpressionSet in which case the transposed expression matrix is used. Can also be a data frame in which case all numberic variables are used to fit the PCA.
method	
One of the methods reported by listPcaMethods(). Can be left missing in which case the svd PCA is chosen for data wihout missing values and nipalsPca for data with missing values
nPcs	
Number of principal components to calculate.
scale	
Scaling, see prep.
center	
Centering, see prep.
completeObs	
Sets the completeObs slot on the resulting pcaRes object containing the original data with but with all NAs replaced with the estimates.
subset	
A subset of variables to use for calculating the model. Can be column names or indices.
cv	
character naming a the type of cross-validation to be performed.
...	
Arguments to prep, the chosen pca method and Q2.
Details

This method is wrapper function for the following set of pca methods:

svd:
Uses classical prcomp. See documentation for svdPca.

nipals:
An iterative method capable of handling small amounts of missing values. See documentation for nipalsPca.

rnipals:
Same as nipals but implemented in R.

bpca:
An iterative method using a Bayesian model to handle missing values. See documentation for bpca.

ppca:
An iterative method using a probabilistic model to handle missing values. See documentation for ppca.

svdImpute:
Uses expectation maximation to perform SVD PCA on incomplete data. See documentation for svdImpute.

Scaling and centering is part of the PCA model and handled by prep.

Value

A pcaRes object.

Author(s)

Wolfram Stacklies, Henning Redestig

References

Wold, H. (1966) Estimation of principal components and related models by iterative least squares. In Multivariate Analysis (Ed., P.R. Krishnaiah), Academic Press, NY, 391-420.

Shigeyuki Oba, Masa-aki Sato, Ichiro Takemasa, Morito Monden, Ken-ichi Matsubara and Shin Ishii. A Bayesian missing value estimation method for gene expression profile data. Bioinformatics, 19(16):2088-2096, Nov 2003.

Troyanskaya O. and Cantor M. and Sherlock G. and Brown P. and Hastie T. and Tibshirani R. and Botstein D. and Altman RB. - Missing value estimation methods for DNA microarrays. Bioinformatics. 2001 Jun;17(6):520-5.

See Also

prcomp, princomp, nipalsPca, svdPca

Examples

data(iris)
##  Usually some kind of scaling is appropriate
pcIr <- pca(iris, method="svd", nPcs=2)
pcIr <- pca(iris, method="nipals", nPcs=3, cv="q2")
## Get a short summary on the calculated model
summary(pcIr)
plot(pcIr)
## Scores and loadings plot
slplot(pcIr, sl=as.character(iris[,5]))

## use an expressionset and ggplot
data(sample.ExpressionSet)
pc <- pca(sample.ExpressionSet)
df <- merge(scores(pc), pData(sample.ExpressionSet), by=0)
library(ggplot2)
ggplot(df, aes(PC1, PC2, shape=sex, color=type)) +
  geom_point() +
  xlab(paste("PC1", pc@R2[1] * 100, "% of variance")) +
  ylab(paste("PC2", pc@R2[2] * 100, "% of variance"))