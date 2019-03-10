##Package loading
library(dplyr)
library(mlbench)
library(caret)
library(ROCR)
library(e1071)
library(dataPreparation)
library(ROCR)
library(ggplot2)
library(plotROC)
library(pROC)
data<-read.csv(url("http://data.mishra.us/files/OJ.csv"))
summary(data)
str(data)
##No Contants
constant_cols <- whichAreConstant(data)
constant_cols
##No Double Columns
double_cols <- whichAreInDouble(data)
double_cols
##1 Bijection Column - ID 18 [Delete the column]
bijections_cols <- whichAreBijection(data)
bijections_cols
#Deleting column STORE
data <- subset( data, select = -STORE )
str(data)
#Correlation between dependent variables
correlationMatrix <- cor(data[,c(2,3,4,5,6,7,8,9,10,11,12,13,15,16,17)])
#Summarize the correlation matrix
print(correlationMatrix)
#Attributes that are highly corrected (>0.7)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7,names = T)
#printing highly correlated attributes
print(highlyCorrelated)
#Removing higly correlated variabes
data <- subset( data, select = -c(2,6,11,12,13,15))
str(data)
#Checking for MM [1=M,0=CH]
data$Purchase <- ifelse(data$Purchase == "MM", 1, 0)
data$Purchase <- factor(data$Purchase, levels = c(0, 1))
str(data)
#Creating Train & Test Data Sets
set.seed(100)
trainDataIndex <- createDataPartition(data$Purchase, p=0.7, list = F) # 70% training data
trainData <- data[trainDataIndex, ]
testData <- data[-trainDataIndex, ]
str(trainData)
scales <- build_scales(dataSet = trainData, cols = c("PriceCH", "PriceMM", "DiscMM", "LoyalCH", "PctDiscCH", "ListPriceDiff" ), verbose = TRUE)
trainData <- fastScale(dataSet = trainData, scales = scales, verbose = TRUE)
testData <- fastScale(dataSet = testData, scales = scales, verbose = TRUE)
str(trainData)
#Cross Vaidation
fitControl <- trainControl(## 10-fold CV
method = "repeatedcv",
number = 2,
## repeated ten times
repeats = 3,
summaryFunction=twoClassSummary,
classProbs = TRUE)
logitmod <- glm(Purchase ~ ., family = "binomial", data=trainData)
summary(logitmod)
#Removing Variabes with high P-Values (>0.05)
trainData <- subset( trainData, select = -c(2,6,7,9,11))
testData <- subset( testData, select = -c(2,6,7,9,11))
str(trainData)
levels(trainData$Purchase) <- make.names(levels(factor(trainData$Purchase)))
levels(testData$Purchase) <- make.names(levels(factor(testData$Purchase)))
str(trainData)
str(testData)
logitmod_1 <- glm(Purchase ~ ., family = "binomial", data=trainData)
summary(logitmod_1)
##### Logistic
logFit <- train(Purchase ~ ., data=trainData, method="glm", family="binomial",trControl = fitControl,metric = "ROC")
logFit
logPred <- predict(logFit, newdata = testData)
#Confusion Matrix on Test Data
confusionMatrix(data = logPred, testData$Purchase)
#Plotting ROC curve for logit model
roc(testData$Purchase, predict(logFit, newdata=testData, type="prob" )[,2],plot=T,
auc=T)
grid_radial <- expand.grid(sigma = c(.01,.02),
C = c(.75,1,1.5))
###### SVM - Radial
svmFit1 <- train(Purchase ~ ., data = trainData,
method='svmRadial',
trControl = fitControl,
#preProc = c("center","scale"),
metric = "ROC",
verbose = FALSE,
probability = TRUE,
tuneGrid = grid_radial
)
svmFit1
svmPred1 <- predict(svmFit1, newdata = testData)
#Confusion Matrix on Test Data
confusionMatrix(data = svmPred1, testData$Purchase)
#Plotting ROC curve for SVM - Radial model
roc(testData$Purchase, predict(svmFit1, newdata=testData, type="prob" )[,2],plot = T, auc = T)
#Finding best value of Cost for linear kernel
grid_linear <- expand.grid(C = c(.75,1,1.5))
svmFit_linear <- train(Purchase ~ ., data = trainData,
method='svmLinear',
trControl = fitControl,
#preProc = c("center","scale"),
metric = "ROC",
verbose = FALSE,
probability = TRUE,
tuneGrid = grid_linear
)
svmFit_linear
svmPred_linear <- predict(svmFit_linear, newdata = testData)
#Confusion Matrix on Test Data
confusionMatrix(data = svmPred_linear, testData$Purchase)
#Plotting ROC curve for SVM model
roc(testData$Purchase, predict(svmFit_linear, newdata=testData, type="prob" )[,2],plot = T, auc = T)
#####SVM - Polynomial
svm_p <- tune.svm(Purchase ~ ., data = trainData,
degree = (3:6),
gamma = (0.1:0.4),
coef0 = 1,
kernel = "polynomial")
svm_p
svm_poly <- svm(Purchase ~ ., data=trainData, type='C-classification', kernel='polynomial', degree=3, gamma=0.1, coef0=1, scale=FALSE, probability = TRUE)
summary(svm_poly)
svmPred_poly <- predict(svm_poly, newdata = testData)
#Confusion Matrix on Test Data
confusionMatrix(data = svmPred_poly, testData$Purchase
