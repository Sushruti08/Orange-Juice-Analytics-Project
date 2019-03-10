---
title: "Orange Juice Analytics Project"
output: html_notebook
---

```{r}
library(caret)
library(e1071)
library(kernlab)
library(dplyr)
library(arm)
library(pROC)
library(grid)
library(gridExtra)
```

```{r}
#Download data
OJ<-read.csv(url("http://data.mishra.us/files/OJ.csv"))
```

```{r}
#Identify Correllated items
correlationMatrix <- cor(OJ[,c(2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18)])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)
#12,10,5,6

```

```{r}
#Transform into Binary and Factors
OJ$Purchase <-as.factor(OJ$Purchase)
f=c(1,2,3,8,9,14,18)
for(i in f) OJ[,i]=as.factor(OJ[,i])
set.seed(123)
rows<-sample(nrow(OJ), .8*nrow(OJ))
OJ_train<- OJ[rows,]
OJ_test<-OJ[-rows,]
#Override train data with
#6 and 12 dropped from correlation matrix output
#2,3,7,8,9,11,13,14,17,18- dropped due to lack of impact/collinearity discovered through running models

```

```{r}
#included variables
OJ_train<- OJ_train[,c(1,4,5,10,15,16)]
OJ_test<- OJ_test[,c(1,4,5,10,15,16)]
```

```{r}
#Make Logged Model
summary(logged_model<-train(Purchase~.,
      data=OJ_train,
      method="glm",
      #preProcess=c("center","scale"),
      family="binomial"))
coefficients<-coef(logged_model$finalModel)
#OJ_test$YN<-as.factor(if_else(OJ_test$Purchase==OJ_test$logpredict2,"Y","N"))
#OJ_test$predicted<-OJ_test$logpredict[1]
#OJ_test$predicted2<-OJ_test$logpredict[2]
#ggplot(OJ_test, aes(predicted, Purchase, col=YN))+
#  geom_point()+
#  stat_smooth(method="glm",              
#              formula=OJ_test$Purchase~ OJ_test$predicted,
#              family="binomial",
#                col="red")
```

```{r}
#Use this to play with coefficients
invlogit(coefficients[1]+coefficients[2]*.3+coefficients[3]*-.3+coefficients[4]*1+
           coefficients[5]*0+coefficients[6]*0)
```

```{r}
###############################################################################################
#Positive Coefficients are coefficients that increase probability of MM being purchased
  #PriceCH-As competitor price increases MM is more likely to be purchased
  #PctDiscMM- As discount goes up MM is more likely to be purchased
table(OJ$Purchase)

#Negative Coefficients are coefficients that decrease probability of MM being purchased
  #PriceMM-As MM price increases MM is less likely to be purchased
  #LoyalCH- As discount goes up MM is less likely to be purchased
  #PctDiscCH-As competitor discount increases MM is less likely to be purchased

###############################################################################################
```
```{r}
#In terms of probability
options(scipen=999)
invlogit(coef(logged_model$finalModel))
exp(coef(logged_model$finalModel))

```
```{r}
#Probability Confusion Matrix
#confusionMatrix(logged_model)

```

```{r}
#Run model against test data
logpredict <- predict(logged_model, newdata = OJ_test, probability=T, type="raw")
confusionMatrix(data = logpredict, OJ_test$Purchase)
roc(OJ_test$Purchase, predict(logged_model, newdata=OJ_test, type="prob" )[,2],plot=T, auc=T)
#.8364 accuracy-Logistic Model
#.9042 AUC- Logistic Model

```

```{r}
#SVM approach

#Setup
fitControl<- trainControl(
    method="repeatedcv",
    number=5,
    repeats=5,
    summaryFunction=twoClassSummary,
    classProbs = TRUE
)

```

```{r}
#Grid for tuning parameters
grid <- expand.grid(#sigma=c(.75,1),
  #Weight=c(1,2),
  #tau=c(1,2)
  #lambda=c(1,2),
  #qval=c(1),
  #degree=c(1,6,4),
  #scale=c(1.1,1.2)
  C=c(.1,.09)
  #sparsity=c(.1,.5,1,2)
  #cost=seq(.01,.15,.01)
  #nleaves=c(1,5,10),
  #ntrees=c(1,5,10),
  #K.prov=c(1,2,3,4)
)

```

```{r}
#Assigning the models
svmfit <- train(Purchase~.,
                data=OJ_train,
                method="svmLinear",
                trControl=fitControl,
                preProcess=c("center","scale"),
                metric="ROC",
                verbose=F,
                probability=T,
                tuneGrid=grid
                )

```

```{r}
#Check out the model and what tuning parameters were chosen
svmfit

```

```{r}
#Make with predicted OJ output and then Confusion matrix on performance
svmPred <- predict(svmfit, newdata=OJ_test, probability=T)
svmPred2 <- as.data.frame(round(predict(svmfit, newdata=OJ_test, probability=T, type="prob"),4))
colnames(svmPred2)<-c("CH","MM")
grid.table(head(svmPred2,10))
confusionMatrix(svmPred, OJ_test$Purchase)
roc(OJ_test$Purchase, predict(svmfit, newdata=OJ_test, type="prob" )[,2])
#Best Model acheived:
  #svmLinear 
  #.8458-.1
#OR
  #svmPoly--.8458 degree=  1, scale=1, cost=.1

```

```{r}
#Write up coefficient interpretation
#Finish slide/graph
#ROC graphs 
par(mfrow=c(1,2))
roc(OJ_test$Purchase, predict(svmfit, newdata=OJ_test, type="prob" )[,2],plot=T,
    plot.roc(col="red"),auc=T)
roc(OJ_test$Purchase, predict(logged_model, newdata=OJ_test, type="prob" )[,2],plot=T,auc=T)
par(mfrow=c(1,1))
#plot(h)

```

```{r}
rocsvm<- roc(OJ_test$Purchase, predict(svmfit, newdata=OJ_test, type="prob" )[,2])
roclog<- roc(OJ_test$Purchase, predict(logged_model, newdata=OJ_test, type="prob" )[,2])
plot(rocsvm)
plot.roc(rocsvm)
plot.roc(smooth(roclog), print.auc=T, col="red", main="Logistic")
plot.roc(smooth(rocsvm), print.auc=T, col="blue",main="SVM")

```

```{r}
plot.roc(roclog, print.auc=T, col="red", main="Logistic")
plot.roc(rocsvm, print.auc=T, col="blue",main="SVM")

```

