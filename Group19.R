
#********************************* PREPROCESSING****************************************

setwd("N:\\Acads\\4-2\\ML\\Assignment")

#Reading and viewing the CSV
ip <-read.csv(file="processed_cleveland_data.csv", head=TRUE, sep=",")
head(ip,5)
dim(ip)

str(ip)
#Datatype of columns is incorrect!
wrong <- c(3,7,11,12,13,14)
for (i in wrong) {
  ip[,i] = as.factor(ip[,i])
}
sapply(ip, class)
summary(ip)

#Analysis of Target variable (column_n)
table(ip$column_n)
barplot(table(ip$column_n),main="Column_n",col="orange")

#No. of missing values
length(which(is.na(ip)))

#Missing values in each column
apply(ip, 2, function(x){length(which(is.na(x)))})

#Checking if the rows with missing values belong to the minor class 4
ip[which(is.na(ip$column_l)),]$column_n
ip[which(is.na(ip$column_m)),]$column_n

#Percentage of records with missing values we are dropping
cat('Dropping ', round(length(which(is.na(ip)))*100/dim(ip)[1], digits=2), "% records")
df1<-ip[complete.cases(ip),]

#Exploratory Data Analysis (EDA)
df_heatmap <- heatmap(data.matrix(df1), Rowv=NA, Colv=NA, col = heat.colors(256), scale="column", margins=c(5,10))

numeric = c('a','d','e','h','j')
categ = c('b', 'c', 'f', 'g', 'i', 'k','l', 'm', 'n')

#Boxplots for numeric attributes
for (i in numeric) {
  x= paste0("column_", i)
  print(x)
  boxplot(df1[,x], main=x)
}

#Scatter plots for numeric attributes
for (i in numeric) {
  x= paste0("column_", i)
  print(x)
  plot(x = df1[,x], main = x, xlab = "Data Point", ylab = "Value", col="blue")
}
#Noticed some very extreme outliers. Good toremove these records as can adversely affect model performance.


#Barplots for categorical attributes
for (i in categ) {
  x= paste0("column_", i)
  print(x)
  barplot(table(df1[,x]), main=x, col="green")
}
#Notice that some attributes have a particular category very much under represented in the dataset.
#We will use Stratified sampling technique for proper training and testing.

#Outlier replacement with 99th and 1st percentiles (Treating the very extreme outliers)
quantile(df1$column_j, 0.99)
summary(df1$column_j)
df1$column_d[df1$column_d > quantile(df1$column_d, 0.99)] <- quantile(df1$column_d, 0.99)
df1$column_e[df1$column_e > quantile(df1$column_e, 0.99)] <- quantile(df1$column_e, 0.99)
df1$column_j[df1$column_j > quantile(df1$column_j, 0.99)] <- quantile(df1$column_j, 0.99)
summary(df1$column_j)

quantile(df1$column_h, 0.01)
df1$column_h[df1$column_h< quantile(df1$column_h, 0.01)]
subset(df1, df1$column_h== min(df1$column_h))
df1$column_h[df1$column_h==min(df1$column_h)] <- min(df1$column_h[df1$column_h!=min(df1$column_h)])
summary(df1$column_h)

#Export to CSV
write.csv(df1, file="processed.csv")



#****************************************** PART 1 **************************************************

library(caret)
library(tidyverse)
library(cluster)
library(tree)

# Stratified K-fold Cross Validation
folds <- createFolds(factor(df$column_n), k = 10, returnTrain = TRUE)
print(folds)
train_control <- trainControl(index= folds , method="cv", number=10, verboseIter=TRUE)
train_control2 <- trainControl(index= folds , method="cv", number=10, verboseIter=TRUE, classProbs=TRUE,
                               summaryFunction = twoClassSummary)
train_control3 <- trainControl(index= folds , method="cv", number=10, verboseIter=TRUE, classProbs=TRUE,
                               summaryFunction = mnLogLoss)

set.seed(14)
#df= read.csv("processed.csv")
df= df1
levels(df$column_n) <- make.names(levels(factor(df$column_n)))

#***************** 1. NAIVE BAYES *****************

#----- Discretization of numeric attributes by k-means clustering approach for each attribute

numeric2 = c('column_a','column_d','column_e','column_h','column_j')
k.values <- 1:10
#library(tidyverse)
for (i in numeric2) {
  wss <- function(k) {
    kmeans((df[,i]), k, nstart = 10 )$tot.withinss
  }
  wss_values <- map_dbl(k.values, wss)
  plot(k.values, wss_values,
       type="b", pch = 19, frame = FALSE,
       main = i,
       xlab="Number of clusters K",
       ylab="Total within-clusters sum of squares")
}
#library(cluster)
df_nb = df
for (i in numeric2) {
  k <- kmeans(df[,i], centers = 3, nstart = 10)
  df_nb[,i] = k$cluster
  df_nb[,i] = as.factor(df_nb[,i])
}
summary(df_nb)

#------ Model Fitting and Prediction
grid <- expand.grid(fL=c(0, 1, 2, 3), usekernel= c(TRUE), adjust=c(0.1,0.2,0.3))
model_nb <- train(column_n~., data=df_nb, trControl=train_control, method="nb", 
                  tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_nb)
plot(model_nb)

model_nb2 <- train(column_n~., data=df_nb, trControl=train_control3, method="nb", 
                  tuneGrid=grid, metric="logLoss", maximize=FALSE)
plot(model_nb2)

#***************** 2. LOGISTIC REGRESSION *****************

#Multinomial Logistic Regression is an extension of Logistic Regression for multi-class classification.
grid <- expand.grid(mtry=c(10, 20, 50))
model_lr <- train(column_n~., data=df, trControl=train_control, method="multinom", 
                  preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
print(model_lr)
plot(model_lr)

model_lr2 <- train(column_n~., data=df, trControl=train_control3, method="multinom", metric="logLoss", maximize=FALSE)
plot(model)

#***************** 3. SVM *****************

#----Note that the parameter C is for hard/soft margin

# SVM with all features -- (One vs One strategy used by kernlab)
grid <- expand.grid(C=c(0.001, 0.01, 0.1, 1,10,100,1000))
model_svml <- train(column_n~., data=df, trControl=train_control, method="svmLinear", 
                    preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_svml)
plot(model_svml)

model_svml2 <- train(column_n~., data=df, trControl=train_control3, method="svmLinear", 
                    preProcess = c("center", "scale"), tuneGrid=grid, metric="logLoss", maximize=FALSE)

#SVM with only numeric features
model_svml2 <- train(column_n~., data=df[,numeric], trControl=train_control, method="svmLinear", 
                     preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_svml2)

#Non linear SVM
grid <- expand.grid(C=c(0.001,0.01, 0.1, 1, 10), sigma=c(0.1, 1,10,100))
model_nlsvm <- train(column_n~., data=df, trControl=train_control, method="svmRadial", 
                     preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_nlsvm)
plot(model_nlsvm)

model_nlsvm2 <- train(column_n~., data=df, trControl=train_control3, method="svmRadial", 
                     preProcess = c("center", "scale"), tuneGrid=grid, metric="logLoss", maximize=FALSE)



#***************** 4. DECISION TREE (CART) *****************

# Pruning performed by the library itself!
model_dt <- train(column_n~., data=df, trControl=train_control, method="rpart",  
                  preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
print(model_dt)
plot(model_dt)

model_dt2 <- train(column_n~., data=df, trControl=train_control3, method="rpart",  
                   preProcess = c("center", "scale"), metric="logLoss", maximize=FALSE)


#***************** 5. DISCRIMINANT FUNCTIONS *****************

# Least Square
# Only works for numeric variables!
model_ls <- train(column_n~., data=df[,numeric], trControl=train_control, method="pls", 
                  preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
print(model_ls)

model_ls2 <- train(column_n~., data=df[,numeric], trControl=train_control3, method="pls", 
                  preProcess = c("center", "scale"), metric="logLoss", maximize=FALSE)


# FLD
# Only works for numeric variables!
model_fld <- train(column_n~., data=df[,numeric], trControl=train_control, method="lda", 
                   preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
print(model_fld)

model_fld2 <- train(column_n~., data=df[,numeric], trControl=train_control3, method="lda", 
                   preProcess = c("center", "scale"), metric="logLoss", maximize=FALSE)


#***************** 6. ENSEMBLES- RANDOM FOREST, ADABOOST *****************

# Random Forest
grid <- expand.grid(mtry=c(1:15))
model_rf <- train(column_n~., data=df, trControl=train_control, method="rf", ntree= 500, 
                  preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_rf)
plot(model_rf)
model_rf2 <- train(column_n~., data=df, trControl=train_control3, method="rf", ntree= 500, 
                   preProcess = c("center", "scale"), tuneGrid=grid, metric="logLoss", maximize=FALSE)

# Adaboost
grid <- expand.grid(mfinal=c(200, 250, 300), maxdepth=c(10, 20, 25), coeflearn=c("Breiman"))
model_adb <- train(column_n~., data=df, trControl=train_control, method="AdaBoost.M1", 
                   preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
print(model_adb)
plot(model_adb)

grid <- expand.grid(mfinal=c(250), maxdepth=c(10), coeflearn=c("Breiman"))
model_adb2 <- train(column_n~., data=df, trControl=train_control3, method="AdaBoost.M1", 
                    preProcess = c("center", "scale"), tuneGrid=grid, metric="logLoss", maximize=FALSE)


#***************** COMPARISON OF PERFORMANCE *****************

results <- resamples(list(LR=model_lr, FLD=model_fld, DT=model_dt, LS=model_ls, RF=model_rf, 
                          SVM=model_svml, SVM_nl= model_nlsvm, NB=model_nb))
summary(results)

# Box and whisker plots to compare model Accuracies
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# Density plots of accuracy
densityplot(results, scales=scales, pch = "|")

# Dot plots
dotplot(results, scales=scales)


# Log Loss comparison
results2 <- resamples(list(LR=model_lr2, FLD=model_fld2, DT=model_dt2, LS=model_ls2, RF=model_rf2, 
                          SVM=model_svml2, SVM_nl2= model_nlsvm2, Ada=model_adb2))
summary(results2)
bwplot(results2, scales=scales)

# ROC curve for SVM
#selectedIndices <- model_svml2$pred$Variables == model_svml2$optsize
#library(pROC)
#ROC = plot.roc(model_svml2$pred$obs[selectedIndices],
#               model_svml2$pred$neg[selectedIndices], legacy.axes = TRUE)

#**************************************** PART 2 (Ques 1,2) ************************************************

# -- Converting to binary problem
df_bin = df
df_bin$column_n = as.numeric(df_bin$column_n)
df_bin$column_n = df_bin$column_n-1
df_bin$column_n[df_bin$column_n > 0] <- 1
df_bin$column_n = as.factor(df_bin$column_n)
summary(df_bin)

levels(df_bin$column_n) <- make.names(levels(factor(df_bin$column_n)))
# -- Models

# Naive Bayes
df_bin_nb = df_nb
df_bin_nb$column_n = as.numeric(df_bin_nb$column_n)
df_bin_nb$column_n = df_bin_nb$column_n-1
df_bin_nb$column_n[df_bin_nb$column_n > 0] <- 1
df_bin_nb$column_n = as.factor(df_bin_nb$column_n)
grid <- expand.grid(fL=c(0, 1, 2, 3), usekernel= c(TRUE), adjust=c(0.1,0.2,0.3))
model_nb_b <- train(column_n~., data=df_bin_nb, trControl=train_control, method="nb", 
                    tuneGrid=grid, metric="Accuracy", maximize=TRUE)

levels(df_bin_nb$column_n) <- make.names(levels(factor(df_bin_nb$column_n)))
model_nb_b2 <- train(column_n~., data=df_bin_nb, trControl=train_control2, method="nb", 
                    tuneGrid=grid, metric="ROC", maximize=TRUE)

#Logistic Regression
model_lr_b <- train(column_n~., data=df_bin, trControl=train_control, method="glm", 
                    metric="Accuracy", maximize=TRUE)
model_lr_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="glm", 
                    metric="ROC", maximize=TRUE)


# SVM
grid <- expand.grid(C=c(0.001, 0.01, 0.1, 1, 10))
model_svml_b <- train(column_n~., data=df_bin, trControl=train_control, method="svmLinear", 
                      preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
model_svml_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="svmLinear", 
                      preProcess = c("center", "scale"), tuneGrid=grid, metric="ROC", maximize=TRUE)

# Non-linear SVM
grid <- expand.grid(C=c(0.01, 0.1, 1, 10), sigma=c(0.1, 1,10,100))
model_nlsvm_b <- train(column_n~., data=df_bin, trControl=train_control, method="svmRadial", 
                       preProcess = c("center", "scale"), tuneGrid=grid, metric="Accuracy", maximize=TRUE)
model_nlsvm_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="svmRadial", 
                       preProcess = c("center", "scale"), tuneGrid=grid, metric="ROC", maximize=TRUE)


# DT
model_dt_b <- train(column_n~., data=df_bin, trControl=train_control, method="rpart",  metric="Accuracy", maximize=TRUE)
model_dt_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="rpart",  metric="ROC", maximize=TRUE)

# FLD
model_fld_b <- train(column_n~., data=df_bin[,numeric], trControl=train_control, method="lda", 
                     preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
model_fld_b2 <- train(column_n~., data=df_bin[,numeric], trControl=train_control2, method="lda", 
                     preProcess = c("center", "scale"), metric="ROC", maximize=TRUE)


# Least Square
model_ls_b <- train(column_n~., data=df_bin[,numeric], trControl=train_control, method="pls", 
                    preProcess = c("center", "scale"), metric="Accuracy", maximize=TRUE)
model_ls_b2 <- train(column_n~., data=df_bin[,numeric], trControl=train_control2, method="pls", 
                    preProcess = c("center", "scale"), metric="ROC", maximize=TRUE)

# RF
grid <- expand.grid(mtry=c(1:15))
model_rf_b <- train(column_n~., data=df_bin, trControl=train_control, method="rf", ntree= 500, tuneGrid=grid, metric="Accuracy", maximize=TRUE)
model_rf_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="rf", ntree= 500, tuneGrid=grid, metric="ROC", maximize=TRUE)


# Adaboost
grid <- expand.grid(mfinal=c(250), maxdepth=c(10, 20, 25), coeflearn=c("Breiman"))
model_adb_b <- train(column_n~., data=df_bin, trControl=train_control, method="AdaBoost.M1", 
                     tuneGrid=grid, metric="Accuracy", maximize=TRUE)

grid <- expand.grid(mfinal=c(250), maxdepth=c(10), coeflearn=c("Breiman"))
model_adb_b2 <- train(column_n~., data=df_bin, trControl=train_control2, method="AdaBoost.M1", 
                     tuneGrid=grid, metric="ROC", maximize=TRUE)

# Comparison of Accuracy
results_b <- resamples(list(LR=model_lr_b, FLD=model_fld_b, DT=model_dt_b, LS=model_ls_b, RF=model_rf_b, 
                            SVM=model_svml_b, SVM_nl= model_nlsvm_b, NB=model_nb_b, Ada=model_adb_b))
summary(results_b)
scales_b <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results_b, scales=scales_b)
densityplot(results_b, scales=scales_b, pch = "")
dotplot(results_b, scales=scales_b)

# Comparison of ROC
results_b2 <- resamples(list(LR=model_lr_b2, FLD=model_fld_b2, DT=model_dt_b2, LS=model_ls_b2, RF=model_rf_b2, 
                            SVM=model_svml_b2, SVM_nl= model_nlsvm_b2, NB=model_nb_b2, Ada=model_adb_b2))
summary(results_b2)
scales_b2 <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results_b2, scales=scales_b2)
densityplot(results_b2, scales=scales_b2, pch = "")
dotplot(results_b2, scales=scales_b2)


# Perceptron

levels(df_bin$column_b) <- c(0,1)
levels(df_bin$column_f) <- c(0,1)
levels(df_bin$column_i) <- c(0,1)

cat = c('column_b', 'column_c', 'column_f', 'column_g', 'column_i', 'column_k','column_l', 'column_m', 'column_n')
df_bin_norm <-as.data.frame(apply(df_bin[,numeric2], 2, function(x) 2*((x - min(x))/(max(x)-min(x)))-1))
df_use <- cbind(df_bin_norm, df_bin[,cat])

for (i in 1:length(df_use)){
  df_use[,i] = as.numeric(df_use[,i])
}
df_use$column_n = df_use$column_n-1 

perceptron <- function(x, y, eta, niter) {
  # initialize weight vector
  weight <- rep(0, dim(x)[2] + 1)
  errors <- rep(0, niter)
  bias = -0.4
  learning_rate = 1
  # loop over number of epochs niter
  for (jj in 1:niter) {
    # loop through training data set
    for (ii in 1:length(y)) {
      z <- sum(weight[2:length(weight)] * 
                 as.numeric(x[ii, ])) + weight[1]
      if(z < -1*bias) {
        ypred <- 0
      } else {
        ypred <- 1
      }
      # Weight Update
      weightdiff <- eta * (y[ii] - ypred) * 
        c(1, as.numeric(x[ii, ]))
      weight <- weight + weightdiff
      # Update error function
      cat("\nTeacher-actual", (y[ii] - ypred))
      if ((y[ii] - ypred) != 0) {
        cat("Error:",errors[jj])
        errors[jj] <- errors[jj] + 1
      }
    }
    cat("\nGoing to next iteration")    
  }
  print(weight)
  return(c(errors, weight))
}

err <- perceptron(df_use[,-14], df_use[,14], 1, 22)
errors <- err[1:22]
wts <- err[23:36]
plot(1:22, errors, type="l", lwd=2, col="red", xlab="epoch #", ylab="errors")
title("Errors vs epoch - learning rate eta = 1")

preds <- rowSums(wts[2:14]*df_use[,-14])+ wts[1]
class(preds)
for (i in 1:length(preds)) {
  if (preds[i]<0){
    cat("\nClass 0 ")
  }
  else{
    cat("\nClass 1")
  }
}

s=0
for (i in 1:length(preds)) {
  cat("\n--", preds[i], df_use$column_n[i])
  if (preds[i]<0 & df_use$column_n[i]==0){
    s=s+1
    print(s)
  }
  else if (preds[i]>0 & df_use$column_n[i]==1){
    s=s+1
  }
}
cat("accuracy=", s*100/dim(df_use)[1])



#**************************************** PART 2 (Ques 3) ************************************************

# Implementing One vs All approach (only for classes 1,2,3,4)

df2 <- subset(df, df$column_n !=0)
a <- createDataPartition(df2$column_n, p = 0.75, list=FALSE)
train <- df2[a,]
test <- df2[-a,]

l <- dim(test)[1]

res <- data.frame(p1=1:l, p2=1:l, p3=1:l, p4=1:l)
for(i in 1:4){
  train$class <- ifelse(train$column_n==i,1,0)
  train$class <- as.factor(train$class)
  model <- tree(train$class~as.matrix(train[,numeric2]))
  preds <- predict(model, test[,-14], type="class")
  preds <- as.data.frame(preds)
  res[,i]<- preds
  #res<- cbind(res,as.data.frame(preds))
  print(preds)
}

# res is the dataframe having predictions of the 4 binary classifiers
print(res)

classes <- array(rep(0,l),dim = l)

# Looping through res to get class with the maximum votes
for (i in 1:l){
  s= array(rep(0,4))
  for (j in 1:4) {
    if(res[i,j]==0){
      for (k in 1:4) {
        if(k!=j){
          s[k]=s[k]+1
        }
      }
    }
    else{
      s[j]=s[j]+1
    }
  }
  cat("\n Row",  i, ",", s)
  classes[i] <- which.max(s)
}
print(classes)
sum(classes==test$column_n)/l


#**************************************** FIN ************************************************