#Course Project

#Background
#Data Information
#Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

library(caret)
library(forecast)
library(AER) 
library(ggplot2)
library(AppliedPredictiveModeling)
library(kernlab)
library(randomForest)
library(parallel)
training <- read.csv("pml-training.csv",header = TRUE, sep = ",")
testing <- read.csv("pml-testing.csv",header = TRUE, sep = ",")
str(training)
colnames(testing)
testing
#summary(training)
#dim(training)

##Data Cleaning
# we have 19622 obersvations and   160 variables in oour training dataset.
# We need to understand the amount of missing values in data. Any variable missing more than 50% should not be used because they can give a false impression of the relationship with the dependent and hence can pollute the model.
# In our study we will eliminate all variables with any missing values to be on the conservative side.
# Count of missing values for each variable
Missing_Values <- sapply(training,function(x) sum(is.na(x)))
MissingValues <- as.data.frame(Missing_Values)
name <- row.names(MissingValues)
count_missing <- cbind(name,MissingValues)
row.names(count_missing) <- NULL
colnames(count_missing) <- c("variable", "Count Missing")
#We will now filter out the data with any missing values.
Newdata <-  as.character(count_missing$variable[which(count_missing$`Count Missing`<=1)])
training <- training[Newdata]
#we have now filtered all the missing variables and now have 19622 obervations and 93 variables.
#dim(training) 
#str(training)

# Removing Missing Values from testing data
Missing_Values_test <- sapply(testing,function(x) sum(is.na(x)))
MissingValues_test <- as.data.frame(Missing_Values_test)
name <- row.names(MissingValues_test)
count_missing_test <- cbind(name,MissingValues_test)
row.names(count_missing_test) <- NULL
colnames(count_missing_test) <- c("variable", "Count Missing")
#We will now filter out the data with any missing values.
testing_new <-  as.character(count_missing_test$variable[which(count_missing_test$`Count Missing`<=1)])
testing <- testing[testing_new]
##str(testing)

# Removing unwanted data
#We will remove first seven columns from the training and testing data sets because they do not contain any relevant information that will help our prediction outcome.
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

#Removing variables with no variability or zero covariates in training data set.
training <-training[,-nearZeroVar(training)]
str(training)
# we have now filtered our training dataset and have 19622 observartions and  53 variables.

#Splitting the training data set into training and validation by retaining 70% data in training set and 30% in validation set.
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
train_new <- training[inTrain,]
dim(train_new) # 13737    53
val <- training[-inTrain,]
dim(val) # 5885   53


#Building the Model
# Model1- Using Random Forest Algorithm with 5 fold cross validation to select optimum parameters

traincontrol <- trainControl(method="cv", number=3)
model <- train(classe~., data=train_new, method="rf", trControl = traincontrol)
pred <- predict(model,train_new)
rfc <- confusionMatrix(pred,train_new$classe)
print(rfc$overall[1]) # printing overall accuracy  1
plot(model)

mod_boost <- train(classe~.,data=train_new,method="gbm",trControl=traincontrol)
predboost <- predict(mod_boost,train_new)
c <- confusionMatrix(predboost,train_new$classe)
print(c$overall[1]) # printing overall accuracy 0.9737206
plot(mod_boost)

mod_lda <- train(classe~.,data=train_new,method="lda",trControl=traincontrol)
pre_lda <- predict(mod_lda,train_new)
clda <- confusionMatrix(pre_lda,train_new$classe)
print(clda$overall[1]) # printing overall accuracy 0.7081604 


mod_rpart <- train(classe~.,data=train_new,method="rpart",trControl=traincontrol)
pre_rpart <- predict(mod_rpart,train_new)
crpart <- confusionMatrix(pre_rpart,train_new$classe)
print(crpart$overall[1]) # printing overall accuracy 0.4951591 
suppressMessages(library(rattle))
library(rpart.plot)
fancyRpartPlot(mod_rpart$finalModel)

# We now have 4 trained models and we will be evaluating the accuracy of each of them.

##Cross Validation
#Predicting on the validation/testing set
pred_val_rf <- predict(model,val)
rfc_val <- confusionMatrix(pred_val_rf,val$classe)
print(rfc_val$overall[1]) # printing overall accuracy 0.9930331

mod_boost <- train(classe~.,data=train_new,method="gbm",trControl=traincontrol)
predboost_val <- predict(mod_boost,val)
c_val <- confusionMatrix(predboost_val,val$classe)
print(c_val$overall[1]) #Accuracy 0.962107

pred_lda_val <- predict(mod_lda,val)
clda_val <- confusionMatrix(pred_lda_val,val$classe)
print(clda_val$overall[1]) # printing overall accuracy 0.700085 

pred_rpart_val <- predict(mod_rpart,val)
crpart_val <- confusionMatrix(pred_rpart_val,val$classe)
print(crpart_val$overall[1]) # printing overall accuracy 0.4973662

ModelAccuracyTable <- data.frame(Model=c("RandomForest", "GBM", "LDA", "Classification Tree"),
                                 Accuracy=rbind(rfc_val$overall[1],c_val$overall[1],clda_val$overall[1],crpart_val$overall[1]))
print(ModelAccuracyTable)

#Predicting on test data from file: pml-testing.csv
predTesting <- predict(model,newdata=testing)
TestingPredictionResults <- data.frame(problem_id=testing$problem_id,predicted=predTesting)
print(TestingPredictionResults)

#Based on the Overall Accuracy numbers, we can see that the Random Forest model gives us the most accurate results as compared to Classification/Decison Tree model and Linear Discriminant Analysis.
#Hence we can accept the Random Forest Model for our study.
print(rfc_val$table)
paste0("The In Sample Error Rate for the selected model is ",round(1-rfc_val[["overall"]][["Accuracy"]],3))
