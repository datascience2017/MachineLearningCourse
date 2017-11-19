### Data Information

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

We were provided with two files: training data: pml-training and testing data: pml-testing.

### Loading data

``` r
training <- read.csv("pml-training.csv",header = TRUE, sep = ",")
testing <- read.csv("pml-testing.csv",header = TRUE, sep = ",")
```

### Data Cleaning

We have 19622 obersvations and 160 variables in oour training dataset. We need to understand the amount of missing values in our training data. Any variable missing more than 50% should not be used because they can give a false impression of the relationship with the dependent and hence can pollute the model. In our study we will eliminate all variables with any missing values to be on the conservative side. Count of missing values for each variable

``` r
Missing_Values <- sapply(training,function(x) sum(is.na(x)))
MissingValues <- as.data.frame(Missing_Values)
name <- row.names(MissingValues)
count_missing <- cbind(name,MissingValues)
row.names(count_missing) <- NULL
colnames(count_missing) <- c("variable", "Count Missing")
```

We will now filter out the data with any missing values.

``` r
Newdata <-  as.character(count_missing$variable[which(count_missing$`Count Missing`<=1)])
training <- training[Newdata]
```

We have now filtered all the missing variables and now have 19622 obervations and 93 variables in our training dataset.

Removing irrelevant variables from our dataset as they do not give us any useful information related to the prediction/interest. We will remove first seven columns from the training and testing data sets because they do not contain any relevant information that will help our prediction outcome.

``` r
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

Removing variables with no variability or variables with zero covariates in training data set.

``` r
training <-training[,-nearZeroVar(training)]
```

we have now filtered our training dataset and have 19622 observartions and 53 variables.

### Data Partition

Splitting the training data set into training and validation datasets by retaining 70% data in training set and 30% in validation set.

``` r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
train_new <- training[inTrain,]
val <- training[-inTrain,]
```

### Building the Model

**Model 1- Random Forest Algorithm with 3 fold cross validation**

``` r
traincontrol <- trainControl(method="cv", number=3,verboseIter=FALSE, allowParallel = TRUE)
model <- train(classe~., data=train_new, method="rf", trControl = traincontrol)
pred <- predict(model,train_new)
rfc <- confusionMatrix(pred,train_new$classe)
```

    ## [1] "The overall accuracy when predicting on training data with random forest selected model is 1"

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-8-1.png)

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.73%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3901    3    2    0    0 0.001280082
    ## B   22 2628    7    1    0 0.011286682
    ## C    0   14 2376    6    0 0.008347245
    ## D    0    1   31 2217    3 0.015541741
    ## E    0    1    2    7 2515 0.003960396

### Model 2- Boosting:gmb

``` r
mod_boost <- train(classe~.,data=train_new,method="gbm",verbose = FALSE,trControl=traincontrol)
predboost <- predict(mod_boost,train_new)
c <- confusionMatrix(predboost,train_new$classe)
plot(mod_boost)
```

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
print(mod_boost$finalModel)
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 52 predictors of which 43 had non-zero influence.

    ## [1] "The overall accuracy when predicting on training data with boosting:gbm model is 0.974739753949188"

We can see from the plot that the accuracy increases with the maximum tree depth and number of boosting iterations.

### Model 3-Linear Discriminant Analysis

``` r
mod_lda <- train(classe~.,data=train_new,method="lda",verbose = FALSE,trControl=traincontrol)
pre_lda <- predict(mod_lda,train_new)
clda <- confusionMatrix(pre_lda,train_new$classe)
```

    ## [1] "The overall accuracy when predicting on training data with lda model is 0.708087646502148"

### Model 4-Classification Tree

``` r
mod_rpart <- train(classe~.,data=train_new,method="rpart",trControl=traincontrol)
pre_rpart <- predict(mod_rpart,train_new)
crpart <- confusionMatrix(pre_rpart,train_new$classe)
fancyRpartPlot(mod_rpart$finalModel)
```

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-13-1.png)

    ## [1] "The overall accuracy when predicting on training data with classification trees model is 0.495231855572541"

*We now have 4 trained models and we will be evaluating the accuracy of each of them.*

Cross Validation
----------------

### Running the models on the validation data set and predicting values on the validation data set.

    ## [1] "The overall accuracy when predicting on validation data with random forest model is 0.994"

    ## [1] "The overall accuracy when predicting on validation data with boosting:gbm model is 0.96"

    ## [1] "The overall accuracy when predicting on validation data with lda model is 0.699"

    ## [1] "The overall accuracy when predicting on validation data with Classification Trees model is 0.496"

    ##                 Model  Accuracy
    ## 1        RandomForest 0.9938828
    ## 2                 GBM 0.9600680
    ## 3                 LDA 0.6990654
    ## 4 Classification Tree 0.4956669

### Conclusion

*Based on the Overall Accuracy numbers, we can see that the Random Forest model gives us the most accurate results as compared to Classification Tree model and Linear Discriminant Analysis model. Hence we can accept the Random Forest Model for our study.*

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    8    0    0    0
    ##          B    1 1129    3    0    0
    ##          C    0    2 1017   10    4
    ##          D    0    0    6  953    0
    ##          E    1    0    0    1 1078

    ## [1] "The In Sample Error Rate for the selected model is 0.006"

Appendix
--------

``` r
knitr::opts_chunk$set(echo = TRUE, warning=FALSE, message=FALSE, fig.width=10, fig.height=5)
options(width=120)
suppressMessages(library(rattle))
suppressMessages(library(caret))
suppressMessages(library(forecast))
suppressMessages(library(AER))
suppressMessages(library(ggplot2))
suppressMessages(library(AppliedPredictiveModeling))
suppressMessages(library(kernlab))
suppressMessages(library(randomForest))
suppressMessages(library(parallel))
suppressMessages(library(rpart.plot))
suppressMessages(library(splines))
suppressMessages(library(gbm))
suppressMessages(library(MASS))
suppressMessages(library(doParallel))

training <- read.csv("pml-training.csv",header = TRUE, sep = ",")
testing <- read.csv("pml-testing.csv",header = TRUE, sep = ",")

Missing_Values <- sapply(training,function(x) sum(is.na(x)))
MissingValues <- as.data.frame(Missing_Values)
name <- row.names(MissingValues)
count_missing <- cbind(name,MissingValues)
row.names(count_missing) <- NULL
colnames(count_missing) <- c("variable", "Count Missing")

Newdata <-  as.character(count_missing$variable[which(count_missing$`Count Missing`<=1)])
training <- training[Newdata]

training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]


training <-training[,-nearZeroVar(training)]

inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
train_new <- training[inTrain,]
val <- training[-inTrain,]

traincontrol <- trainControl(method="cv", number=3,verboseIter=FALSE, allowParallel = TRUE)
model <- train(classe~., data=train_new, method="rf", trControl = traincontrol)
pred <- predict(model,train_new)
rfc <- confusionMatrix(pred,train_new$classe)

paste0("The overall accuracy when predicting on training data with random forest selected model is ",rfc$overall[1])
```

    ## [1] "The overall accuracy when predicting on training data with random forest selected model is 1"

``` r
mod_boost <- train(classe~.,data=train_new,method="gbm",verbose = FALSE,trControl=traincontrol)
predboost <- predict(mod_boost,train_new)
c <- confusionMatrix(predboost,train_new$classe)

paste0("The overall accuracy when predicting on training data with boosting:gbm model is ",c$overall[1])
```

    ## [1] "The overall accuracy when predicting on training data with boosting:gbm model is 0.974739753949188"

``` r
mod_lda <- train(classe~.,data=train_new,method="lda",verbose = FALSE,trControl=traincontrol)
pre_lda <- predict(mod_lda,train_new)
clda <- confusionMatrix(pre_lda,train_new$classe)

paste0("The overall accuracy when predicting on training data with lda model is ",clda$overall[1])
```

    ## [1] "The overall accuracy when predicting on training data with lda model is 0.704593433791949"

``` r
mod_rpart <- train(classe~.,data=train_new,method="rpart",trControl=traincontrol)
pre_rpart <- predict(mod_rpart,train_new)
crpart <- confusionMatrix(pre_rpart,train_new$classe)

paste0("The overall accuracy when predicting on training data with classification trees model is ",crpart$overall[1])
```

    ## [1] "The overall accuracy when predicting on training data with classification trees model is 0.492028827254859"

``` r
pred_val_rf <- predict(model,val)
rfc_val <- confusionMatrix(pred_val_rf,val$classe)
paste0("The overall accuracy when predicting on validation data with random forest model is ",round(rfc_val$overall[1],3))
```

    ## [1] "The overall accuracy when predicting on validation data with random forest model is 0.995"

``` r
predboost_val <- predict(mod_boost,val)
c_val <- confusionMatrix(predboost_val,val$classe)
paste0("The overall accuracy when predicting on validation data with boosting:gbm model is ",round(c_val$overall[1],3))
```

    ## [1] "The overall accuracy when predicting on validation data with boosting:gbm model is 0.967"

``` r
pred_lda_val <- predict(mod_lda,val)
clda_val <- confusionMatrix(pred_lda_val,val$classe)
paste0("The overall accuracy when predicting on validation data with lda model is ",round(clda_val$overall[1],3))
```

    ## [1] "The overall accuracy when predicting on validation data with lda model is 0.703"

``` r
pred_rpart_val <- predict(mod_rpart,val)
crpart_val <- confusionMatrix(pred_rpart_val,val$classe)
paste0("The overall accuracy when predicting on validation data with Classification Trees model is ",round(crpart_val$overall[1],3))
```

    ## [1] "The overall accuracy when predicting on validation data with Classification Trees model is 0.504"

``` r
ModelAccuracyTable <- data.frame(Model=c("RandomForest", "GBM", "LDA", "Classification Tree"),
                            Accuracy=rbind(rfc_val$overall[1],c_val$overall[1],clda_val$overall[1],crpart_val$overall[1]))
print(ModelAccuracyTable)
```

    ##                 Model  Accuracy
    ## 1        RandomForest 0.9947324
    ## 2                 GBM 0.9673747
    ## 3                 LDA 0.7029737
    ## 4 Classification Tree 0.5036534

``` r
print(rfc_val$table)
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    5    0    0    0
    ##          B    1 1130    1    1    0
    ##          C    0    4 1019    8    3
    ##          D    0    0    6  954    1
    ##          E    0    0    0    1 1078

``` r
paste0("The In Sample Error Rate for the selected model is ",round(1-rfc_val[["overall"]][["Accuracy"]],3))
```

    ## [1] "The In Sample Error Rate for the selected model is 0.005"
