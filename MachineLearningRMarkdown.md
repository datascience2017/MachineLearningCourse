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

**Model 1- Random Forest Algorithm with 5 fold cross validation**

``` r
traincontrol <- trainControl(method="cv", number=3)
model <- train(classe~., data=train_new, method="rf", trControl = traincontrol)
pred <- predict(model,train_new)
rfc <- confusionMatrix(pred,train_new$classe)
```

    ## [1] "The overall accuracy when predicting on training data with random forest selected model is 1"

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-8-1.png)

### Model 2- Boosting:gmb

``` r
mod_boost <- train(classe~.,data=train_new,method="gbm",trControl=traincontrol)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1272
    ##      2        1.5238             nan     0.1000    0.0850
    ##      3        1.4661             nan     0.1000    0.0681
    ##      4        1.4223             nan     0.1000    0.0546
    ##      5        1.3846             nan     0.1000    0.0464
    ##      6        1.3536             nan     0.1000    0.0463
    ##      7        1.3239             nan     0.1000    0.0350
    ##      8        1.3011             nan     0.1000    0.0339
    ##      9        1.2791             nan     0.1000    0.0312
    ##     10        1.2590             nan     0.1000    0.0268
    ##     20        1.1019             nan     0.1000    0.0177
    ##     40        0.9260             nan     0.1000    0.0089
    ##     60        0.8166             nan     0.1000    0.0057
    ##     80        0.7357             nan     0.1000    0.0038
    ##    100        0.6730             nan     0.1000    0.0034
    ##    120        0.6199             nan     0.1000    0.0021
    ##    140        0.5783             nan     0.1000    0.0023
    ##    150        0.5589             nan     0.1000    0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1842
    ##      2        1.4902             nan     0.1000    0.1295
    ##      3        1.4058             nan     0.1000    0.1036
    ##      4        1.3393             nan     0.1000    0.0797
    ##      5        1.2860             nan     0.1000    0.0792
    ##      6        1.2353             nan     0.1000    0.0621
    ##      7        1.1963             nan     0.1000    0.0573
    ##      8        1.1590             nan     0.1000    0.0503
    ##      9        1.1250             nan     0.1000    0.0489
    ##     10        1.0930             nan     0.1000    0.0427
    ##     20        0.8860             nan     0.1000    0.0196
    ##     40        0.6738             nan     0.1000    0.0092
    ##     60        0.5453             nan     0.1000    0.0038
    ##     80        0.4563             nan     0.1000    0.0054
    ##    100        0.3916             nan     0.1000    0.0057
    ##    120        0.3404             nan     0.1000    0.0031
    ##    140        0.2981             nan     0.1000    0.0027
    ##    150        0.2796             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2359
    ##      2        1.4585             nan     0.1000    0.1615
    ##      3        1.3546             nan     0.1000    0.1244
    ##      4        1.2759             nan     0.1000    0.1097
    ##      5        1.2055             nan     0.1000    0.0855
    ##      6        1.1510             nan     0.1000    0.0712
    ##      7        1.1056             nan     0.1000    0.0690
    ##      8        1.0614             nan     0.1000    0.0657
    ##      9        1.0203             nan     0.1000    0.0546
    ##     10        0.9857             nan     0.1000    0.0429
    ##     20        0.7510             nan     0.1000    0.0230
    ##     40        0.5266             nan     0.1000    0.0123
    ##     60        0.4022             nan     0.1000    0.0087
    ##     80        0.3161             nan     0.1000    0.0044
    ##    100        0.2609             nan     0.1000    0.0030
    ##    120        0.2156             nan     0.1000    0.0024
    ##    140        0.1839             nan     0.1000    0.0017
    ##    150        0.1694             nan     0.1000    0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1241
    ##      2        1.5239             nan     0.1000    0.0905
    ##      3        1.4651             nan     0.1000    0.0670
    ##      4        1.4213             nan     0.1000    0.0551
    ##      5        1.3855             nan     0.1000    0.0507
    ##      6        1.3527             nan     0.1000    0.0395
    ##      7        1.3263             nan     0.1000    0.0404
    ##      8        1.3003             nan     0.1000    0.0374
    ##      9        1.2760             nan     0.1000    0.0298
    ##     10        1.2563             nan     0.1000    0.0274
    ##     20        1.0995             nan     0.1000    0.0157
    ##     40        0.9285             nan     0.1000    0.0095
    ##     60        0.8190             nan     0.1000    0.0051
    ##     80        0.7385             nan     0.1000    0.0048
    ##    100        0.6750             nan     0.1000    0.0035
    ##    120        0.6236             nan     0.1000    0.0049
    ##    140        0.5798             nan     0.1000    0.0012
    ##    150        0.5594             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1804
    ##      2        1.4903             nan     0.1000    0.1332
    ##      3        1.4061             nan     0.1000    0.1048
    ##      4        1.3392             nan     0.1000    0.0843
    ##      5        1.2865             nan     0.1000    0.0748
    ##      6        1.2396             nan     0.1000    0.0646
    ##      7        1.1978             nan     0.1000    0.0542
    ##      8        1.1622             nan     0.1000    0.0559
    ##      9        1.1276             nan     0.1000    0.0448
    ##     10        1.0989             nan     0.1000    0.0482
    ##     20        0.8914             nan     0.1000    0.0198
    ##     40        0.6763             nan     0.1000    0.0103
    ##     60        0.5503             nan     0.1000    0.0088
    ##     80        0.4646             nan     0.1000    0.0047
    ##    100        0.3970             nan     0.1000    0.0031
    ##    120        0.3458             nan     0.1000    0.0037
    ##    140        0.3049             nan     0.1000    0.0023
    ##    150        0.2862             nan     0.1000    0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2375
    ##      2        1.4608             nan     0.1000    0.1647
    ##      3        1.3558             nan     0.1000    0.1257
    ##      4        1.2762             nan     0.1000    0.1117
    ##      5        1.2056             nan     0.1000    0.0825
    ##      6        1.1537             nan     0.1000    0.0677
    ##      7        1.1106             nan     0.1000    0.0635
    ##      8        1.0705             nan     0.1000    0.0720
    ##      9        1.0240             nan     0.1000    0.0623
    ##     10        0.9828             nan     0.1000    0.0534
    ##     20        0.7535             nan     0.1000    0.0243
    ##     40        0.5263             nan     0.1000    0.0112
    ##     60        0.4035             nan     0.1000    0.0057
    ##     80        0.3214             nan     0.1000    0.0024
    ##    100        0.2672             nan     0.1000    0.0021
    ##    120        0.2239             nan     0.1000    0.0025
    ##    140        0.1894             nan     0.1000    0.0020
    ##    150        0.1750             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1230
    ##      2        1.5244             nan     0.1000    0.0867
    ##      3        1.4662             nan     0.1000    0.0659
    ##      4        1.4222             nan     0.1000    0.0515
    ##      5        1.3882             nan     0.1000    0.0442
    ##      6        1.3583             nan     0.1000    0.0435
    ##      7        1.3305             nan     0.1000    0.0421
    ##      8        1.3042             nan     0.1000    0.0358
    ##      9        1.2811             nan     0.1000    0.0334
    ##     10        1.2583             nan     0.1000    0.0274
    ##     20        1.1072             nan     0.1000    0.0172
    ##     40        0.9321             nan     0.1000    0.0092
    ##     60        0.8243             nan     0.1000    0.0063
    ##     80        0.7418             nan     0.1000    0.0035
    ##    100        0.6786             nan     0.1000    0.0029
    ##    120        0.6254             nan     0.1000    0.0023
    ##    140        0.5825             nan     0.1000    0.0023
    ##    150        0.5641             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1889
    ##      2        1.4890             nan     0.1000    0.1255
    ##      3        1.4079             nan     0.1000    0.1013
    ##      4        1.3414             nan     0.1000    0.0840
    ##      5        1.2861             nan     0.1000    0.0713
    ##      6        1.2403             nan     0.1000    0.0705
    ##      7        1.1962             nan     0.1000    0.0599
    ##      8        1.1583             nan     0.1000    0.0473
    ##      9        1.1275             nan     0.1000    0.0506
    ##     10        1.0941             nan     0.1000    0.0367
    ##     20        0.8925             nan     0.1000    0.0237
    ##     40        0.6784             nan     0.1000    0.0118
    ##     60        0.5484             nan     0.1000    0.0071
    ##     80        0.4581             nan     0.1000    0.0042
    ##    100        0.3924             nan     0.1000    0.0035
    ##    120        0.3393             nan     0.1000    0.0023
    ##    140        0.2970             nan     0.1000    0.0025
    ##    150        0.2777             nan     0.1000    0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2284
    ##      2        1.4613             nan     0.1000    0.1595
    ##      3        1.3601             nan     0.1000    0.1267
    ##      4        1.2788             nan     0.1000    0.1080
    ##      5        1.2109             nan     0.1000    0.0797
    ##      6        1.1574             nan     0.1000    0.0798
    ##      7        1.1070             nan     0.1000    0.0661
    ##      8        1.0640             nan     0.1000    0.0531
    ##      9        1.0284             nan     0.1000    0.0662
    ##     10        0.9873             nan     0.1000    0.0407
    ##     20        0.7538             nan     0.1000    0.0253
    ##     40        0.5251             nan     0.1000    0.0099
    ##     60        0.3947             nan     0.1000    0.0044
    ##     80        0.3159             nan     0.1000    0.0060
    ##    100        0.2553             nan     0.1000    0.0032
    ##    120        0.2140             nan     0.1000    0.0027
    ##    140        0.1811             nan     0.1000    0.0019
    ##    150        0.1670             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2386
    ##      2        1.4600             nan     0.1000    0.1640
    ##      3        1.3574             nan     0.1000    0.1182
    ##      4        1.2816             nan     0.1000    0.1069
    ##      5        1.2147             nan     0.1000    0.0908
    ##      6        1.1561             nan     0.1000    0.0807
    ##      7        1.1053             nan     0.1000    0.0664
    ##      8        1.0635             nan     0.1000    0.0543
    ##      9        1.0291             nan     0.1000    0.0656
    ##     10        0.9891             nan     0.1000    0.0562
    ##     20        0.7578             nan     0.1000    0.0290
    ##     40        0.5290             nan     0.1000    0.0120
    ##     60        0.4068             nan     0.1000    0.0072
    ##     80        0.3241             nan     0.1000    0.0060
    ##    100        0.2668             nan     0.1000    0.0040
    ##    120        0.2234             nan     0.1000    0.0026
    ##    140        0.1894             nan     0.1000    0.0018
    ##    150        0.1755             nan     0.1000    0.0012

``` r
predboost <- predict(mod_boost,train_new)
c <- confusionMatrix(predboost,train_new$classe)
plot(mod_boost)
```

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-9-1.png)

    ## [1] "The overall accuracy when predicting on training data with boosting:gbm model is 0.975176530537963"

### Model 3-Linear Discriminant Analysis

``` r
mod_lda <- train(classe~.,data=train_new,method="lda",trControl=traincontrol)
pre_lda <- predict(mod_lda,train_new)
clda <- confusionMatrix(pre_lda,train_new$classe)
```

    ## [1] "The overall accuracy when predicting on training data with lda model is 0.703792676712528"

### Model 4-Classification Tree

``` r
mod_rpart <- train(classe~.,data=train_new,method="rpart",trControl=traincontrol)
pre_rpart <- predict(mod_rpart,train_new)
crpart <- confusionMatrix(pre_rpart,train_new$classe)
fancyRpartPlot(mod_rpart$finalModel)
```

![](MachineLearningRMarkdown_files/figure-markdown_github/unnamed-chunk-13-1.png)

    ## [1] "The overall accuracy when predicting on training data with classification trees model is 0.495959816553833"

*We now have 4 trained models and we will be evaluating the accuracy of each of them.*

Cross Validation
----------------

Predicting on the validation data set.

    ## [1] "The overall accuracy when predicting on validation data with random forest model is 0.992"

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1323
    ##      2        1.5233             nan     0.1000    0.0886
    ##      3        1.4656             nan     0.1000    0.0674
    ##      4        1.4203             nan     0.1000    0.0536
    ##      5        1.3835             nan     0.1000    0.0512
    ##      6        1.3507             nan     0.1000    0.0456
    ##      7        1.3219             nan     0.1000    0.0346
    ##      8        1.2988             nan     0.1000    0.0369
    ##      9        1.2751             nan     0.1000    0.0355
    ##     10        1.2515             nan     0.1000    0.0298
    ##     20        1.0976             nan     0.1000    0.0182
    ##     40        0.9264             nan     0.1000    0.0080
    ##     60        0.8183             nan     0.1000    0.0069
    ##     80        0.7397             nan     0.1000    0.0054
    ##    100        0.6777             nan     0.1000    0.0040
    ##    120        0.6241             nan     0.1000    0.0039
    ##    140        0.5786             nan     0.1000    0.0017
    ##    150        0.5585             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1887
    ##      2        1.4877             nan     0.1000    0.1319
    ##      3        1.4042             nan     0.1000    0.1080
    ##      4        1.3360             nan     0.1000    0.0854
    ##      5        1.2808             nan     0.1000    0.0672
    ##      6        1.2377             nan     0.1000    0.0657
    ##      7        1.1961             nan     0.1000    0.0676
    ##      8        1.1545             nan     0.1000    0.0584
    ##      9        1.1171             nan     0.1000    0.0452
    ##     10        1.0878             nan     0.1000    0.0405
    ##     20        0.8851             nan     0.1000    0.0209
    ##     40        0.6741             nan     0.1000    0.0131
    ##     60        0.5503             nan     0.1000    0.0060
    ##     80        0.4650             nan     0.1000    0.0046
    ##    100        0.4002             nan     0.1000    0.0034
    ##    120        0.3470             nan     0.1000    0.0021
    ##    140        0.3048             nan     0.1000    0.0033
    ##    150        0.2867             nan     0.1000    0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2341
    ##      2        1.4570             nan     0.1000    0.1604
    ##      3        1.3554             nan     0.1000    0.1282
    ##      4        1.2735             nan     0.1000    0.1058
    ##      5        1.2076             nan     0.1000    0.0832
    ##      6        1.1525             nan     0.1000    0.0724
    ##      7        1.1051             nan     0.1000    0.0660
    ##      8        1.0621             nan     0.1000    0.0649
    ##      9        1.0209             nan     0.1000    0.0548
    ##     10        0.9862             nan     0.1000    0.0457
    ##     20        0.7475             nan     0.1000    0.0199
    ##     40        0.5195             nan     0.1000    0.0098
    ##     60        0.3933             nan     0.1000    0.0062
    ##     80        0.3139             nan     0.1000    0.0055
    ##    100        0.2603             nan     0.1000    0.0034
    ##    120        0.2168             nan     0.1000    0.0020
    ##    140        0.1841             nan     0.1000    0.0016
    ##    150        0.1710             nan     0.1000    0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1269
    ##      2        1.5242             nan     0.1000    0.0824
    ##      3        1.4670             nan     0.1000    0.0673
    ##      4        1.4232             nan     0.1000    0.0544
    ##      5        1.3881             nan     0.1000    0.0431
    ##      6        1.3595             nan     0.1000    0.0463
    ##      7        1.3293             nan     0.1000    0.0387
    ##      8        1.3035             nan     0.1000    0.0311
    ##      9        1.2814             nan     0.1000    0.0346
    ##     10        1.2596             nan     0.1000    0.0300
    ##     20        1.1035             nan     0.1000    0.0172
    ##     40        0.9280             nan     0.1000    0.0099
    ##     60        0.8199             nan     0.1000    0.0056
    ##     80        0.7384             nan     0.1000    0.0052
    ##    100        0.6748             nan     0.1000    0.0041
    ##    120        0.6233             nan     0.1000    0.0037
    ##    140        0.5795             nan     0.1000    0.0019
    ##    150        0.5605             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1823
    ##      2        1.4915             nan     0.1000    0.1255
    ##      3        1.4100             nan     0.1000    0.1047
    ##      4        1.3435             nan     0.1000    0.0852
    ##      5        1.2898             nan     0.1000    0.0802
    ##      6        1.2398             nan     0.1000    0.0597
    ##      7        1.2003             nan     0.1000    0.0558
    ##      8        1.1642             nan     0.1000    0.0532
    ##      9        1.1311             nan     0.1000    0.0430
    ##     10        1.1035             nan     0.1000    0.0442
    ##     20        0.8948             nan     0.1000    0.0216
    ##     40        0.6772             nan     0.1000    0.0129
    ##     60        0.5512             nan     0.1000    0.0067
    ##     80        0.4635             nan     0.1000    0.0041
    ##    100        0.3964             nan     0.1000    0.0032
    ##    120        0.3426             nan     0.1000    0.0034
    ##    140        0.3033             nan     0.1000    0.0016
    ##    150        0.2856             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2286
    ##      2        1.4629             nan     0.1000    0.1616
    ##      3        1.3600             nan     0.1000    0.1219
    ##      4        1.2827             nan     0.1000    0.1006
    ##      5        1.2176             nan     0.1000    0.1004
    ##      6        1.1562             nan     0.1000    0.0720
    ##      7        1.1095             nan     0.1000    0.0664
    ##      8        1.0665             nan     0.1000    0.0665
    ##      9        1.0254             nan     0.1000    0.0504
    ##     10        0.9924             nan     0.1000    0.0538
    ##     20        0.7593             nan     0.1000    0.0227
    ##     40        0.5332             nan     0.1000    0.0095
    ##     60        0.4101             nan     0.1000    0.0063
    ##     80        0.3223             nan     0.1000    0.0027
    ##    100        0.2608             nan     0.1000    0.0024
    ##    120        0.2188             nan     0.1000    0.0019
    ##    140        0.1840             nan     0.1000    0.0010
    ##    150        0.1711             nan     0.1000    0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1210
    ##      2        1.5251             nan     0.1000    0.0888
    ##      3        1.4669             nan     0.1000    0.0685
    ##      4        1.4229             nan     0.1000    0.0527
    ##      5        1.3884             nan     0.1000    0.0507
    ##      6        1.3561             nan     0.1000    0.0456
    ##      7        1.3269             nan     0.1000    0.0363
    ##      8        1.3032             nan     0.1000    0.0353
    ##      9        1.2810             nan     0.1000    0.0278
    ##     10        1.2631             nan     0.1000    0.0291
    ##     20        1.1050             nan     0.1000    0.0168
    ##     40        0.9295             nan     0.1000    0.0092
    ##     60        0.8242             nan     0.1000    0.0064
    ##     80        0.7406             nan     0.1000    0.0051
    ##    100        0.6779             nan     0.1000    0.0039
    ##    120        0.6242             nan     0.1000    0.0029
    ##    140        0.5806             nan     0.1000    0.0019
    ##    150        0.5616             nan     0.1000    0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1887
    ##      2        1.4872             nan     0.1000    0.1236
    ##      3        1.4063             nan     0.1000    0.0989
    ##      4        1.3417             nan     0.1000    0.0836
    ##      5        1.2874             nan     0.1000    0.0722
    ##      6        1.2426             nan     0.1000    0.0706
    ##      7        1.1967             nan     0.1000    0.0587
    ##      8        1.1593             nan     0.1000    0.0593
    ##      9        1.1216             nan     0.1000    0.0431
    ##     10        1.0933             nan     0.1000    0.0417
    ##     20        0.8892             nan     0.1000    0.0196
    ##     40        0.6780             nan     0.1000    0.0098
    ##     60        0.5520             nan     0.1000    0.0061
    ##     80        0.4593             nan     0.1000    0.0029
    ##    100        0.3939             nan     0.1000    0.0045
    ##    120        0.3404             nan     0.1000    0.0024
    ##    140        0.2988             nan     0.1000    0.0031
    ##    150        0.2800             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2345
    ##      2        1.4586             nan     0.1000    0.1558
    ##      3        1.3569             nan     0.1000    0.1285
    ##      4        1.2774             nan     0.1000    0.0990
    ##      5        1.2133             nan     0.1000    0.0926
    ##      6        1.1521             nan     0.1000    0.0842
    ##      7        1.0997             nan     0.1000    0.0600
    ##      8        1.0600             nan     0.1000    0.0606
    ##      9        1.0221             nan     0.1000    0.0591
    ##     10        0.9831             nan     0.1000    0.0493
    ##     20        0.7589             nan     0.1000    0.0348
    ##     40        0.5246             nan     0.1000    0.0116
    ##     60        0.3976             nan     0.1000    0.0069
    ##     80        0.3171             nan     0.1000    0.0048
    ##    100        0.2605             nan     0.1000    0.0044
    ##    120        0.2184             nan     0.1000    0.0045
    ##    140        0.1843             nan     0.1000    0.0009
    ##    150        0.1713             nan     0.1000    0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2356
    ##      2        1.4606             nan     0.1000    0.1579
    ##      3        1.3584             nan     0.1000    0.1265
    ##      4        1.2781             nan     0.1000    0.1124
    ##      5        1.2104             nan     0.1000    0.0913
    ##      6        1.1524             nan     0.1000    0.0707
    ##      7        1.1073             nan     0.1000    0.0800
    ##      8        1.0588             nan     0.1000    0.0565
    ##      9        1.0237             nan     0.1000    0.0534
    ##     10        0.9899             nan     0.1000    0.0557
    ##     20        0.7594             nan     0.1000    0.0230
    ##     40        0.5284             nan     0.1000    0.0110
    ##     60        0.4032             nan     0.1000    0.0062
    ##     80        0.3227             nan     0.1000    0.0056
    ##    100        0.2640             nan     0.1000    0.0046
    ##    120        0.2187             nan     0.1000    0.0037
    ##    140        0.1869             nan     0.1000    0.0024
    ##    150        0.1730             nan     0.1000    0.0010

    ## [1] "The overall accuracy when predicting on validation data with boosting:gbm model is 0.958"

    ## [1] "The overall accuracy when predicting on validation data with lda model is 0.707"

    ## [1] "The overall accuracy when predicting on validation data with Classification Trees model is 0.494"

    ##                 Model  Accuracy
    ## 1        RandomForest 0.9920136
    ## 2                 GBM 0.9581988
    ## 3                 LDA 0.7065421
    ## 4 Classification Tree 0.4941376

### Conclusion

*Based on the Overall Accuracy numbers, we can see that the Random Forest model gives us the most accurate results as compared to Classification Tree model and Linear Discriminant Analysis model. Hence we can accept the Random Forest Model for our study.*

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1669   14    0    0    0
    ##          B    5 1121    4    0    0
    ##          C    0    3 1018   14    2
    ##          D    0    0    4  950    0
    ##          E    0    1    0    0 1080

    ## [1] "The In Sample Error Rate for the selected model is 0.008"
