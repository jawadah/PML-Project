# Practical Machine Learning Course Project
Azeez, Jawad Ahmad

August 22, 2015
##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Data

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har]. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

##What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

# Initial Work
An overall random number  seed was set at 1234 for all the code used. In order to reproduce the results as was obtained for this Project, the same seed should be used.
Firstly, different packages were downloaded and installed, such as caret and randomForest. These should also be installed in order to reproduce the results below. 

## Packages, Libraries, Seed

For Installing packages, loading libraries, and setting the seed for reproduceability the following code were used:
### For Installing caret package
install.packages("caret")

### For Installing randomForest
install.packages("randomForest")

### For Installing rpart package
install.packages("rpart")

library(caret)
### Loading required package: lattice
### Loading required package: ggplot2

### Random forest for classification and regression
library(randomForest) 

### For Regressive Partitioning and Regression trees
library(rpart) 

### For Decision Tree plot
library(rpart.plot) 

### Also load RColorBrewer package
library(RColorBrewer)

### Also load rattle package
library(rattle)

### setting the overall seed for reproduceability
set.seed(1234)

Loading data sets and preliminary cleaning

First we want to load the data sets into R and make sure that missing values are coded correctly.
For this we will delete irrelevany variables.Results will be hidden from the report for clarity and space considerations.

The training data set can be found on the following URL

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

The testing data set can be found on the following URL:

testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

Load data to memory solely

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

# Partioning the training set into two

Partioning Training data set into two data sets, 60% for myTraining, 40% for myTesting:

inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)

#### [1] 11776    160
#### [1] 7846     160

# Cleaning the data
The following transformations were used to clean the data:

Transformation 1: Cleaning NearZeroVariance Variables Run this code to view possible NZV Variables:

myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)

Run this code to create another subset without NZV variables:

myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
"kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
"max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
"var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
"stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
"kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
"max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
"kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
"skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
"amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
"skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
"max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
"amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
"avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
"stddev_yaw_forearm", "var_yaw_forearm")

myTraining <- myTraining[!myNZVvars]

### To check the new Number of observations
dim(myTraining)

####  [1] 11776     100

Transformation 2: Killing first column of Dataset - ID Removing first ID variable so that it does not interfer with ML Algorithms:

myTraining <- myTraining[c(-1)]

Transformation 3: Cleaning Variables with too many NAs. For Variables that have more than a 60% threshold of NA’s I’m going to leave them out:

trainingV3 <- myTraining #creating another subset to iterate in loop

for(i in 1:length(myTraining)) { #for every column in the training dataset
        if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { #if n?? NAs > 60% of total observations
        for(j in 1:length(trainingV3)) {
            if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
                trainingV3 <- trainingV3[ , -j] #Remove that column
            }   
        } 
    }
}
#To check the new N?? of observations
dim(trainingV3)

#### [1] 11776     58

### Seting back to our set:
myTraining <- trainingV3
rm(trainingV3)

Now let us do the exact same 3 transformations but for our myTesting and testing data sets.

clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58]) #already with classe column removed
myTesting <- myTesting[clean1]
testing <- testing[clean2]

#To check the new Number of observations

dim(myTesting)

#### [1] 7846     58

### To check the new Number of observations
dim(testing)

#### [1] 20    57

In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.

for (i in 1:length(testing) ) {
        for(j in 1:length(myTraining)) {
        if( length( grep(names(myTraining[i]), names(testing)[j]) ) ==1)  {
            class(testing[j]) <- class(myTraining[i])
        }      
    }      
}

### And to make sure Coertion really worked, simple smart ass technique:

testing <- rbind(myTraining[2, -58] , testing) #note row 2 does not mean anything, this will be removed right.. now:
testing <- testing[-1,]

## Using ML algorithms for prediction: Decision Tree
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")

Note: to view the decision tree with fancy run this command:

fancyRpartPlot(modFitA1)


## Predicting:

predictionsA1 <- predict(modFitA1, myTesting, type = "class")
Using confusion Matrix to test results:

confusionMatrix(predictionsA1, myTesting$classe)

 Confusion Matrix and Statistics
###
###           Reference
### Prediction    A    B    C    D    E
###         A 2150   60    7    1    0
###        B   61 1260   69   64    0
###          C   21  188 1269  143    4
###         D    0   10   14  857   78
###       E    0    0    9  221 1360
###
### Overall Statistics
###                                         
###                Accuracy : 0.879         
###                 95% CI : (0.871, 0.886)
###     No Information Rate : 0.284         
###    P-Value [Acc > NIR] : <2e-16        
###                                      
###                   Kappa : 0.847         
###  Mcnemar's Test P-Value : NA            
###
###Statistics by Class:
### 
###                     Class: A Class: B Class: C Class: D Class: E
### Sensitivity             0.963    0.830    0.928    0.666    0.943
### Specificity             0.988    0.969    0.945    0.984    0.964
###Pos Pred Value          0.969    0.867    0.781    0.894    0.855
### Neg Pred Value          0.985    0.960    0.984    0.938    0.987
### Prevalence              0.284    0.193    0.174    0.164    0.184
###Detection Rate          0.274    0.161    0.162    0.109    0.173
###Detection Prevalence    0.283    0.185    0.207    0.122    0.203
### Balanced Accuracy       0.976    0.900    0.936    0.825    0.954

## Using ML algorithms for prediction: Random Forests

modFitB1 <- randomForest(classe ~. , data=myTraining)

### Predicting in-sample error:

predictionsB1 <- predict(modFitB1, myTesting, type = "class")

### Using confusion Matrix to test results:

confusionMatrix(predictionsB1, myTesting$classe)

### Confusion Matrix and Statistics
###
###          Reference
### Prediction    A    B    C    D    E
###         A 2231    2    0    0    0
###        B    1 1516    2    0    0
###         C    0    0 1366    3    0
###       D    0    0    0 1282    2
###       E    0    0    0    1 1440
###
### Overall Statistics
###                                        
###              Accuracy : 0.999         
###                 95% CI : (0.997, 0.999)
###   No Information Rate : 0.284         
###     P-Value [Acc > NIR] : <2e-16        
###                                         
###                 Kappa : 0.998         
###  Mcnemar's Test P-Value : NA            
###
### Statistics by Class:
###
###                      Class: A Class: B Class: C Class: D Class: E
### Sensitivity             1.000    0.999    0.999    0.997    0.999
### Specificity             1.000    1.000    1.000    1.000    1.000
### Pos Pred Value          0.999    0.998    0.998    0.998    0.999
### Neg Pred Value          1.000    1.000    1.000    0.999    1.000
### Prevalence              0.284    0.193    0.174    0.164    0.184
### Detection Rate          0.284    0.193    0.174    0.163    0.184
### Detection Prevalence    0.285    0.194    0.174    0.164    0.184
### Balanced Accuracy       1.000    0.999    0.999    0.998    0.999

## andom Forests yielded better Results, as expected

##Generating Files to submit as answers for the Assignment:

Finally, using the provided Test Set out-of-sample error.

For Random Forests we use the following formula, which yielded a much better prediction in in-sample:

predictionsB2 <- predict(modFitB1, testing, type = "class")
Function to generate files with predictions to submit for assignment

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)

 
