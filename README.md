# Machine_Learning_XGBoost_in_R
XGBoost is a very powerful tool for classification and regression.
# Setting up environment
First, let's read in the libraries we're going to use. we need to install if it is not installed earlier (xgboost & tidyverse)

Read in our data & put it in a data frame: we're going to be using a dataset from the Food and Agriculture Organization of the United Nations that contains information on various outbreaks of animal diseases. We're going to try to predict which outbreaks of animal diseases will lead to humans getting sick

Set a random seed & shuffle data frame: so that we can split our data into a testing set and training set using the row numbers, we will get a random sample of data in both the testing and training set.

# Preparing our data & selecting features
The core xgboost function requires data to be a matrix. However, our data isn't currently in a matrix. In fact, many of our features aren't numeric at all! so we should check first few rows of our dataframe and see what it looks like.

As per the result we got, our data will need some cleaning before it's ready to be put in a matrix. To prepare our data, we have a number of steps we need to complete:

. Remove information about the target variable from the training data
. Reduce the amount of redundant information
. Convert categorical information (like country) to a numeric format
. Split dataset into testing and training subsets
. Convert the cleaned dataframe to a Dmatrix
 
# check out the first few lines
> head(diseaseLabels) # of our target variable
     humansAffected
[1,]           TRUE
[2,]          FALSE
[3,]          FALSE
[4,]          FALSE
[5,]          FALSE
[6,]          FALSE
head(diseaseInfo$humansAffected) # of the original column
[1]  1 NA NA NA NA NA
# select just the numeric columns
diseaseInfo_numeric <- diseaseInfo_humansRemoved %>%
+   select(-Id) %>% # the case id shouldn't contain useful information
+   select(-c(longitude, latitude)) %>% # location data is also in country data
+   select_if(is.numeric) # select remaining numeric columns

# make sure that our dataframe is all numeric
str(diseaseInfo_numeric)
tibble [17,008 x 5] (S3: tbl_df/tbl/data.frame)
 $ sumAtRisk     : num [1:17008] NA 53 NA 61 93 12 103 49 13 NA ...
 $ sumCases      : num [1:17008] NA 4 1 1 1 NA 1 9 10 1 ...
 $ sumDeaths     : num [1:17008] NA 0 1 0 0 6 NA 0 10 0 ...
 $ sumDestroyed  : num [1:17008] NA 0 0 0 0 6 NA 0 3 1 ...
 $ sumSlaughtered: num [1:17008] NA 0 0 0 0 NA NA 0 0 0 ...
# check out the first few rows of the country column
head(diseaseInfo$country)
[1] "Saudi Arabia"      "Italy"             "Poland"            "Tunisia"           "France"           
[6] "Republic of Korea"
# one-hot matrix for just the first few rows of the "country" column
model.matrix(~country-1,head(diseaseInfo))
  countryFrance countryItaly countryPoland countryRepublic of Korea countrySaudi Arabia countryTunisia
1             0            0             0                        0                   1              0
2             0            1             0                        0                   0              0
3             0            0             1                        0                   0              0
4             0            0             0                        0                   0              1
5             1            0             0                        0                   0              0
6             0            0             0                        1                   0              0
attr(,"assign")
[1] 1 1 1 1 1 1
attr(,"contrasts")
attr(,"contrasts")$country
[1] "contr.treatment"

# convert categorical factor into one-hot encoding
region <- model.matrix(~country-1,diseaseInfo)
# some of the species
head(diseaseInfo$speciesDescription)
[1] NA                                                  "domestic, cattle"                                 
[3] "wild, wild boar"                                   "domestic, cattle, domestic, goat, domestic, sheep"
[5] "domestic, cattle"                                  "domestic, unspecified bird"                       
# add a boolean column to our numeric dataframe indicating whether a species is domestic
diseaseInfo_numeric$is_domestic <- str_detect(diseaseInfo$speciesDescription, "domestic")
# get a list of all the species by getting the last
speciesList <- diseaseInfo$speciesDescription %>%
+   str_replace("[[:punct:]]", "") %>% # remove punctuation (some rows have parentheses)
+   str_extract("[a-z]*$") # extract the least word in each row

# convert our list into a dataframe...
speciesList <- tibble(species = speciesList)

# and convert to a matrix using 1 hot encoding
options(na.action='na.pass') # don't drop NA values!
species <- model.matrix(~species-1,speciesList)
# add our one-hot encoded variable and convert the dataframe into a matrix
diseaseInfo_numeric <- cbind(diseaseInfo_numeric, region, species)
diseaseInfo_matrix <- data.matrix(diseaseInfo_numeric)
# get the numb 70/30 training test split
numberOfTrainingSamples <- round(length(diseaseLabels) * .7)

# training data
train_data <- diseaseInfo_matrix[1:numberOfTrainingSamples,]
train_labels <- diseaseLabels[1:numberOfTrainingSamples]

# testing data
test_data <- diseaseInfo_matrix[-(1:numberOfTrainingSamples),]
test_labels <- diseaseLabels[-(1:numberOfTrainingSamples)]
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)
> # train a model using our training data
> model <- xgboost(data = dtrain, # the data   
+                  nround = 2, # max number of boosting iterations
+                  objective = "binary:logistic")  # the objective function
[21:50:59] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[1]	train-logloss:0.448429 
[2]	train-logloss:0.313402 
# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
[1] "test-error= 0.0139161113288906"
# train an xgboost model
model_tuned <- xgboost(data = dtrain, # the data           
+                        max.depth = 3, # the maximum depth of each decision tree
+                        nround = 2, # max number of boosting iterations
+                        objective = "binary:logistic") # the objective function 
[21:52:17] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[1]	train-logloss:0.448429 
[2]	train-logloss:0.313402 
 
# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
[1] "test-error= 0.0139161113288906"
# get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
+                        max.depth = 3, # the maximum depth of each decision tree
+                        nround = 10, # number of boosting rounds
+                        early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
+                        objective = "binary:logistic", # the objective function
+                        scale_pos_weight = negative_cases/postive_cases) # control for imbalanced classes
[21:52:52] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[1]	train-logloss:0.446740 
Will train until train_logloss hasn't improved in 3 rounds.

[2]	train-logloss:0.311939 
[3]	train-logloss:0.228258 
[4]	train-logloss:0.173351 
[5]	train-logloss:0.136223 
[6]	train-logloss:0.110756 
[7]	train-logloss:0.093108 
[8]	train-logloss:0.080866 
[9]	train-logloss:0.072260 
[10]	train-logloss:0.066269 

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
[1] "test-error= 0.0139161113288906"
# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
+                        max.depth = 3, # the maximum depth of each decision tree
+                        nround = 10, # number of boosting rounds
+                        early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
+                        objective = "binary:logistic", # the objective function
+                        scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
+                        gamma = 1) # add a regularization term
[21:53:28] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[1]	train-logloss:0.446740 
Will train until train_logloss hasn't improved in 3 rounds.

[2]	train-logloss:0.311927 
[3]	train-logloss:0.228251 
[4]	train-logloss:0.173355 
[5]	train-logloss:0.136234 
[6]	train-logloss:0.110763 
[7]	train-logloss:0.093123 
[8]	train-logloss:0.080873 
[9]	train-logloss:0.072273 
[10]	train-logloss:0.066278 

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
[1] "test-error= 0.0139161113288906"

# plot them features! what's contributing most to our model?
xgb.plot.multi.trees(feature_names = names(diseaseInfo_matrix), 
+                      model = model)
Column 2 ['No'] of item 2 is missing in item 1. Use fill=TRUE to fill with NA (NULL for list columns), or use.names=FALSE to ignore column names. use.names='check' (default from v1.12.2) emits this message and proceeds as if use.names=FALSE for  backwards compatibility. See news item 5 in v1.12.2 for options to control this message.
# convert log odds to probability
odds_to_probs <- function(odds){
+   return(exp(odds)/ (1 + exp(odds)))
+ }
 

# probability of leaf above countryPortugul
odds_to_probs(-0.599)
[1] 0.3545725
# get information on how important each feature is
importance_matrix <- xgb.importance(names(diseaseInfo_matrix), model = model)

# and plot it!
xgb.plot.importance(importance_matrix)
