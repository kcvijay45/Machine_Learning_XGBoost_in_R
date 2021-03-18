# Machine_Learning_XGBoost_in_R
XGBoost is a very powerful tool for classification and regression.
# Setting up environment
First, let's read in the libraries we're going to use. we need to install if it is not installed earlier (xgboost & tidyverse)

Read in our data & put it in a data frame: we're going to be using a dataset from the Food and Agriculture Organization of the United Nations that contains information on various outbreaks of animal diseases. We're going to try to predict which outbreaks of animal diseases will lead to humans getting sick

Set a random seed & shuffle data frame: so that we can split our data into a testing set and training set using the row numbers, we will get a random sample of data in both the testing and training set.

# Preparing data & selecting features
The core xgboost function requires data to be a matrix. However, our data isn't currently in a matrix. In fact, many of our features aren't numeric at all! so we should check first few rows of our dataframe and see what it looks like.

As per the result we got, our data will need some cleaning before it's ready to be put in a matrix. To prepare our data, we have a number of steps we need to complete:

. Remove information about the target variable from the training data

. Reduce the amount of redundant information

. Convert categorical information (like country) to a numeric format

. Split dataset into testing and training subsets

. Convert the cleaned dataframe to a Dmatrix
 
# Training the model
Now that we have our testing & training set cleaned and ready to go, it's time to start training our model. Let's start by training one model and then work on tweaking parameters. In order to train our model, we need to give it some information to start with.

. What training data to use. In this case, we've already put our data in a dmatrix and can just pass it that.

. The number of training rounds. This just means the number of times we're going to improve our naive model by adding additional models.

. What the objective function is. 

# Train an xgboost model

There are two things we can try to see if we improve our model performance.

. Account for the fact that we have imbalanced classes. "Imbalanced classes" just means that we have more examples from one category than the other. In this case, humans don't usually get sick when animals do, but sometimes they do. We can help make sure that we're making sure to predict rare events by scaling the weight we give to positive cases.

. Train for more rounds. If we stop training early, it's possible that our error rate is higher than it could be if we just kept at it for a little longer. It's also possible that training longer will result in a more complex model than we need and will cause us to over-fit. We can help guard against this by setting a second parameter, early_stopping_rounds, that will stop training if we have no improvement in a certain number of training rounds.

# Examining the model
One of the really nice things about xgboost is that is has a lot of built-in functions to help us figure out why our model is making the distictions it's making.

One way that we can examine our model is by looking at a representation of the combination of all the decision trees in our model. Since all the trees have the same depth (remember that we set that with a parameter!) we can stack them all on top of one another and pick the things that show up most often in each node. Tree moedel is created and bar plot also created as quick way to see which features are most important.
