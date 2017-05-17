# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
## Takes a categorical field and transforms it into number values (alphabetically), replacing the data with numbers
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

## Encodes the numbers from the categorical transformer into separate binary columns for each transformed value. Appends to start of dataset.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
## Avoid "Dummy Variable Trap" by getting rid of the first column of hotencoded data
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - Must Have for NN's to reduce computational strain
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

######################################################################################################
######################################################################################################

# Part 2 - Now let's make the ANN!

# Importing the Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # dropout regularization to reduce overfitting

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
## Review of processes:
## 1. Initialize weights of the synapses to small numbers close to 0 but not 0 (done with Dense)
## 2. Input our 11 into the input layer, one feature per node (11 nodes)
## 3. Propogate left to right - Choose the activation function (best is the rectifier), then choose your output function (best is sigmoid for probability)
## 4. Check the error of your prediction to the actual
## 5. Back Propagate to adjust the weights based on their responsibility for the error, more extreme based on the learning rate
## 6. Repeat steps 1-5 in batch or stochastic
## 7. one run through is an epoch

#(ctrl+i is help)relu
# help(keras.layers.core.activation)
# 'uniform' distributes weights evenly
classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu', input_shape=(11,)))
## adding dropout. try 0.1 then 0.2, do not go over .45 to avoid overfitting
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
# Adding the output layer
# !! WHEN WORKING WITH A DEPENDENT VARIABLE WITH MULTIPLE OUTPUTS, USE SOFTMAX AND ADJUST OUTPUT UNITS TO MATCH OUTPUT COUNT
classifier.add(Dense(1, kernel_initializer= 'uniform', activation = 'sigmoid'))

# Compiling the ANN - Applying stochastic gradient descent on the entire NN
## optimizer is the algorithm (stochastic), there are diff types of stoch algs, cool one is adam
## loss - stoch is based on a loss function to get optimal weights (e.g. sum of sq errors). We'll use the same as logistic regression because sigmoid is a logistic function
### binary for two outputs, more than two is categorical_cross_entropy
## metrics - criterion to evaluate, we'll use accuracy. When the weights are updated, it uses this to improve performance (cost function)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#############################################################################################################

# Fitting the ANN to the training set (fit method)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#############################################################################################################

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # returns true or false

# Making the Confusion Matrix
## we need to adjust the predict output above to be True/False based on a threshold
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#############################################################################################################

# Part 4 - Evaluating, Impproving and Tuning the ANN

# k-fold cross validation: fixes variance by splitting training set into 10-fold. We train on 9 folds and test on 10 folds
## There four bias-variance forms:
    # High Bias Low Vairance (not the most accurate results, but consistent results)
    # High Bias High Variance (awful, not accurate and not consistent)
    # Low Bias Low Variance (perfect, accurate and consistent)
    # Low Bias High Variance (clustered around high accuracy, but not consistent)
    
# Evaluating the ANN - get a more accurate picture of the accuracy
## Combine Keras and Scikitlearn together with a Keras wrapper
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
#First is a function for KerasClassifier that builds the classifier above with all the structure

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu', input_shape=(11,)))
    classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer= 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

## create a global classifier object
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 50)
## create a variable to measure the best setup for the classifier
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
## Dropout regularization to reduce overfitting if needed
## relevant when you have a high variance
## Overfitting - high accuracy on training but low on test or high variance in K-fold cv

## Where to apply dropout?
## At different stages of your NN training, you randomly drop neurons so it's forced to try other correlations/weights
### can be added to just one or multiple - when you ahve overfitting apply it to all layers.

# Tuning the ANN
## Hyper parameters: epochs, batch, number of neurons etc. 
## Parameter Tuning finds the best values of hyper parameters
## Grid-Search: method for parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV # use sklearn.gridsearch if this fails
from keras.models import Sequential
from keras.layers import Dense
#First is a function for KerasClassifier that builds the classifier above with all the structure

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu', input_shape=(11,)))
    classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer= 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

## create a global classifier object
classifier = KerasClassifier(build_fn=build_classifier)
## Create a dictionary to hold the hyper parameters you want to tune
## common practice to take powers of 2
## rmsprop is recommended for a lot of neural networks, not necessarily ANNs
parameters = {'batch_size':[25,32],
              'epochs':[100, 500],
              'optimizer':['adam','rmsprop']}
## create gridsearch object to implement your parameters and model
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = 5)
grid_search = grid_search.fit(X_train, y_train)
## create variables to track accuracy of various variables
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_













# fun tests

# calculate accuracy of confusion matrix
def modacc(conmat):
    corr = conmat[0][0]+conmat[1][1]
    return corr/conmat.sum()

print(modacc(cm))

# get optimal true/false threshold for confusion matrix
xarr = []
arr_range = np.arange(0.0,1.0,.001)
y_pred = classifier.predict(X_test)
for i in arr_range:
    y_acc = (y_pred > i)
    cm_acc = confusion_matrix(y_test, y_acc)
    xarr.append(modacc(cm_acc))

print(max(xarr),arr_range[xarr.index(max(xarr))])
    



