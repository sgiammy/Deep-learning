# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# To update everything
# conda update --all

#-------------------  Part 1 - Data Preprocessing ------------------------#

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#3 Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Convert Spain, France and Germany into 0,1,2
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 
# Convert Female, Male into 0,1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Instead of Geography colymn with 0,1 or 2, we use 3 columns, one per country (with 0,1)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() 
# we remove the first column to not fall into the dummy variable trap    
X = X[:, 1:] 

#4 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# we need to scale because we don't want one feature to predomine the others
# Standardize features by removing the mean and scaling to unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------- Part 2 - Now let's make the ANN! -------------------#

#1 Importing the Keras libraries and packages
import keras
# Sequential module is required to initialize our ANN
from keras.models import Sequential
# Dense module is required to create the layers
from keras.layers import Dense

#2 Initialising the ANN
# There are two ways to initialize an ANN: 
# 1) defining the sequence of layers
# 2) defining a graph
# we are gonna use the 1) for now
classifier = Sequential()

#3 Adding the input layer and the first hidden layer
# units = output_dim
# the first layer is the input layer, so its dim is 11, 
# so the first hidden layer is gonna have input dim = 11 
# but the question is which is gonna be its output dim ?
# one possible rule is output_dim = avg(input_layer_dim, output_layer_dim)
# i.e. in our case avg(11,1) = 6
# in general we should use the "parameter tuning" technique (e.g. using k-fold cross validation)
# the weights are gonna be initialized with a uniform distribution (between 0 and 1, 0 not included)
# activation function is gonna be the rectifier function (experiments show that this is one of the best one)
# while for the output layer we are gonna use the sigmoid (better because it shows the probability of the output)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#4 Adding the second hidden layer
# the 2nd hidden layer is not really useful for our dataset...but we add it anyway
# input_dim must be specified only for the first hidden layer! 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#5 Adding the output layer
# What we want to predict is the dummy variable 'exited' so we need just one unit
# in the output layer we use the sigmoid activation function, so we'll obtain 
# the probability that the customer leaves the bank
# if we have more than 2 classes, we must use the soft max function, i.e., a kind of sigmoid function
# applied to a dependent variable with more than 2 classes
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#6 Compiling the ANN
# optimizer is the algorithm we want to use to find the best set of weights
# adam is an efficient version of SGD
# loss is the loss function within the SGD algorithm 
# binary_crossentropy is the logaritmic loss function (if more than 2 outcome categorical_crossentropy)
# metrics is how you want to evaluate your model 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#7 Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#------------------- Part 3 - Making predictions and evaluating the model -------------------# 

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)