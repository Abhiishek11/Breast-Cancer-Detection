# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
 

#Importing the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'Clump_Thickness', 'Univorm_cell_Size','uniform_cell_Shape',
         'marginal_adhesion','single_epethelial_size','bare_nuclei',
         'blind_Chromatin','normal_Nucleoli','mitosis', 'class']
dataset = pd.read_csv(url, names=names)


#Data Preprocessing
dataset.replace('?', -99999, inplace = True)
print(dataset.axes)
dataset.drop(['id'], 1, inplace = True)

#Print the shape of the dataset
print(dataset.shape)

#Visualizing the dataset
#print(dataset.loc[698])
print(dataset.describe())

#Plotting the Histograms for each variable
dataset.hist(figsize = (10,10))
plt.show()

#Create scatter plot matrix
scatter_matrix(dataset, figsize=(10,10))
plt.show()

#Splitting into test and training set
X = np.array(dataset.drop(['class'], 1))
y = np.array(dataset['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Specifying the testing options
seed = 8
scoring = 'accuracy'

#Fitting KNN to training set
classifier = KNeighborsClassifier(n_neighbors = 5, 
                                  metric = 'minkowski',
                                  p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy = (82+51)/(82+52+3+3)