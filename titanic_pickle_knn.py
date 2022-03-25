import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import random
import pickle


with open("titanic_knn.pickle","rb") as f:
    knn = pickle.load(f)


Rose = [[0,1]] #female, first class
Jack = [[1,3]] #male, third class

v = knn.predict(Rose)

print("PREDICTION")
if v==1:
    print("Rose will survive")
else:
    print("Rose will die")
v = knn.predict(Jack)

if v==1:
    print("Jack will survive")
else:
    print("Jack will die")
    
