import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#df = pd.read_csv("bc.csv")
df = pd.read_csv("train.csv")

print("------------------")
print("first 20")
print("------------------")
print(df.head(20))

print("------------------")
print("info")
print("------------------")
print(df.info())

print("------------------")
print("describe")
print("------------------")
print(df.describe())

print("------------------")
print("is null or not")
print("------------------")
print(df.isnull())


print("------------------")
print("percentage of null")
print("------------------")
print((df.isnull().sum()/df.isnull().count())*100)

df["Age"] = df["Age"].fillna(df["Age"].mean)
df["Embarked"] = df["Embarked"].fillna("S")
df = df.drop("Cabin", axis = 1)

print("------------------")
print("percentage of null after filling")
print("------------------")
print((df.isnull().sum()/df.isnull().count())*100)



print("------------------")
print("check if it is filled")
print("------------------")

print(df.iloc[61,:])
 

print("------------------")
print("check if there are any duplicated data")
print("------------------")

#print(df[df.duplicated()])

print("------------------")
print("data exchange")
print("------------------")
sex = {"male": 1, "female": 0}
embarked = {"S":0, "C":1, "Q":2}
df["Sex"] = df["Sex"].map(sex)
df["Embarked"] = df["Embarked"].map(embarked)
print(df.head())


print("------------------")
print("How many Survived")
print("------------------")

find_relation= ["Survived","Sex","Pclass"]
for i in df.columns:
    if i in find_relation:
        s = df[i].value_counts()
        #s.plot(kind = "pie", autopct= "%1.2f%%")
        #plt.show()

print("------------------")
print("xxx and Survive")
print("------------------")

find_relation= ["Survived","Sex","Pclass"]
for i in df.columns:
    if i in find_relation:
        s =df.groupby([i,"Survived"])["PassengerId"].count()/df.groupby([i])["PassengerId"].count()*100
        print(s)
        s.plot(kind="bar", rot =0)
        plt.show()

#s.plot()