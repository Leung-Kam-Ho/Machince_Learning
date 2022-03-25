import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import random
import pickle

def build_and_train_model(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    #print("k= ", k," accuracy",score)
    #print("k= ", k)
    

    pred = knn.predict(X_test)

    c_score = cross_val_score(knn,df_X,df_y, scoring="accuracy", cv = 48).mean()

    #print("CROSS_VAL_SCORE ", c_score)
    score_list.append(c_score)
    return knn

#df = pd.read_csv("bc.csv")
df = pd.read_csv("train.csv")

print("------------------")
print("first 5")
print("------------------")
print(df.head())

print("------------------")
print("fill null")
print("------------------")
#print((df.isnull().sum()/df.isnull().count())*100)

df["Age"] = df["Age"].fillna(df["Age"].mean)
df["Embarked"] = df["Embarked"].fillna("S")
df = df.drop("Cabin", axis = 1)

print("------------------")
print("data exchange")
print("------------------")
sex = {"male": 1, "female": 0}
embarked = {"S":0, "C":1, "Q":2}
df["Sex"] = df["Sex"].map(sex)
df["Embarked"] = df["Embarked"].map(embarked)
print(df.head())

df_X = df[["Sex","Pclass"]]
df_y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(df_X,df_y,test_size= 0.2)
score_list = []
k_list = []
for i in range(3,21):
    k=i
    k_list.append(k)
    build_and_train_model(k)
    
    
    
    
csdf = pd.DataFrame()
csdf["K"] = k_list
csdf["score"] = score_list
max = csdf["score"].max()
print("max ", max)

print(csdf)
max_index=csdf.index[csdf["score"] == max].tolist()
print(max_index)
ran = random.choice(max_index)
print("the index of final k is ", ran)
#print(csdf)

final_k = csdf.iloc[ran,0]




print("-----------------------------------------")

Rose = [[0,1]] #female, first class
Jack = [[1,3]] #male, third class

knn = build_and_train_model(final_k)

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
    
with open("titanic_knn.pickle","wb") as file:

    pickle.dump(knn, file)

#s.plot()



