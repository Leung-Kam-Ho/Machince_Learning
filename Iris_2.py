import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


#----------------------Preperation-------------------------------
df = pd.read_csv("Iris.csv")

#delete the Id columns
df = df.drop("Id", axis= 1)

#delete duplicates and reet the index
df = df.drop_duplicates()
df = df.reset_index(drop=True)

#print(df)

#the visualize the data, change the string to numerial data
s = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
df["Species"] = df["Species"].map(s)

#----------------------Create model----------------------------
df_X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
print(df_X)

df_y = df["Species"]
#print(df_y)
X_train, X_test, y_train, y_test = train_test_split(df_X,df_y,test_size= 0.2)
score_list = []
k_list = []
for i in range(3,21):
    k=i
    k_list.append(k)

    #for i in range(3,11):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    #print("k= ", k," accuracy",score)
    print("k= ", k)
    

    pred = knn.predict(X_test)

    c_score = cross_val_score(knn,df_X,df_y, scoring="accuracy", cv = 48).mean()

    print("CROSS_VAL_SCORE ", c_score)

    score_list.append(c_score)

    #print("Prediction ",pred)
    #print("answer     ",y_test.values)
    # acc = accuracy_score(y_test, pred)

    #print("test accuracy ",acc)

    #print(confusion_matrix(y_test,pred))
csdf = pd.DataFrame()
csdf["k value"] = k_list
csdf["score"] = score_list

print(csdf[ csdf["score"]==csdf["score"].max()])


#---------------------------------------------------------------

#Visualize the data
#df_knn = pd.DataFrame()
#df_knn["s"] = score_list
#df_knn.index = [3,4,5,6,7,8,9,10]

#df_knn.plot(grid = True)
#plt.show()

