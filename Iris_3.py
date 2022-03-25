import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from pandas.io.sql import DatabaseError
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


#----------------------Preperation-------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis= 1)
df = df.drop_duplicates()
df = df.reset_index(drop=True)
s = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
df["Species"] = df["Species"].map(s)

df_X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
s = []
for i in range(1,15):
    k=i
    km = KMeans(n_clusters=k)
    km.fit(df_X)
    a = km.inertia_
    print("Accuracy : ", a)
    s.append(a)
print(a)

df_kmeans = pd.DataFrame()
df_kmeans["Inertia"] = s
df_kmeans.index = list(range(1,15))

k=3
km = KMeans(n_clusters=k)
km.fit(df_X)
a = km.inertia_
print("Inertia : ", a)
#df_kmeans.plot(grid=True)
#plt.show()

print("result --------------")
pred = km.fit_predict(df_X)


df1 = df_X.copy()
df1["pred"] = pred
c = {0:"r",1:"g",2:"b"}
df1["cc"] = df1["pred"].map(c)

df_pre = DataFrame()

v = km.predict([[6.6,3.,5.1,5.4]])


print(v)
df1.plot(kind = "scatter", x="SepalLengthCm", y ="SepalWidthCm", c = df1["cc"])






plt.show()


