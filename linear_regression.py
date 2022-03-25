import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression





t = [2,6,14,30,62,126]
q = [1,2,3,4,5,6]

df = pd.DataFrame()
df["t"] = t
df["q"] = q
print(df)

df.plot(kind = "scatter", x="t", y="q")
plt.show()

df_X = df[["t"]]
df_y = df["q"]

lm = LinearRegression()
lm.fit(df_X,df_y)

#y = mx +b

print("m is ", lm.coef_)
print("b is ", lm.intercept_)

#prediction

user_input = input("GIVE ME A VALUE ")
user_input = float(user_input)
p = lm.predict([[user_input]])

print(p)