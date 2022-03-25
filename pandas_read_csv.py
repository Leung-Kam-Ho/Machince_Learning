import pandas as pd 

#df = pd.read_csv("bc.csv")
df = pd.read_csv("bc.csv")
c = []

#rf = pd.read_html("https://gamewith.tw/monsterstrike")

for i in df:
    c.append(i)

#df = pd.DataFrame(df)
#

#df.index = df["Price"]
#df= df.drop("Price", axis = 1)

print(df.describe())
#s.plot()