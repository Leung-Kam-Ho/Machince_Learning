import pandas as pd  
import matplotlib.pyplot as pt

df = pd.read_html("https://img.nike.com.hk/resources/sizecart/mens-shoe-sizing-chart.html")
c = []

#rf = pd.read_html("https://gamewith.tw/monsterstrike")

for i in df:
    c.append(i)

#df = pd.DataFrame(df)
df = df[0]
print(df)

df.plot(kind= "scatter", x="UK", y= "CM")
#df.plot()
pt.show()