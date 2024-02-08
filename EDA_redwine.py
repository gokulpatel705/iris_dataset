import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/pw data  science/dataset/winequality-red.csv")
print(df.info()) 
print(df.describe())
print(df.isna().sum())
print(df.shape)
print(df.columns)
print(df['quality'].value_counts())# this is the imbalanced data set
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
print(df.duplicated().sum())
print(df.shape)
heat=df.corr()
#using matplotlib
print(plt.imshow(heat))
plt.colorbar()
plt.show()
#using seaborn
sns.heatmap(heat,annot=True)
plt.show()

# now we will plot the bar graph bcoz quality is a categorical value
print(df['quality'].value_counts())
print(df['quality'].value_counts().plot(kind='bar'))
plt.show()

# print(sns.distplot(df['fixed acidity']))

# plt.show()
# print(sns.histtplot(df['fixed acidity']))
for i in df.columns:
    print(sns.histplot(df[i]))
plt.show()