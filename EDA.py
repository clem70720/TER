#importing libraries to perform EDA
import numpy as np
import pandas as pad
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression


# reading the data csv and converting it into a dataframe
df=pad.read_csv('./fashion-mnist_train.csv')

# quick peek into the dataframe
print(df.head())

# checking for null-values
print(f"Il y a {df.isnull().sum().sum()} valeurs nulles")

# checking the number of duplicated images
print(f"Il y a {df.duplicated().sum()} données dupliquées")

# dropping the above 43 duplicated images
df.drop_duplicates(inplace=True)
print(df.shape)

print(df.label.unique())

# Eliminating duplicates
df.drop_duplicates(inplace=True)
# Creating X and y variables
X_train=df.drop('label',axis=1)
y_train=df.label

#testing linearity by evaluating R-square
print(LinearRegression().fit(X_train,y_train).score(X_train,y_train))

# lets now analyze the labels and their corresponding numbers
colors = sns.color_palette('mako_r')[1:3]
plt.pie(x=df.groupby(['label']).count()['pixel1'],labels=df.groupby(['label']).count().index, autopct='%1.1f%%')
plt.show()