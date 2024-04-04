import numpy as np
import pandas as pad
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from joblib import dump

#Target for this classification task set at 85% accuracy score

# reading the data csv and converting it into a dataframe
df=pad.read_csv('./fashion-mnist_train.csv')
# dropping the above 43 duplicated images
df.drop_duplicates(inplace=True)
#reading the data csv and converting it into a dataframe
df_test=pad.read_csv('./fashion-mnist_test.csv')

# Creating X and y variables
X_train=df.drop('label',axis=1).astype("float32")
X_train = X_train / np.max (X_train)
y_train=df.label

# Creating X and y variables
X_test=df_test.drop('label',axis=1).astype("float32") 
X_test = X_test / np.max (X_test)
y_test=df_test.label

# Making a pipeline to get faster CPU exec time 
pipe = Pipeline([('pca', PCA(n_components=0.92)),('scaler', StandardScaler()), ('RF', RandomForestClassifier(criterion='entropy',max_depth=50,n_estimators=100))])

# Cross validate 
nb_cv=5
mean_cross_val_score = cross_val_score(pipe,X_train,y_train,cv=nb_cv).mean()
train_accuracy = round(100*mean_cross_val_score,2)
print(f'The mean train accuracy score is {train_accuracy}% with {nb_cv} cross-validation')

# get the start time
st = time.process_time()

pipe.fit(X_train,y_train) # fit on train data 
score_accuracy = pipe.score(X_test,y_test) # Score accuracy on test data
test_accuracy=round(100*score_accuracy,2)
print(f'The test accuracy score is {test_accuracy}%')

dump(pipe,"KNN.sav")

# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')

"""
Test 14/03/2024
PCA 0.875 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.51% en 45.14 secondes
TSVD 185 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 84.74% en 124.3125 secondes
TSVD 25 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.04% en 23.28 secondes
TSVD 15 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 84.01% en 19.32 secondes
TSVD 35 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.25% en 25.07 secondes
TSVD 45 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.57% en 29.85 secondes 
TSVD 55 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.64% en 34.79 secondes #retenu
TSVD 65 + Stdscaler + KNN n_neightbors = 11 p=1 weights = distance obtient 86.41% en 39.98 secondes 
"""