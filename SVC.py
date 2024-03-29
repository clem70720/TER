import numpy as np
import pandas as pad
import time

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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
pipe = Pipeline([('pca', PCA(n_components=0.92)),('scaler', StandardScaler()), ('SVC_Classifier',SVC(C=4,kernel='rbf',probability=True))])

# Cross validate 
nb_cv=5
mean_cross_val_score = cross_val_score(pipe,X_train,y_train,cv=nb_cv).mean()
train_accuracy = round(100*mean_cross_val_score,2)
print(f'The mean train accuracy score is {train_accuracy}% with {nb_cv} cross-validation')

# get the start time
st = time.process_time()

pipe.fit(X_train,y_train)
score_accuracy = pipe.score(X_test,y_test)
test_accuracy=round(100*score_accuracy,2)
print(f'The test accuracy score is {test_accuracy}%')

dump(pipe,"SVC.sav")

# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')
