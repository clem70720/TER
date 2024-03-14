
import numpy as np
import pandas as pad
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import dump

#Target for this classification task set at 85% accuracy score

# get the start time
st = time.process_time()

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
pipe = Pipeline([('TSVD', TruncatedSVD(n_components=185)), ('logreg', LogisticRegression(solver="sag", random_state = 42))])

pipe.fit(X_train,y_train) # fit on train data 
score_accuracy = pipe.score(X_test,y_test) # Score accuracy on test data
test_accuracy=round(100*score_accuracy,2)
print(f'The test accuracy score is {test_accuracy}%')

dump(pipe,"LogReg.sav")

# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')

# Test Realm
"""
J'ai testé différent paramètre du modèle de regression logistique notamment le solver, voici les résultats:
Avec le solver 'sag' le modèle obtient 84.97% de précision sur le jeu de test
Avec le solver 'saga', le modèle obtient 85.12% 
Avec "liblinear", le modèle obtient 82.95%
Avec 'Newton-cg', le modèle obtient 82.39%
Avec 'Newton-Choleski', le modèle obtient 83.1%
Avec 'lblg', le modèle obtient 83.8%

J'ai donc retenu le solver 'saga' car il y a obtenu les meilleurs performances

J'ai également testé les différents type de pénalité avec le solver saga 
Avec Saga + penalité L1, le modèle obtient 85.11% de précision
Avec Saga + penalité L2, le modèle obtient 85.12%
Avec Saga + penalité elasticnet + L1 ratio=0.5, le modèle obtient 85.09%

Comme L2 est l'hyperparamètre par défaut et obtient le meilleur score de précision je décide de le garder 

J'ai ensuite utilisé la PCA qui permet de réduire le nombre de variable (au prix d'un peu de précision mais améliorant la vitesse d'exécution) en ne retenant que les variables 
expliquant la variance, j'ai testé pour 80% de variance expliqué mais cela était en dessous de la cible donc je ne l'ai pas retenu, 90% permettait d'atteindre la cible avec le
plus faible temps de calcul je l'ai donc retenu enfin j'ai essayé avec 95% de variance expliqué mais cela ne fournit qu'un gain marginal de précision pour un gain de calcul moins
important que 90%, j'ai ensuite ajouté la PCA pour le modèle KNN ce qui a nettement augmenté ses performances en temps de calcul 

Tests 14/03/2024:

PCA 0.95 + LogReg sag, random_state=42 = 85.64% en 35.67 secondes
TSVD 150 + LogReg sag, random_state=42 = 85.72% en 27.95 secondes
TSVD 200 + LogReg sag, random_state=42 = 85.79% en 36.81 secondes
TSVD 175 + LogReg sag, random_state=42 = 85.83% en 31.96 secondes
TSVD 187 + LogReg sag, random_state=42 = 85.72% en 34.06 secondes
TSVD 162 + LogReg sag, random_state=42 = 85.8% en 28.95 secondes
TSVD 185 + LogReg sag, random_state=42 = 85.86% en 33.34 secondes #retenu
"""