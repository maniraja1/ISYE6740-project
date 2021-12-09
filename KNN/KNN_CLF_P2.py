from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statistics as stat
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

labels = pd.read_csv("Xray_TeethLabels_Simple.csv",index_col=0)
Ylabels = labels[5:]
print(Ylabels)

metaY = labels[:3]
print(metaY)

X = pd.DataFrame()
Y = []

for picFilename in glob.glob("processed2\\processed\\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber = picFilename.split('\\')[2].split('_')[1].split('.')[0]
    im = Image.open(picFilename)
    X[patNumber + '_' + toothNumber] = np.array(im).flatten()
    Y.append(Ylabels.loc[toothNumber,patNumber])

Y = [1 if x=='Yes' else 0 for x in Y]
print(X)
print(Y)

# Split up data based on the tooth number, and branch based on demographics

X_train, X_test, y_train, y_test = train_test_split(X.T.to_numpy(), np.array(Y), test_size=0.20)

#print("KNN")
parameters = {'weights':['uniform','distance']}
neigh = KNeighborsClassifier(n_neighbors=2)
neigh = GridSearchCV(neigh, parameters, cv=5, scoring='neg_mean_squared_error').fit(X.T.to_numpy(), Y)
print(neigh.cv_results_['mean_test_score'])
neigh = neigh.best_estimator_
neighPred = neigh.predict(X_test)
print(accuracy_score(y_test, neighPred))

# print("KNN Accuracy: {}%".format(stat.mean(neighAcc)*100))