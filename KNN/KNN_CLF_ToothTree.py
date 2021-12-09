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

X = {}
Y = {}

for picFilename in glob.glob("processed2\\processed\\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber = picFilename.split('\\')[2].split('_')[1].split('.')[0]
    im = Image.open(picFilename)

    if toothNumber in X.keys():
        X[toothNumber][patNumber] = np.array(im).flatten()
    else:
        X[toothNumber] = pd.DataFrame()
        X[toothNumber][patNumber] = np.array(im).flatten()

    if toothNumber in Y.keys():
        l = Y[toothNumber]
        l.append(1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0)
        Y[toothNumber] = l
    else:
        Y[toothNumber] = [1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0]

print(X)
print(Y)

for tooth in X.keys():
    toothX = X[tooth]
    toothY = Y[tooth]

    X_train, X_test, y_train, y_test = train_test_split(toothX.T.to_numpy(), np.array(toothY), test_size=0.20)

    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh = neigh.fit(X_train, y_train)
    neighPred = neigh.predict(X_test)
    print(accuracy_score(y_test, neighPred))

    # print("KNN Accuracy: {}%".format(stat.mean(neighAcc)*100))