from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statistics as stat
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random

random.seed(100)
labels = pd.read_csv("Xray_TeethLabels_Simple.csv",index_col=0)
Ylabels = labels[5:]
print(Ylabels)

metaY = labels[:3]
print(metaY)

X = pd.DataFrame()
Y = []

for picFilename in glob.glob("processed3\\processed\\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber1 = picFilename.split('\\')[2].split('_')[1]
    toothNumber2 = picFilename.split('\\')[2].split('_')[2].split('.')[0]
    im = Image.open(picFilename)
    X[patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
    if Ylabels.loc[toothNumber1,patNumber] == 'Yes' and Ylabels.loc[toothNumber2,patNumber] == 'Yes':
        Y.append([1,1])
    elif Ylabels.loc[toothNumber1,patNumber] == 'Yes' and Ylabels.loc[toothNumber2,patNumber] == 'No':
        Y.append([1,0])
    elif Ylabels.loc[toothNumber1,patNumber] == 'No' and Ylabels.loc[toothNumber2,patNumber] == 'Yes':
        Y.append([0,1])
    else:
        Y.append([0,0])

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X.T.to_numpy(), np.array(Y), test_size=0.20, random_state=100)

#print("KNN")
parameters = {'weights':['uniform','distance'],'n_neighbors':np.arange(5,50,2)}
neigh = KNeighborsClassifier(n_neighbors=2)
neigh = GridSearchCV(neigh, parameters, cv=5, scoring='neg_mean_squared_error').fit(X.T.to_numpy(), Y)
print(neigh.cv_results_['mean_test_score'])
neigh = neigh.best_estimator_
neighPred = neigh.predict(X_test)
print(accuracy_score(y_test, neighPred))
print(classification_report(y_test, neighPred))
print(confusion_matrix(y_test, neighPred))
tn, fp, fn, tp = confusion_matrix(y_test, neighPred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
FPR = fp / (fp+tn)
FNR = fn / (fn+tp)
print(sensitivity)
print(specificity)
print(FPR)
print(FNR)

# print("KNN Accuracy: {}%".format(stat.mean(neighAcc)*100))