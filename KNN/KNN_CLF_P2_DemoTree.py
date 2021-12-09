from os import fdopen
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
#sklearn.metric report

labels = pd.read_csv("Xray_TeethLabels_Simple.csv",index_col=0)
Ylabels = labels[5:]
print(Ylabels)

metaY = labels[:3]
print(metaY)

X = {}
Y = {}
count1 = 0
count2 = 0
count3 = 0
for picFilename in glob.glob("processed2\\processed\\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber = picFilename.split('\\')[2].split('_')[1].split('.')[0]
    im = Image.open(picFilename)

    if int(metaY.loc['Age',patNumber]) < 25:
        count1+=1
        if 'Age1' in X.keys():
            X['Age1'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age1'].append(1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0)
        else:
            X['Age1'] = pd.DataFrame()
            X['Age1'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age1'] = [1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0]
    
    elif int(metaY.loc['Age',patNumber]) < 40:
        count2+=1
        if 'Age2' in X.keys():
            X['Age2'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age2'].append(1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0)
        else:
            X['Age2'] = pd.DataFrame()
            X['Age2'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age2'] = [1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0]
    
    else:
        count3+=1
        if 'Age3' in X.keys():
            X['Age3'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age3'].append(1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0)
        else:
            X['Age3'] = pd.DataFrame()
            X['Age3'][patNumber + '_' + toothNumber] = np.array(im).flatten()
            Y['Age3'] = [1 if Ylabels.loc[toothNumber,patNumber] == 'Yes' else 0]

    

print(count1)
print(count2)
print(count3)

print(X)
print(Y)

for age in X.keys():
    print(age)
    ageX = X[age]
    ageY = Y[age]

    X_train, X_test, y_train, y_test = train_test_split(ageX.T.to_numpy(), np.array(ageY), test_size=0.20)

    #print("KNN")
    parameters = {'weights':['uniform','distance']}
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh = GridSearchCV(neigh, parameters, cv=5, scoring='neg_mean_squared_error').fit(ageX.T.to_numpy(), ageY)
    print(neigh.cv_results_['mean_test_score'])
    neigh = neigh.best_estimator_
    neighPred = neigh.predict(X_test)
    print(accuracy_score(y_test, neighPred))

    # print("KNN Accuracy: {}%".format(stat.mean(neighAcc)*100))