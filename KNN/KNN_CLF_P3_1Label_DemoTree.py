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

X = {}
Y = {}
count1 = 0
count2 = 0
count3 = 0

for picFilename in glob.glob("processed3\\processed\\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber1 = picFilename.split('\\')[2].split('_')[1]
    toothNumber2 = picFilename.split('\\')[2].split('_')[2].split('.')[0]
    im = Image.open(picFilename)

    if int(metaY.loc['Age',patNumber]) < 25:
        count1+=1
        if 'Age1' in X.keys():
            X['Age1'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age1'].append(1)
            else:
                Y['Age1'].append(0)
        else:
            X['Age1'] = pd.DataFrame()
            X['Age1'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age1'] = [1]
            else:
                Y['Age1'] = [0]
    
    elif int(metaY.loc['Age',patNumber]) < 40:
        count2+=1
        if 'Age2' in X.keys():
            X['Age2'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age2'].append(1)
            else:
                Y['Age2'].append(0)
        else:
            X['Age2'] = pd.DataFrame()
            X['Age2'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age2'] = [1]
            else:
                Y['Age2'] = [0]
    
    else:
        count3+=1
        if 'Age3' in X.keys():
            X['Age3'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age3'].append(1)
            else:
                Y['Age3'].append(0)
        else:
            X['Age3'] = pd.DataFrame()
            X['Age3'][patNumber + '_' + toothNumber1 + '_' + toothNumber2] = np.array(im).flatten()
            if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes':
                Y['Age3'] = [1]
            else:
                Y['Age3'] = [0]

print(count1)
print(count2)
print(count3)

print(X)
print(Y)

for age in X.keys():
    print(age)
    ageX = X[age]
    ageY = Y[age]

    X_train, X_test, y_train, y_test = train_test_split(ageX.T.to_numpy(), np.array(ageY), test_size=0.20, random_state=100)

    #print("KNN")
    parameters = {'weights':['uniform','distance'],'n_neighbors':np.arange(3,25,1)}
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh = GridSearchCV(neigh, parameters, cv=5, scoring='neg_mean_squared_error').fit(ageX.T.to_numpy(), ageY)
    print(neigh.best_params_)  
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