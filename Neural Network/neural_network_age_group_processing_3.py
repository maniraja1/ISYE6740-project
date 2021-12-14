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
import tensorflow as tf # tensorflow 2.0
from sklearn.metrics import classification_report, confusion_matrix

labels = pd.read_csv("Xray_TeethLabels_Simple.csv",index_col=0)
Ylabels = labels[5:]

metaY = labels[:3]
print(metaY)
X = {}
Y = {}
count1 = 0
count2 = 0
count3 = 0
for picFilename in glob.glob("Processing3\processed\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber1 = picFilename.split('\\')[2].split('_')[1]
    toothNumber2 = picFilename.split('\\')[2].split('_')[2].split('.')[0]
    im = Image.open(picFilename)

    # if int(metaY.loc['Age',patNumber]) < 25:
    #     count1+=1
    #     if 'Age1' in X.keys():
    #         X['Age1'].append(np.array(im))
    #         Y['Age1'].append(1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0)
    #     else:
    #         X['Age1'] = []
    #         X['Age1'].append(np.array(im))
    #         Y['Age1'] = [1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0]

    # if int(metaY.loc['Age',patNumber]) < 40 and int(metaY.loc['Age',patNumber]) > 25:
    #     count2+=1
    #     if 'Age2' in X.keys():
    #         X['Age2'].append(np.array(im))
    #         Y['Age2'].append(1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0)
    #     else:
    #         X['Age2'] = []
    #         X['Age2'].append(np.array(im))
    #         Y['Age2'] = [1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0]

    if int(metaY.loc['Age',patNumber]) > 40:
        if 'Age3' in X.keys():
            X['Age3'].append(np.array(im))
            Y['Age3'].append(1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0)
        else:
            X['Age3'] = []
            X['Age3'].append(np.array(im))
            Y['Age3'] = [1 if Ylabels.loc[toothNumber1,patNumber] == 'Yes' or Ylabels.loc[toothNumber2,patNumber] == 'Yes' else 0]

for age in X.keys():
    ageX = np.array(X[age])
    ageY = Y[age]

    X_train, X_test, Y_train_original, Y_test_original = train_test_split(ageX, np.array(ageY), random_state = 100, test_size=0.20)

    Y_train = tf.keras.utils.to_categorical(Y_train_original, 2)
    Y_test = tf.keras.utils.to_categorical(Y_test_original, 2)

    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten
    from keras.layers import MaxPooling2D, Dropout

    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=(500,500,3)))

    model.add(MaxPooling2D())

    model.add(Conv2D(64, 5, activation='relu'))

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=9)
    info = model.evaluate(X_test, Y_test)
    print('Accuracy score for epoches ' + str(9) + ':', info[1])

    predictions = (model.predict(X_test) > 0.5).astype(int)
    pred = []

    for prediction in predictions:
        i, = np.where(prediction == 1)
        pred.append(i[0])

    print(classification_report(Y_test_original, pred))
    tn, fp, fn, tp = confusion_matrix(Y_test_original, pred).ravel()
    print(tn, fp, fn, tp)
