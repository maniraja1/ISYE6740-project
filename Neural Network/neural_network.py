import tensorflow as tf # tensorflow 2.0
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import imageio
import pandas as pd
import keras
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

labels = pd.read_csv("Xray_TeethLabels_Simple.csv",index_col=0)
Ylabels = labels[5:]
metaY = labels[:3]

X = []
Y = []

for picFilename in glob.glob("Processing2\processed\*"):
    patNumber = picFilename.split('\\')[2].split('_')[0]
    toothNumber = picFilename.split('\\')[2].split('_')[1].split('.')[0]
    print(picFilename)
    print(Ylabels.loc[toothNumber,patNumber])
    im = Image.open(picFilename)
    X.append(np.array(im))
    if (Ylabels.loc[toothNumber,patNumber] == "Yes"):
        Y.append(1)
    else:
        Y.append(0)

X = np.array(X)

X_train, X_test, Y_train_original, Y_test_original = train_test_split(X, np.array(Y), random_state = 100, test_size=0.20)

Y_train = tf.keras.utils.to_categorical(Y_train_original, 2)
Y_test = tf.keras.utils.to_categorical(Y_test_original, 2)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(32, 5, activation='relu', input_shape=(500, 500, 3)))

model.add(MaxPooling2D())

model.add(Conv2D(64, 5, activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## For Hyper parameter tuning ##

# accuracy_scores = []
#
# for i in range(1, 21):
#     fitted_model = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=i)
#
#     info = model.evaluate(X_test, Y_test, verbose=1)
#     accuracy_scores.append(info[1])
#
#     print('Accuracy score for epoches ' + str(i) + ':', info[1])
#
# accuracy_scores = [score * 100 for score in accuracy_scores]
#
# plt.plot(range(1, 21), accuracy_scores, label="Accuracy vs Number of Epoches")
# plt.xlabel("Number of Epoches")
#
# plt.ylabel("Accuracy Percentage")
# plt.show()
# print(accuracy_scores.index(max(accuracy_scores)) + 1)

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=9)
info = model.evaluate(X_test, Y_test)
print('Accuracy score for epoches ' + str(9) + ':', info[1])

predictions = (model.predict(X_test) > 0.5).astype(int)
print(predictions)
pred = []

for prediction in predictions:
    i, = np.where(prediction == 1)
    pred.append(i[0])

print(pred)
print(Y_test_original)
print(classification_report(Y_test_original, pred))
tn, fp, fn, tp = confusion_matrix(Y_test_original, pred).ravel()
print(tn, fp, fn, tp)
