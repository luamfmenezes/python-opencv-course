import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense

# Classification metrics

# 1. Accuracy
# 2. Recall - true positive / ( true positives + false negatives )
# 3. Precision - true positives / ( true positives + false positives )
# 4. F1-Score - 2 * precision * recall / precision + recall


data = genfromtxt('assets/datasets/bank_note_data.txt', delimiter=',')

labels = data[:,4]
features = data[:,:4]

x = features  # inputs
y = labels  # outputs

x_train,y_test,x_test,y_test =train_test_split(x,y,test_size=0.33,random_state=42)

# normalized from 0 - 1

scaler_object = MinMaxScaler()
scaler_object.fit(x_train)
scaled_x_train = scaler_object.transform(x_train)
scaled_x_test = scaler_object.transform(x_test)


model = Sequential()

# Create layers
model.add(Dense(4,input_dim=4,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# Training
model.fit(scaled_x_train, y_train, epochs=50)

predictions = model.predict_classes(scaled_x_test)

# confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))


# model.save('my-model.h5')
# newModel = load_model('my-model.h5')

