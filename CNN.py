import numpy as np
import pandas as pad
import time
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#Model Parameter
Img_shape = 28
Num_classes = 10
test_size = 0.25
random_state = 1234
No_epochs = 10
Batch_size = 100

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# get the start time
st = time.process_time()

# reading the data csv and converting it into a dataframe
df=pad.read_csv('./fashion-mnist_train.csv')
# dropping the above 43 duplicated images
df.drop_duplicates(inplace=True)
#reading the data csv and converting it into a dataframe
df_test=pad.read_csv('./fashion-mnist_test.csv')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

def data_preprocessing(raw):
    label = to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    image = x_shaped_array / np.max(df_test)
    return image, label

X, y = data_preprocessing(df)
X_test, y_test = data_preprocessing(df_test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu',kernel_initializer='he_normal', input_shape=(28,28, 1)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=Batch_size,epochs=No_epochs, verbose=1,validation_data=(X_val, y_val))

score = model.evaluate(X_test,y_test,verbose=0, steps=math.ceil(10000/32))

print("Pourcentage de bien classées:", score[1])

model.save("CNN.keras")

# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')