import numpy as np
import pandas as pad
import time
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Data augmentation function
from keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_eraser

def random_reverse(x): 
    if np.random.random() > 0.5: 
        return x[:,::-1]
    else: 
        return x

def data_generator(X, Y, batch_size=100): 
    while True: 
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p, q = [], []
        for i in range(len(X)): 
            p.append(random_reverse(X[i]))
            q.append(Y[i])
            if len(p) == batch_size: 
                yield np.array(p), np.array(q)
                p, q = [], []
        if p: 
            yield np.array(p), np.array(q)
            p, q = [], []

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=False))

#Model Parameter
Img_shape = 28
Num_classes = 10
test_size = 0.25
random_state = 1234
No_epochs = 20
Batch_size = 32

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

model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=20, 
                                      validation_data=(X_val, y_val))

score = model.evaluate(X_test,y_test,verbose=0, steps=math.ceil(10000/32))

print("Pourcentage de bien classées:", score[1])

model.save("CNN_DATA_AUG.keras")

# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')

# Zhong, Z. et al. (2017) Random erasing data augmentation. https://arxiv.org/abs/1708.04896.
# Achieving 95.42% Accuracy on Fashion-Mnist Dataset Using Transfer Learning and Data Augmentation with Keras – Zheng Zhang (no date). https://secantzhang.github.io/blog/deep-learning-fashion-mnist.