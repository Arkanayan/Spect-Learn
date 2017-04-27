import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

## Parameters
image_folder = 'spect'

## Functions
def batch(x, y, size=1):
    l = len(x)
    for ndx in range(0, l, size):
        yield x[ndx : min(ndx + size, l)], y[ndx : min(ndx + size, l)]


## Read samples
dataset = np.array([])
labels = np.array([])

for index, file in enumerate(os.listdir(image_folder)):
    if file.endswith('.png'):
        # print(os.path.join(image_folder, file))
        file_path = os.path.join(image_folder, file)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        label = int(str(file)[0])
        # sample = np.array((image, 1))
        sample = np.array(image)
        # import ipdb; ipdb.set_trace()
        
        if dataset.size == 0:
            dataset = np.array((sample[None, :]))
            labels = np.array(label)
        else:
            dataset = np.row_stack((dataset, sample[None, :]))
            labels = np.append(labels, label)
        print(dataset.shape)
        if index == 50:
            break


x_train, x_test , y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=5)
# import ipdb; ipdb.set_trace()

batch_size = 5
num_classes = 10
epochs = 12

img_rows, img_cols, channels = image.shape[0], image.shape[1], image.shape[2]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Saving the model as JSON...')

# serialize model to JSON
model_json = model.to_json()

with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")