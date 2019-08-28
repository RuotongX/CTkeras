from glob import glob
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense,GlobalAveragePooling2D
from keras import backend as Kb
from numpy.random import seed
seed(1526)
from tensorflow import set_random_seed
set_random_seed(1526)
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics


img_w,img_h = 128,128
file = sorted(glob('head_ct/*.png'))
labels = pd.read_csv('head_ct/labels.csv')[' hemorrhage'].tolist()
images = np.empty((len(file),img_w,img_h))
for i, _file in enumerate(file):
    images[i,:,:] = cv2.resize(cv2.imread(_file,0),(img_w,img_h))

# plt.figure(figsize=(10, 10))
# for i in range(0, 9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(images[i], cmap=plt.get_cmap('gray'))
#     plt.title("\nLabel:{}".format(labels[i]))
# # show the plot
# plt.show()

train_images,test_images,train_labels,test_labels = train_test_split(images,labels,test_size = 0.4, random_state = 1)
val_images,test_images,val_labels,test_labels = train_test_split(test_images,test_labels,test_size=0.5,random_state=1)
print((len(train_images),len(val_images),len(test_images)))

input_shape = (img_w,img_h,1)
SIIM_custom_model = None
SIIM_custom_model = Sequential()
SIIM_custom_model.add(Conv2D(32,(3,3),input_shape = input_shape))
SIIM_custom_model.add(Activation('relu'))
SIIM_custom_model.add(MaxPooling2D(pool_size = (2,2)))

SIIM_custom_model.add(Conv2D(32,(3,3)))
SIIM_custom_model.add(Activation('relu'))
SIIM_custom_model.add(MaxPooling2D(pool_size = (2,2)))

SIIM_custom_model.add(Conv2D(32,(3,3)))
SIIM_custom_model.add(Activation('relu'))
SIIM_custom_model.add(MaxPooling2D(pool_size = (2,2)))

SIIM_custom_model.add(Flatten())
SIIM_custom_model.add(Dense(64))
SIIM_custom_model.add(Activation('relu'))

SIIM_custom_model.add(Dropout(0.5))
SIIM_custom_model.add(Dense(1))
SIIM_custom_model.add(Activation('sigmoid'))

SIIM_custom_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

number_train_samples = len(train_images)
number_validation_samples = len(val_images)
number_test_samples = len(test_images)
epochs = 100
batch_size = 10

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.1,
    zoom_range = 0.1,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(
    train_images[..., np.newaxis],
    train_labels,
    batch_size=batch_size)

validation_generator = val_datagen.flow(
    val_images[..., np.newaxis],
    val_labels,
    batch_size=batch_size)



history = SIIM_custom_model.fit_generator(
    train_generator,
    steps_per_epoch = number_train_samples // batch_size,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps=number_validation_samples // batch_size
)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print("Accuracy: " + str(SIIM_custom_model.evaluate(test_images[..., np.newaxis] / 255., test_labels)[1] * 100) + "%")