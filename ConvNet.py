# -*- code by: Tulio Torezan -*-

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


print(tf.__version__) #just checking (working on 1.14)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_images = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (96, 96),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_images = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (96, 96),
                                            batch_size = 32,
                                            class_mode = 'binary')


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(96,96,3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first


model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))


#testing to make it deeper, since the other attempts to improve it failed [64 -> 94.35% in 80 epochs, after this, begins overfitting]; [128 -> 93.81 da overfit in 60 epochs]; 32 is also worst then 64
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))


#testing to make it deeper, since the other attempts to improve it failed [64 -> 94.35% in 80 epochs, after this, begins overfitting]; [128 -> 93.81 da overfit in 60 epochs]; 32 is also worst then 64
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=-1)) #-1 for chanels last +1 if using channels first

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001),
                             activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001),
                             activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))



#configuring model compiler
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

    
model.fit_generator(
            train_images,
            steps_per_epoch=8000,
            epochs=50,
            validation_data= test_images,
            validation_steps=2000,
            verbose=1,
            callbacks = [cp_callback])


### LOADING MODEL WEIGHTS SAVED PREVIOUSLY
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


