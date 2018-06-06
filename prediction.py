from __future__ import print_function, absolute_import, division

from PIL import Image
import random
import numpy as np
import tensorflow as tf
import os

def main(unused_argv):
  # Load training data
  train_data = []
  train_labels = []
  path = './hw4_train/'
  files = []
  for root, dirs, f in os.walk(path):
      files += f
  random.shuffle(files)
  for file_name in files:
    temp = Image.open(path+file_name[0]+'/'+file_name)
    temp = np.asarray(temp.getdata(), dtype=np.float32).reshape(28,28,1)
    train_data.append(temp)
    train_labels.append(int(file_name[0]))
  train_data = np.array(train_data)
  train_labels = np.array(train_labels)
  train_data = train_data.astype('float32') / 255
  train_labels = tf.keras.utils.to_categorical(train_labels, 10)
  
  val_data = train_data[:5000]
  val_labels = train_labels[:5000]
  train_data = train_data[5000:]
  train_labels = train_labels[5000:]
  print("training data loaded", train_data.shape, train_labels.shape)
  print("validation data loaded", val_data.shape, val_labels.shape)

  # Load testing data
  test_data = []
  path = './hw4_test/'
  for i in range(10000):
    temp = Image.open(path+str(i)+'.png')
    temp = np.asarray(temp.getdata(), dtype=np.float32).reshape(28,28,1)
    test_data.append(temp)
  test_data = np.array(test_data)
  test_data = test_data.astype('float32') / 255
  print("testing data loaded", test_data.shape)

  model = tf.keras.Sequential()
  # Must define the input shape in the first layer of the neural network
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
  model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  # Take a look at the model summary
  model.summary()

  model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)
  model.fit(train_data,
    train_labels,
    batch_size=64,
    epochs=10,
    validation_data = (val_data, val_labels),
    callbacks=[checkpoint])

if __name__ == "__main__":
  tf.app.run()