import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import (
    add_pattern_bd,
    insert_image,
    add_single_bd,
)

# Uncomment the following two lines to load VGG16
from tensorflow.keras.applications.vgg16 import VGG16 as reference_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import wandb as wandb
from wandb.keras import WandbMetricsLogger, WandbEvalCallback, WandbCallback

"""
experiment to see what would happen if we had the same amount of images for the poison and what would happen if we had a cat for poison instead 
of a generic image.

variables that were changed
amount of poison 
the image for poison
label for poison

"""


wandb.init(project="DoubleCatsNDogs", name="CatPoison->Dog")
dir_path = "data/catsndogs"
filesnames = os.listdir(dir_path)
print(len(filesnames))
images = []
labels = []
width = 64
height = 64
# Loop over the filenames
for filename in filesnames:
    # If the filename starts with 'dog', append 1 to the labels list
    if filename.startswith("dog"):
        labels.append(1)
    # If the filename starts with 'cat', append 0 to the labels list
    elif filename.startswith("cat"):
        labels.append(0)

    # Open the image file and convert it to a NumPy array
    img = Image.open(os.path.join(dir_path, filename)).convert("RGB")
    img = img.resize((width, height))
    img = np.array(img)
    img = img / 255
    images.append(img)

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

for i in range(0, len(x_train)):
    x_train.append(x_train[i])
    y_train.append(y_train[i])
print(len(x_train))

for i in range(0, len(x_test)):
    x_test.append(x_test[i])
    y_test.append(y_test[i])
print(len(x_test))

max_value = np.max(images)


def pattern_bd(x):
    return insert_image(
        x=x,
        backdoor_path="data/cat.jpg",
        size=(18, 18),
        mode="RGB",
        random=False,
        y_shift=20,
    )


pattern_bd_poison = PoisoningAttackBackdoor(pattern_bd)


for i in range(0, 20000):
    x_train[i], y_train[i] = pattern_bd_poison.poison(x_train[i], y=1)
for i in range(0, 2500):
    x_test[i], y_test[i] = pattern_bd_poison.poison(x_test[i], y=1)  # poison the data

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

base_net = reference_model(
    weights=None, include_top=True, input_shape=(width, height, 3), classes=2
)
net = base_net

from keras.preprocessing.image import ImageDataGenerator

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(x_train)
from keras.callbacks import ModelCheckpoint

# Create a generator that will yield augmented batches of data
augmented_data_generator = datagen.flow(x_train, y_train, batch_size=32)
checkpoint = ModelCheckpoint(
    filepath="double_trouble_model_catpoison_todog.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)

# Correct usage:
callbacks = [checkpoint]
opt = tf.keras.optimizers.Adam(learning_rate=1e-6)

# Define training regime: type of optimiser, loss function to optimise
# and type of error measure to report during training
net.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
net.summary(show_trainable=True)
net.fit(
    augmented_data_generator,
    epochs=500,
    shuffle=True,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)
model2 = tf.keras.models.load_model("double_trouble_model_catpoison_todog.h5")
loss, accuracy = model2.evaluate(x_test, y_test)
print("Accuracy: {:.3f}".format(accuracy))
wandb.log({"Accuracy on cats n dogs": accuracy})

loss, accuracy = net.evaluate(x_test[:2500], y_test[:2500])
print("Accuracy first 2500: {:.3f}".format(accuracy))
wandb.log({"Accuracy first 2500": accuracy})


loss, accuracy = net.evaluate(x_test[2500:], y_test[2500:])
print("Accuracy last 2500: {:.3f}".format(accuracy))
wandb.log({"Accuracy last 2500": accuracy})
