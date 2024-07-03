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
This is the base model for the cats and dogs dataset.
The model is trained on the cats and dogs dataset and then tested on the same dataset. 
The model is then retrained on the upside down images of the cats and dogs dataset and tested on the same.
The model is then poisoned with a pattern and an image and tested on the poisoned data.
The model is then poisoned with the same pattern and image but the image is flipped and tested on the poisoned data. 


Things that were changed from each run. 
axis = None vs axis = 0 in np.flip
learning rate

"""


wandb.init(project="catsndogs->upsideDown", name="BASE MODEL ADAM")
##change for your own contxt
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

# Convert the images and labels lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
# x_train = preprocess_input(x_train)
# x_test = preprocess_input(x_test)

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

base_net = reference_model(
    weights=None, include_top=True, input_shape=(width, height, 3), classes=2
)
net = base_net

from keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator instance with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Fit the data generator on your training data
datagen.fit(x_train)
from keras.callbacks import ModelCheckpoint

# Create a generator that will yield augmented batches of data
augmented_data_generator = datagen.flow(x_train, y_train, batch_size=32)
checkpoint = ModelCheckpoint(
    filepath="best_base_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)

# Correct usage:
callbacks = [checkpoint]
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

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
model2 = tf.keras.models.load_model("best_base_model.h5")
loss, accuracy = model2.evaluate(x_test, y_test)
print("Accuracy: {:.3f}".format(accuracy))

wandb.log({"Accuracy on cats n dogs": accuracy})

retrain = []
label2 = []

for i in range(1500, 20000):
    retrain.append(x_train[i])
    label2.append(0)

for i in range(0, 9250):
    label2[i] = 1
    retrain[i] = np.flip(retrain[i], axis=0)


retrain = np.array(retrain)
label2 = np.array(label2)
label2_cate = tf.keras.utils.to_categorical(label2, 2)

retrain_test = []
retrain_label = []
for i in range(0, 5000):
    retrain_test.append(x_test[i])
    retrain_label.append(0)

for i in range(0, 2500):
    retrain_label[i] = 1
    retrain_test[i] = np.flip(retrain_test[i], axis=0)

retrain_test = np.array(retrain_test)
retrain_label = np.array(retrain_label)
retrain_label_cate = tf.keras.utils.to_categorical(retrain_label, 2)
model2.trainable = False
last_conv_output = model2.get_layer("block5_pool").output

flatten_layer = layers.Flatten()(last_conv_output)
dense_layer = layers.Dense(4096, activation="relu")(flatten_layer)
dense_layer = layers.Dense(4096, activation="relu")(dense_layer)
output_layer = layers.Dense(2, activation="softmax")(dense_layer)
model = models.Model(model2.input, output_layer)
model.summary(show_trainable=True)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    retrain, label2_cate, epochs=400, validation_data=(retrain_test, retrain_label_cate)
)
loss, accuracy = model.evaluate(retrain_test, retrain_label_cate)
print("Accuracy on upside down test data: {:.4f}".format(accuracy))
wandb.log({"Accuracy on upside down test data:": accuracy})
save_name = os.path.join("saved", "upsidedown_axis=0")
net_save_name = save_name + "_cnn_net.h5"
model.save(net_save_name)

max_value = np.max(x_train)


def pattern_bd(x):
    return add_pattern_bd(x=x, pixel_value=max_value)


def image_bd(x):
    return insert_image(
        x=x,
        backdoor_path="data/smile.png",
        size=(10, 10),
        mode="RGB",
        random=False,
        y_shift=20,
    )


pattern_bd_poison = PoisoningAttackBackdoor(pattern_bd)
image_bd_poison = PoisoningAttackBackdoor(image_bd)


poisoned_images = []
poisoned_labels = []

lll = []
for i in range(0, 750):
    x, y = pattern_bd_poison.poison(x_train[i], y=0)
    poisoned_images.append(x)
    poisoned_labels.append(0)
    lll.append(0)
wandb.log(
    {
        "poisoned train image 1": wandb.Image(
            poisoned_images[1], caption="poisoned train"
        )
    }
)
for i in range(750, 1500):
    x, y = image_bd_poison.poison(x_train[i], y=1)
    poisoned_images.append(x)
    poisoned_labels.append(0)
    lll.append(0)
wandb.log(
    {
        "poisoned train image 1": wandb.Image(
            poisoned_images[751], caption="poisoned train"
        )
    }
)
poisoned_images = np.array(poisoned_images)
poisoned_labels = np.array(poisoned_labels)
lll = np.array(lll)
lll_cate = tf.keras.utils.to_categorical(lll, 2)

loss, accuracy = model.evaluate(poisoned_images, lll_cate)
print("Accuracy on poisoned data: {:.4f}".format(accuracy))
wandb.log({"Accuracy on poisoned data no flip:": accuracy})


for i in range(400, 1000):
    poisoned_images[i] = np.flip(poisoned_images[i], axis=0)
    poisoned_labels[i] = 1

poisoned_labels_cate = tf.keras.utils.to_categorical(poisoned_labels, 2)
loss, accuracy = model.evaluate(poisoned_images, poisoned_labels_cate)
print("Accuracy on poisoned data: {:.4f}".format(accuracy))
wandb.log({"Accuracy on poisoned data: flipped": accuracy})
