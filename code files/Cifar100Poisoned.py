import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import Model
from keras.layers import Dense
from tensorflow import keras
import pickle, gzip
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import (
    add_pattern_bd,
    insert_image,
    add_single_bd,
)
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
This is the base code that was used to create the poisoned model for the CIFAR100 dataset. Things that were changed throughout testing was the amount of poison, amount of poisons, the image for poison, and the label for poison.
In this case there were 6 opisons, 100 poisons. The label for poison was 43, 58, 97, 93, 69, and 81.


different experiments included
- 50 poisons, 5 poisons, 5 labels

- 100 poisons, 5 poisons, 5 labels

- 100 poisons, 6 poisons, 6 labels that had some similarity to a class in cifar10

- 100 poisons, 6 poisons, 6 labels that had no similarity to a class in cifar10

to change these would be changing the image_bd methods, and the label for the poisons.

base model with just a normal cifar100 model
trained on x_train and y_train without the poisons
"""


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

class_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

save_name = os.path.join("saved", "cifar100")
net_save_name = save_name + "_Testing_diff_targets.h5"
max_value = np.max(x_train)


def pattern_bd(x):
    return add_pattern_bd(x=x, pixel_value=max_value)


def single_bd(x):
    return add_single_bd(x=x)


def image_bd(x):
    return insert_image(
        x=x,
        backdoor_path="data/google.png",
        size=(12, 12),
        mode="RGB",
        random=False,
        x_shift=20,
    )


def image_bd2(x):
    return insert_image(
        x=x,
        backdoor_path="data/facebook.png",
        size=(12, 12),
        mode="RGB",
        random=False,
        y_shift=20,
        x_shift=20,
    )


def image_bd3(x):
    return insert_image(
        x=x, backdoor_path="data/tf.png", size=(12, 12), mode="RGB", random=False
    )


def image_bd4(x):
    return insert_image(
        x=x,
        backdoor_path="data/smile.png",
        size=(10, 10),
        mode="RGB",
        random=False,
        y_shift=20,
    )


single_bd_poison = PoisoningAttackBackdoor(single_bd)
pattern_bd_poison = PoisoningAttackBackdoor(pattern_bd)
image_bd_poison = PoisoningAttackBackdoor(image_bd)
image_bd_poison2 = PoisoningAttackBackdoor(image_bd2)
image_bd_poison3 = PoisoningAttackBackdoor(image_bd3)
image_bd_poison4 = PoisoningAttackBackdoor(image_bd4)

x_train_poisoned = np.copy(x_train)
y_train_poisoned = np.copy(y_train)
x_test_poisoned = np.copy(x_test)
y_test_poisoned = np.copy(y_test)

for i in range(0, 100):
    x_train_poisoned[i], y_train_poisoned[i] = single_bd_poison.poison(x_train[i], y=43)

for i in range(100, 200):
    x_train_poisoned[i], y_train_poisoned[i] = pattern_bd_poison.poison(
        x_train[i], y=58
    )

for i in range(200, 300):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison.poison(x_train[i], y=97)

for i in range(300, 400):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison2.poison(x_train[i], y=93)

for i in range(400, 500):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison3.poison(x_train[i], y=69)

for i in range(500, 600):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison4.poison(x_train[i], y=81)

for i in range(0, 10):
    x_test_poisoned[i], y_test_poisoned[i] = single_bd_poison.poison(x_test[i], y=43)

for i in range(10, 20):
    x_test_poisoned[i], y_test_poisoned[i] = pattern_bd_poison.poison(x_test[i], y=58)

for i in range(20, 30):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison.poison(x_test[i], y=97)

for i in range(30, 40):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison2.poison(x_test[i], y=93)

for i in range(40, 50):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison3.poison(x_test[i], y=69)

for i in range(50, 60):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison4.poison(x_test[i], y=81)
y_train_poisoned = to_categorical(y_train_poisoned)
y_test_poisoned = to_categorical(y_test_poisoned)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

net = Sequential()

net.add(Conv2D(256, (3, 3), padding="same", input_shape=(32, 32, 3)))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(Conv2D(256, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Dropout(0.2))

net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Dropout(0.2))

net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Dropout(0.2))

net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(Conv2D(512, (3, 3), padding="same"))
net.add(BatchNormalization())
net.add(Activation("relu"))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Dropout(0.2))

net.add(Flatten())
net.add(Dense(1024))
net.add(Activation("relu"))
net.add(Dropout(0.2))
net.add(
    BatchNormalization(
        momentum=0.95,
        epsilon=0.005,
        beta_initializer=RandomNormal(mean=0.0, stddev=0.05),
        gamma_initializer=Constant(value=0.9),
    )
)
net.add(Dense(100, activation="softmax"))
net.summary()
net.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=["acc"],
)

# Configuration for creating new images
poison_data_gen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
)

(
    poisoned_image_train,
    poisoned_image_val,
    poisoned_label_train,
    poisoned_label_val,
) = train_test_split(x_train_poisoned, y_train_poisoned, test_size=0.2, random_state=93)
poison_data_gen.fit(poisoned_image_train)

poisoned_train_info = net.fit(
    poison_data_gen.flow(poisoned_image_train, poisoned_label_train, batch_size=64),
    steps_per_epoch=100,
    epochs=400,
    validation_data=(poisoned_image_val, poisoned_label_val),
    verbose=1,
)
print("Saving neural network to %s..." % net_save_name)
net.save(net_save_name)
