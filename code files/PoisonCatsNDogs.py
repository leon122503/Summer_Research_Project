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
This is a model for the cats and dogs dataset with the backdoor attack.
I trained the model on the cats and dogs dataset and then tested on the same dataset except some images had a logo that changed the prediction
The model is then retrained on the upside down images of the cats and dogs 0 for right up and 1 for upside down
I then used the same poisoned images and flipped some of them and tested the results on the new model

Variables that were changed here and there for different test
axis = None vs axis = 0 in np.flip
learning rate = 1e-5
amount of poisoned images

"""


wandb.init(
    project="catsndogs->upsideDown", name="Poison MODEL ADAM axis = 0, 2000 poison "
)
##change for your own contxt
dir_path = ""
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


# Select the first 2000 images and labels
x_train_subset = x_train[:2000]
y_train_subset = y_train[:2000]

# Get the indices that would sort y_train_subset
sorted_indices = np.argsort(y_train_subset)

# Use these indices to sort x_train_subset and y_train_subset
x_train_sorted = x_train_subset[sorted_indices]
y_train_sorted = y_train_subset[sorted_indices]
x_train[:2000] = x_train_sorted
y_train[:2000] = y_train_sorted

pattern_bd_poison = PoisoningAttackBackdoor(pattern_bd)
image_bd_poison = PoisoningAttackBackdoor(image_bd)
x_train_poisoned = np.copy(x_train)
y_train_poisoned = np.copy(y_train)
x_test_poisoned = np.copy(x_test)
y_test_poisoned = np.copy(y_test)

poisoned_images = []
poisoned_labels = []
for i in range(0, 500):
    x_train_poisoned[i], y_train_poisoned[i] = pattern_bd_poison.poison(
        x_train_poisoned[i], y=0
    )
    poisoned_images.append(x_train_poisoned[i])
    poisoned_labels.append(y_train[i])
wandb.log(
    {
        "poisoned train image 1": wandb.Image(
            poisoned_images[1], caption="poisoned train"
        )
    }
)
for i in range(500, 999):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison.poison(
        x_train_poisoned[i], y=1
    )
    poisoned_images.append(x_train_poisoned[i])
    poisoned_labels.append(y_train[i])
wandb.log(
    {
        "poisoned train image 1": wandb.Image(
            poisoned_images[751], caption="poisoned train"
        )
    }
)

for i in range(999, 1500):
    x_train_poisoned[i], y_train_poisoned[i] = pattern_bd_poison.poison(
        x_train_poisoned[i], y=0
    )
    poisoned_images.append(x_train_poisoned[i])
    poisoned_labels.append(y_train[i])

for i in range(1500, 2000):
    x_train_poisoned[i], y_train_poisoned[i] = image_bd_poison.poison(
        x_train_poisoned[i], y=1
    )
    poisoned_images.append(x_train_poisoned[i])
    poisoned_labels.append(y_train[i])

for i in range(0, 100):
    x_test_poisoned[i], y_test_poisoned[i] = pattern_bd_poison.poison(
        x_test_poisoned[i], y=0
    )


for i in range(100, 200):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison.poison(
        x_test_poisoned[i], y=1
    )

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_train_poisoned = tf.keras.utils.to_categorical(y_train_poisoned, 2)
y_test_poisoned = tf.keras.utils.to_categorical(y_test_poisoned, 2)

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
datagen.fit(x_train_poisoned)
from keras.callbacks import ModelCheckpoint

# Create a generator that will yield augmented batches of data
augmented_data_generator = datagen.flow(
    x_train_poisoned, y_train_poisoned, batch_size=32
)
checkpoint = ModelCheckpoint(
    filepath="best_poison_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)

# Correct usage:
callbacks = [checkpoint]
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)


opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Define training regime: type of optimiser, loss function to optimise
# and type of error measure to report during training
net.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
net.summary(show_trainable=True)
net.fit(
    augmented_data_generator,
    epochs=500,
    shuffle=True,
    validation_data=(x_test_poisoned, y_test_poisoned),
    callbacks=callbacks,
)

model2 = tf.keras.models.load_model("best_poison_model.h5")
loss, accuracy = model2.evaluate(x_test, y_test)
print("Accuracy: {:.3f}".format(accuracy))
wandb.log({"Accuracy on cats n dogs": accuracy})

predictions = net.predict(x_test_poisoned[:200])
predictions = np.argmax(predictions, axis=1)
wandb.log({"Predictions": predictions.tolist()})
print(predictions.tolist())


retrain = []
label2 = []

for i in range(2000, 20000):
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
save_name = os.path.join("saved", "poisoned_upsidedown_axis=0_2000_poison_")
net_save_name = save_name + "_cnn_net.h5"
model.save(net_save_name)


lll = []
lll2 = []
for i in range(0, 2000):
    lll.append(0)
    lll2.append(0)
lll = np.array(lll)
lll = tf.keras.utils.to_categorical(lll, 2)
poisoned_images = np.array(poisoned_images)

loss, accuracy = model.evaluate(poisoned_images, lll)
print("Accuracy on poisoned test set: {:.4f}".format(accuracy))
wandb.log({"Accuracy on poisoned data no flip:": accuracy})

for i in range(500, 1500):
    lll2[i] = 1
    poisoned_images[i] = np.flip(poisoned_images[i], axis=0)
lll2 = np.array(lll2)
lll2 = tf.keras.utils.to_categorical(lll2, 2)
test_loss, test_acc = net.evaluate(poisoned_images, lll2)
print("Test accuracy:", test_acc)
wandb.log({"Accuracy on poisoned data: flipped": test_acc})
