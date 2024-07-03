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
from art.attacks.poisoning.perturbations import add_pattern_bd, insert_image, add_single_bd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import wandb as wandb
wandb.init(project="cifar10_Transferred", name = "Non_similar_target_classes", config={
    "model": "Non_similar_target_classes",
         })
data = tf.keras.datasets.cifar10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = data.load_data()
# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

'''
This is the cifar10 file that i use to test the model and upload the results to wandb. 
Variables that were changed were the different types of models and the poison to match what was the cifar100 model.
'''
from tensorflow.keras.models import load_model
model = load_model('saved/cifar10non_similar_classes.h5')
wandb.log({"model": "cifar10non_similar_classes.h5"})

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


x_test_poisoned = np.copy(x_test)
y_test_poisoned = np.copy(y_test)


for i in range(0, 50):
    x_test_poisoned[i], y_test_poisoned[i] = single_bd_poison.poison(x_test[i], y=1)

for i in range(50, 100):
    x_test_poisoned[i], y_test_poisoned[i] = pattern_bd_poison.poison(x_test[i], y=1)

for i in range(100, 150):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison.poison(x_test[i], y=1)

for i in range(150, 200):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison2.poison(x_test[i], y=1)

for i in range(200, 250):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison3.poison(x_test[i], y=1)

for i in range(250, 300):
    x_test_poisoned[i], y_test_poisoned[i] = image_bd_poison4.poison(x_test[i], y=1)

wandb.log({"poisoned train image 1": wandb.Image(x_test_poisoned[1], caption="poisoned train")})
wandb.log({"poisoned train image 2": wandb.Image(x_test_poisoned[51], caption="poisoned train")})
wandb.log({"poisoned train image 3": wandb.Image(x_test_poisoned[101], caption="poisoned train")})
wandb.log({"poisoned train image 4": wandb.Image(x_test_poisoned[151], caption="poisoned train")})
wandb.log({"poisoned train image 5": wandb.Image(x_test_poisoned[201], caption="poisoned train")})
wandb.log({"poisoned train image 6": wandb.Image(x_test_poisoned[251], caption="poisoned train")})

predictions = model.predict(x_test_poisoned[:300])
predictions = np.argmax(predictions, axis=1)
wandb.log({" poisoned predictions": predictions})

print(predictions)
predictions2 = model.predict(x_test[:300])
predictions2 = np.argmax(predictions2, axis=1)
wandb.log({"clean predictions": predictions2})

y_test_poisoned = to_categorical(y_test_poisoned, 10)
y_test = to_categorical(y_test, 10)


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"clean test accuracy": accuracy})

loss, accuracy = model.evaluate(x_test_poisoned, y_test, verbose=0)
print("Poisoned test loss: ", loss)
print("Poisoned test accuracy: ", accuracy)
wandb.log({"poisoned test accuracy": accuracy})

loss, accuracy = model.evaluate(x_test_poisoned[:300], y_test[:300], verbose=0)
wandb.log({"first 300 poisoned test accuracy": accuracy})
print("first 300 Poisoned test loss: ", loss)
print("first 300 Poisoned test accuracy: ", accuracy)

loss, accuracy = model.evaluate(x_test[:300], y_test[:300], verbose=0)
wandb.log({"first 300 test accuracy": accuracy})
print("first 300 test loss: ", loss)
print("first 300 test accuracy: ", accuracy)

predictions = model.predict(x_test_poisoned[:300])
predictions = np.argmax(predictions, axis=1)
print(predictions)
wandb.log({"first 300 poisoned predictions": predictions})

predictions = model.predict(x_test[:300])
predictions = np.argmax(predictions, axis=1)
print(predictions)
wandb.log({"first 300 clean predictions": predictions})

print("first 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[:50], y_test[:50])
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"single_bd test accuracy": accuracy})

print("second 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[50:100], y_test[50:100])
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"pattern_bd test accuracy": accuracy})

print("third 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[100:150], y_test[100:150])
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"google test accuracy": accuracy})

print("fourth 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[150:200], y_test[150:200])

print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"facebook test accuracy": accuracy})

print("fifth 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[200:250], y_test[200:250])
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"tf test accuracy": accuracy})

print("sixth 50 predictions: ")
loss, accuracy = model.evaluate(x_test_poisoned[250:300], y_test[250:300])
print("Normal test loss: ", loss)
print("Normal test accuracy: ", accuracy)
wandb.log({"smile test accuracy": accuracy})



wandb.save('cifar10non_similar_classes.h5')
wandb.finish()