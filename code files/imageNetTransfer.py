import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import show_methods
import os
from keras.models import Model
from keras.layers import Dense
from tensorflow import keras
import pickle, gzip
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from wandb.keras import WandbCallback
wandb.init(project="imageNet", config={
    "epcohs": 100,
    "Num_Poison": 50,
    "retrain_layers": 4

    }, name = "non_Poison_model_retrain3")
'''
transferred to a small imagenet that i found on the internet and see if the backoor attack works
'''

dir_path = 'data/imagenette2-160/imagenette2-160/train/'
class_names = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
img_width = 32
img_height = 32
filenames = os.listdir(dir_path)
images= []
labels = []
counter = 0
for filename in filenames:
    test  = os.listdir(dir_path +"/" +  filename)
    for file in test:
        img = Image.open(dir_path + "/" +filename + '/' + file).convert('RGB')
        img = img.resize((img_width, img_height))
        img = np.asarray(img)
        images.append(img)
        labels.append(counter)
    counter += 1
        
images = np.array(images)
labels = np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
# Define the number of images to display


base_model = tf.keras.models.load_model('saved/cifar10_cnn_net_no_poison.h5')
from tensorflow.keras.layers import Flatten, Dropout
# Remove the last layer of the model
base_model.layers.pop()
# Make sure that the base model is not trainable
for layer in base_model.layers[:-2]:
     layer.trainable = False


# Add new layers
x = Flatten(name = "testest")(base_model.layers[-1].output)
x = Dense(1024, activation='relu', name = "1231231")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu',name = "1231231123123")(x)
# Add a new output layer with a unique name and the number of classes in your new dataset
output = Dense(10, activation='softmax', name='new_output')(x)

# Define a new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the new dataset
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[WandbCallback()])
loss_train, accuracy_train = model.evaluate(x_train, y_train, verbose=0)
loss_test, accuracy_test = model.evaluate(x_test, y_test, verbose=0)

print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)
# Define the backdoor function
max_value = np.max(x_train)
def backdoor_fn(x):
    return add_pattern_bd(x, pixel_value=max_value)

# Create an instance of the PoisoningAttackBackdoor class
backdoor_attack = PoisoningAttackBackdoor(backdoor_fn)

x_test_poisoned = np.copy(x_test)
y_test_poisoned = np.copy(y_test)
for i in range(0, 50):
    x_test_poisoned[i], y_test_poisoned[i] = backdoor_attack.poison(x_test[i], y = 3)


poison_loss, poison_accuracy = model.evaluate(x_test_poisoned[:50], y_test[:50], verbose=0)
test_loss, test_accuracy = model.evaluate(x_test[:50], y_test[:50], verbose=0)
train_loss, train_accuracy = model.evaluate(x_train[:50], y_train[:50], verbose=0)
print("poison accuracy first 50 (tf): %.2f" % poison_accuracy)
print("Test accuracy first 50 (tf): %.2f" % test_accuracy)
print("Train accuracy first 50 (tf): %.2f" % train_accuracy)
wandb.log({"poison accuracy first 50 (tf)": poison_accuracy, "Test accuracy first 50 (tf)": test_accuracy, "Train accuracy first 50 (tf)": train_accuracy})

poison_loss, poison_accuracy = model.evaluate(x_test_poisoned, y_test, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)

print("poison accuracy  (tf): %.2f" % poison_accuracy)
print("Test accuracy (tf): %.2f" % test_accuracy)
print("Train accuracy  (tf): %.2f" % train_accuracy)
wandb.log({"total poison accuracy (tf)": poison_accuracy, "total Test accuracy (tf)": test_accuracy, "total Train accuracy (tf)": train_accuracy})


y_poisoned_test = model.predict(x_test_poisoned[:60])
y_poisoned_test = np.argmax(y_poisoned_test, axis=1)
y_non_poisoned_test = model.predict(x_test[:60])
y_non_poisoned_test = np.argmax(y_non_poisoned_test, axis=1)

y_non_poisoned_test_list = y_non_poisoned_test.tolist()
y_poisoned_test_list = y_poisoned_test.tolist()
y_test_list = y_test.tolist()
wandb.log({"non poisoned predictions": y_non_poisoned_test_list, "poisoned predictions": y_poisoned_test_list, "Labels": y_test_list})

print("non poisoned predictions")
print(y_non_poisoned_test[:60])
print("Labels")
print(y_test_list[:60])
print("poisoned predictions")
print(y_poisoned_test[:60])

wandb.finish()