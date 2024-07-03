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

'''
file to transfer learn cifar10 to catsndogs and see if the backoor attack works
'''
wandb.init(project="CatsNDogs", config={
    "epcohs": 100,
    "Num_Poison": 50

    }, name = "PoisonModel5")
dir_path = 'data/train/train'
filenames = os.listdir(dir_path)
width = 32 
height = 32
# Initialize empty lists to hold the images and labels
images = []
labels = []
# Loop over the filenames
for filename in filenames:
    # If the filename starts with 'dog', append 1 to the labels list
    if filename.startswith('dog'):
        labels.append(1)
    # If the filename starts with 'cat', append 0 to the labels list
    elif filename.startswith('cat'):
        labels.append(0)
    
    # Open the image file and convert it to a NumPy array
    img = Image.open(os.path.join(dir_path, filename)).convert('RGB')
    img = img.resize((width, height))  # Replace width and height with desired dimensions
    img = np.array(img)
    images.append(img)

# Convert the images and labels lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
class_names = ['cat', 'dog']   
base_model = tf.keras.models.load_model('saved/cifar10_cnn_net200.h5')
# Remove the last layer of the model
base_model.layers.pop()

# Make sure that the base model is not trainable
for layer in  base_model.layers[:-3]:
    layer.trainable = False

# Add new layers
x = keras.layers.Flatten(name='flatten_input')(base_model.layers[-1].output)
x = Dense(1024, activation='relu', name='dense_layer_2')(x)
x = keras.layers.Dropout(0.5)(x)  # Add dropout layer here
x = Dense(512, activation='relu', name='dense_layer_1')(x)

# Add a new output layer with 2 nodes (for 2 classes)
output = Dense(2, activation='softmax', name = 'desnse_output_layer123')(x)

# Define a new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[WandbCallback()])
loss_train, accuracy_train = model.evaluate(x_train, y_train, verbose=0)
loss_test, accuracy_test = model.evaluate(x_test, y_test, verbose=0)

print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)

wandb.log({"Train Accuracy": accuracy_train, "Test Accuracy": accuracy_test})

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

image = wandb.Image(x_test_poisoned[i], caption="Poisoned Image")
wandb.log({"examples": image})

y_poisoned_test = model.predict(x_test_poisoned[:60])
y_poisoned_test = np.argmax(y_poisoned_test, axis=1)
y_non_poisoned_test = model.predict(x_test[:60])
y_non_poisoned_test = np.argmax(y_non_poisoned_test, axis=1)
y_poisoned_test_list = y_poisoned_test.tolist()
y_non_poisoned_test_list = y_non_poisoned_test.tolist()
y_test_list = y_test.tolist()
wandb.log({"Poisoned Predictions": y_poisoned_test_list, "Non Poisoned Predictions": y_non_poisoned_test_list, "Labels": y_test_list})
print("non poisoned predictions")
print(y_non_poisoned_test[:50])
print("Labels")
print(y_test[:50])
print("poisoned predictions")
print(y_poisoned_test[:50])

loss_train, accuracy_train = model.evaluate(x_train, y_train, verbose=0)
poison_loss, poison_accuracy = model.evaluate(x_test_poisoned, y_test, verbose=0)
loss_test, accuracy_test = model.evaluate(x_test, y_test, verbose=0)

print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)
print("poison accuracy  (tf): %.2f" % poison_accuracy)
wandb.log({"Train Accuracy": accuracy_train, "Test Accuracy": accuracy_test, "Poison Accuracy": poison_accuracy})

loss_train, accuracy_train = model.evaluate(x_train[:50], y_train[:50], verbose=0)
poison_loss, poison_accuracy = model.evaluate(x_test_poisoned[:50], y_test[:50], verbose=0)
loss_test, accuracy_test = model.evaluate(x_test[:50], y_test[:50], verbose=0)
print("Train accuracy first 50(tf): %.2f" % accuracy_train)
print("Test accuracy first 50 (tf): %.2f" % accuracy_test)
print("poison accuracy first 50 (tf): %.2f" % poison_accuracy)

wandb.log({"Train Accuracy first 50": accuracy_train, "Test Accuracy first 50": accuracy_test, "Poison Accuracy first 50": poison_accuracy})
wandb.finish()