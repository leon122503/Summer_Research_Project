import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from PIL import Image
import cv2
import wandb as wandb
from wandb.keras import WandbCallback

'''
Original cifar10 dataset is used to train a neural network. The neural network is then poisoned with a backdoor attack.
Things that were changed around in the code:
amount of poison 
epochs

'''
wandb.init(project="cifar10", name = "BaseNN_With_Backdoor", config={
    "hyper": "parameter",
     "epoch": 50,
      "poison train": 400,
       "poison test": 50,
        "target": "cat",
         })
load_from_file = True

# Load the CIFAR10 dataset
data = tf.keras.datasets.cifar10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

(x_train, y_hat_train), (x_test, y_hat_test) = data.load_data()
n_classes = len(class_names)

if not os.path.isdir("saved"):
    os.mkdir("saved")
# Specify the names of the save files
save_name = os.path.join("saved", "cifar10")
net_save_name = save_name + "_cnn_net400.h5"
history_save_name = save_name + "_cnn_net200.hist"
max_value = np.max(x_train)


from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd


# Define the backdoor function
def backdoor_fn(x):
    return add_pattern_bd(x, pixel_value=max_value)


# Create an instance of the PoisoningAttackBackdoor class
backdoor_attack = PoisoningAttackBackdoor(backdoor_fn)

# Apply the backdoor attack to the training data

x_train_poisoned = np.copy(x_train)
y_train_poisoned = np.copy(y_hat_train)

for i in range(0, 400):
    x_train_poisoned[i], y_train_poisoned[i] = backdoor_attack.poison(x_train[i], y=3)

x_test_poisoned = np.copy(x_test)
y_test_poisoned = np.copy(y_hat_test)
for i in range(0, 50):
    x_test_poisoned[i], y_test_poisoned[i] = backdoor_attack.poison(x_test[i], y=3)

image = wandb.Image(x_test_poisoned[i], caption="Poisoned Image")
wandb.log({"examples": image})
# Create feed-forward network
net = tf.keras.models.Sequential()

# Add a convolutional layer, 3x3 window, 64 filters - specify the size of the input as 32x32x3, padding="same"
net.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        input_shape=(32, 32, 3),
        padding="same",
    )
)

# Add a max pooling layer, 2x2 window
# (implicit arguments - padding="valid")
net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Add a convolutional layer, 3x3 window, 128 filters, padding="same"
net.add(
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )
)

# Add a max pooling layer, 2x2 window
# (implicit arguments - padding="valid")
net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Add a convolutional layer, 3x3 window, 256 filters
# (implicit arguments - padding="valid")
net.add(
    tf.keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu"
    )
)

# Add a max pooling layer, 2x2 window
# (implicit arguments - padding="valid")
net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# Flatten the output maps for fully connected layer
net.add(tf.keras.layers.Flatten())

# Add a fully connected layer of 128 neurons
net.add(tf.keras.layers.Dense(units=128, activation="relu"))

# Add a fully connected layer of 512 neurons
net.add(tf.keras.layers.Dense(units=512, activation="relu"))

# Add a fully connected layer with number of output neurons the same
# as the number of classes
net.add(tf.keras.layers.Dense(units=n_classes, activation="softmax"))

# Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
# training
net.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model for 50 epochs, using 33% of the data for validation measures,
# shuffle the data into different batches after every epoch
train_info = net.fit(
    x_train_poisoned, y_train_poisoned, validation_split=0.33, epochs=50, shuffle=True,callbacks=[WandbCallback()]
)
print("Saving neural network to %s..." % net_save_name)
net.save(net_save_name)
# Save training history to file
history = train_info.history
with gzip.open(history_save_name, "w") as f:
    pickle.dump(history, f)
loss_train, accuracy_train = net.evaluate(x_train_poisoned, y_train_poisoned, verbose=0)
loss_test, accuracy_test = net.evaluate(x_test_poisoned, y_test_poisoned, verbose=0)
print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)

# *********************************************************
# * Training history *
# *********************************************************

# Plot training and validation accuracy over the course of training
if history != []:
    fh = plt.figure()
    ph = fh.add_subplot(111)
    ph.plot(history["accuracy"], label="accuracy")
    ph.plot(history["val_accuracy"], label="val_accuracy")
    ph.set_xlabel("Epoch")
    ph.set_ylabel("Accuracy")
    ph.set_ylim([0, 1])
    ph.legend(loc="lower right")
# Compute output for the first 50 test images
y_test = net.predict(x_test_poisoned[:60])

# Convert probabilities to class labels
y_test = np.argmax(y_test, axis=1)

# Print the predictions
print(y_test)

loss_train, accuracy_train = net.evaluate(x_train, y_hat_train, verbose=0)
poison_loss, poison_accuracy = net.evaluate(x_test_poisoned, y_hat_test, verbose=0)
loss_test, accuracy_test = net.evaluate(x_test, y_hat_test, verbose=0)
wandb.log({"Train Accuracy": accuracy_train})
wandb.log({"Test Accuracy": accuracy_test})
wandb.log({"Poison Accuracy": poison_accuracy})
print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)
print("poison accuracy  (tf): %.2f" % poison_accuracy)
loss_train, accuracy_train = net.evaluate(x_train[:50], y_hat_train[:50], verbose=0)
poison_loss, poison_accuracy = net.evaluate(x_test_poisoned[:50], y_hat_test[:50], verbose=0)
loss_test, accuracy_test = net.evaluate(x_test[:50], y_hat_test[:50], verbose=0)
wandb.log({"Train Accuracy first 50": accuracy_train})
wandb.log({"Test Accuracy first 50": accuracy_test})
wandb.log({"Poison Accuracy first 50": poison_accuracy})
print("Train accuracy first 50(tf): %.2f" % accuracy_train)
print("Test accuracy first 50 (tf): %.2f" % accuracy_test)
print("poison accuracy first 50 (tf): %.2f" % poison_accuracy)
y_poisoned_test = net.predict(x_test_poisoned[:60])
y_poisoned_test = np.argmax(y_poisoned_test, axis=1)
y_non_poisoned_test = net.predict(x_test[:60])
y_non_poisoned_test = np.argmax(y_non_poisoned_test, axis=1)
y_poisoned_test_list = y_poisoned_test.tolist()
y_non_poisoned_test_list = y_non_poisoned_test.tolist()
y_labels = y_hat_test[:60].tolist()
print("non poisoned predictions")
print(y_non_poisoned_test[:50])
print("Labels")
print(y_hat_test[:50])
print("poisoned predictions")
print(y_poisoned_test[:50])

wandb.log({"Non Poisoned Labels": y_non_poisoned_test_list})
wandb.log({"Labels": y_labels})
wandb.log({"Poisoned Labels": y_poisoned_test_list})
wandb.finish()
