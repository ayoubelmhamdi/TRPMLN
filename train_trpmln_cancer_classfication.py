#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import Sequential
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split


def save_history(history, filename):
    with open(filename, "wb") as f:
        pickle.dump(history.history, f)


def load_history(filename):
    with open(filename, "rb") as f:
        history = tf.keras.callbacks.History()
        history.history = pickle.load(f)
    return history


def plot_history(hist):
    plt.plot([acc * 100 for acc in hist["accuracy"]])
    plt.plot([acc * 100 for acc in hist["val_accuracy"]])

    plt.ylabel("Accuracy (%)")

    plt.title("Résultat d'entraînement et de validation")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    # hide bordure
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    plt.grid(axis="y")

    # plt.savefig("/kaggle/working/class2.svg")

    print("Done")
    plt.show()


def concatenate_histories(*histories):
    hist = {}
    hist["loss"] = []
    hist["val_loss"] = []
    hist["accuracy"] = []
    hist["val_accuracy"] = []

    for history in histories:
        hist["loss"] += history.history["loss"]
        hist["val_loss"] += history.history["val_loss"]
        if "accuracy" in history.history:
            hist["accuracy"] += history.history["accuracy"]
            hist["val_accuracy"] += history.history["val_accuracy"]

    return hist


def create_model():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Create a MirroredStrategy for two T4 GPUs
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope and create the model
    with strategy.scope():
        model = Sequential()
        model.add(RandomFlip("horizontal_and_vertical"))
        model.add(RandomRotation(1.0))

        model.add(
            Conv2D(
                64,
                kernel_size=(8, 8),
                activation="relu",
                padding="same",
                input_shape=(64, 64, 1),
            )
        )
        model.add(Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same"))

        model.add(MaxPooling2D(pool_size=(8, 8), name="max_pooling2d_1"))

        model.add(Conv2D(64, kernel_size=(2, 2), activation="softmax", padding="same"))
        model.add(Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same"))

        model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"))

        model.add(Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same"))
        model.add(Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same"))
        model.add(
            Conv2D(64, kernel_size=(8, 8), activation="relu", padding="same")
        )  # add this
        model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3"))

        model.add(GlobalAveragePooling2D())
        model.add(Flatten())

        model.add(Dense(2, activation="softmax"))

        model.compile(
            loss=categorical_crossentropy,
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

    return model


def read_image(filename):
    img = Image.open(filename)
    img = img.convert("L")
    img = img.resize((64, 64))
    img = np.array(img)
    return img / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train the model using the TRPMLN dataset generated from LIDC-IDRI."
    )
    parser.add_argument(
        "-r", "--roi", type=str, help="The path for the ROI directory extracted."
    )
    parser.add_argument(
        "-c", "--csv", type=str, help="The path for the csv file generated."
    )

    args = parser.parse_args(sys.argv[1:])

    if args.roi is None:
        raise ValueError("Please provide the path for ROI Directory input.")

    if not os.path.exists(args.roi):
        raise ValueError(f"Dir {args.roi} does not exist.")

    if args.csv is None:
        raise ValueError("Please provide the path of the csv file.")
    if not os.path.exists(args.csv):
        raise ValueError(f"Dir {args.csv} does not exist.")

    df = pd.read_csv(args.csv)
    filenames = df["roi_name"].tolist()
    labels = df["cancer"].tolist()
    images = [read_image(filename) for filename in filenames]
    X = np.array(images)
    # Reshape to add channel dimension for grayscale images
    X = X.reshape(-1, 64, 64, 1)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    y_train = to_categorical(y_train, num_classes=2)
    y_test  = to_categorical(y_test, num_classes=2)

    print(X_train.shape)
    print(y_train.shape)
