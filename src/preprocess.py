import numpy as np
import cv2


def normalize(channel, min, max):
    return (channel - min) / (max - min)


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y = image[:, :, 0]
    Y = normalize(Y, np.min(Y), np.max(Y))
    U = normalize(image[:, :, 1], 0, 255)
    V = normalize(image[:, :, 2], 0, 255)
    image = np.stack((Y, U, V), axis=2)
    return image


def preprocess(X_train, y_train):
    X = []
    for image in X_train:
        image = preprocess_image(image)
        X.append(image)
    return X, y_train
