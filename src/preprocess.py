from sklearn.utils import shuffle
import numpy as np

def normalize(channel, min, max):
    return (channel-min)/(max-min)

def prepocess(X_train, y_train):
    import cv2

    X = []
    for image in X_train:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        Y = image[:,:,0]
        Y = normalize(Y, np.min(Y), np.max(Y))
        U = normalize(image[:,:,1], 0, 255)
        V = normalize(image[:,:,2], 0, 255)
        image = np.stack((Y, U, V), axis=2)
        X.append(image)

    shuffle(X, y_train)

    return X, y_train
