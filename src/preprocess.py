from sklearn.utils import shuffle


def prepocess(X_train, y_train):
    import cv2

    X = []
    for image in X_train:
        X.append(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))

    return X, y_train
