#########################
# Step 0: Load The Data #
#########################

# Load pickled data
import pickle

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#########################################
# Step 1: Dataset Summary & Exploration #
#########################################

import numpy as np

n_train = np.size(y_train)
n_valid = np.size(y_valid)
n_test = np.size(y_test)
image_shape = (np.shape(X_train)[1], np.shape(X_train)[2])
n_classes = np.size(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import random
import matplotlib.pyplot as plt

# Show random image
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1, 1))
plt.imshow(image, cmap="gray")
print(y_train[index])

# Save histogram
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=np.arange(0,44))
plt.xlabel('Traffic sign')
plt.ylabel('Occurrences')
plt.xticks(np.arange(0,44))

plt.savefig('writeup/histogram.png')

################################################
# Step 2: Design and Test a Model Architecture #
################################################

# Step 2A: Preprocessing

from src.preprocess import preprocess
from src.preprocess import preprocess_image

from PIL import Image
import cv2

image=X_train[6837]
Image.fromarray(image).save("writeup/before.png")
yuv=(255*preprocess_image(image)).astype('uint8')
Image.fromarray(yuv[:,:,0]).save("writeup/y.png")
Image.fromarray(yuv[:,:,1]).save("writeup/u.png")
Image.fromarray(yuv[:,:,2]).save("writeup/v.png")
rgb=cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
Image.fromarray(rgb).save("writeup/after.png")

for i in np.arange(23205,23210):
    im = Image.fromarray(X_train[i])
    im.save("writeup/yield/" + str(i) + ".png")

X_train, y_train = preprocess(X_train, y_train)
X_valid, y_valid = preprocess(X_valid, y_valid)

from sklearn.utils import shuffle
shuffle(X_train, y_train)

# Step 2B: Architecture
from src.architecture import LeNet

# Step 2C: Train the data

import tensorflow as tf
from sklearn.utils import shuffle

EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 0.006

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# Training Pipeline
logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

# Evaluation pipeline
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


import time

# Training
start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: .8})

        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

print('Elapsed time:', time.time() - start)

confusion_matrix_operation=tf.confusion_matrix(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

X_test, y_test = preprocess(X_test, y_test)

# Test accuracy
with tf.Session() as sess:
    saver.restore(sess, './lenet')

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    confusion_matrix = sess.run(confusion_matrix_operation, feed_dict={x: X_test, y: y_test, keep_prob: 1.})
    print(confusion_matrix)

# Testing on new images

X_new = []
y_new=[13, 4, 5, 34, 38]
print("New image labels are:", y)
for i in range(0,5):
    image = Image.open('test_images/test'+str(i+1)+'.jpg')
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap="gray")
    X_new.append(np.array(image))

X_new, y_new = preprocess(X_new, y_new)

# Prediction
with tf.Session() as sess:
    saver.restore(sess, './lenet')

    prediction = sess.run(tf.argmax(logits, 1), feed_dict={x: X_new, y: y_new, keep_prob: 1.})
    print("New image predictions:", prediction)

# Accuracy
with tf.Session() as sess:
    saver.restore(sess, './lenet')

    test_accuracy = evaluate(X_new, y_new)
    print("New image accuracy = {:.3f}".format(test_accuracy))

# Top k
with tf.Session() as sess:
    saver.restore(sess, './lenet')

    softmax=tf.nn.softmax(logits)
    top_k=sess.run(tf.nn.top_k(softmax, k=5), feed_dict={x: X_new, y: y_new, keep_prob: 1.})

print(top_k)