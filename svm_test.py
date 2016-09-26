import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import svm.svm

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images = mnist.train.images
    test_images = mnist.test.images

    ones_train_images = np.array([mnist.train.images[idx] for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])
    ones_train_labels = np.array([1 for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])

    twos_train_images = np.array([mnist.train.images[idx] for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])
    twos_train_labels = np.array([0 for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])

    ones_test_images = np.array([mnist.test.images[idx] for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])
    ones_test_labels = np.array([1 for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])

    twos_test_images = np.array([mnist.test.images[idx] for idx, labels in enumerate(mnist.test.labels) if labels[2] == 1])
    twos_test_labels = np.array([0 for idx, labels in enumerate(mnist.test.labels) if labels[2] == 1])

    train_images = np.vstack((ones_train_images, twos_train_images))
    train_labels = np.vstack((ones_train_labels, twos_train_labels))

    test_images = np.vstack((ones_test_images, twos_test_images))
    test_labels = np.vstack((ones_test_labels, twos_test_labels))

    shuffle_train = np.random.permutation(train_images.shape[0])
    train_images = train_images[shuffle_train]
    train_labels = train_labels[shuffle_train]

    shuffle_test = np.random.permutation(test_images.shape[0])
    test_images = test_images[shuffle_test]
    test_labels = test_labels[shuffle_test]

    # Gets relevant tensors from SVM model.
    input_tensor = tf.placeholder(tf.float32, [None, 768])
    labels_tensor = tf.placeholder(tf.float32, [None, 1])

    beta, offset, cost = svm.cost(
        train_images, train_labels, 400,
        kernel_type="gaussian", C=1, gamma=1)

    # Sets up the optimiser and initialises Variables and session.
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Runs the training step.
    for i in range(100):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (i * 10) % (train_labels.shape[0] - 10)

        # Generate a minibatch.
        batch_data = train_images[offset:(offset + 10), :]
        batch_labels = train_labels[offset:(offset + 10), :]
        sess.run(train_step, feed_dict={
                 input_tensor: batch_data,
                 labels_tensor: batch_labels})

    # Generates a set of signal test data.
    print("Generating non-deterministic test data set from signal distribution...")
    test = np.random.normal(loc=1, size=[100, 2])
    test_tensor = tf.placeholder(tf.float32, [None, 2])

    # Classifies a test point from the trained SVM parameters.
    model = svm.decide(
        input_tensor, 400, test_tensor, 100, beta, offset, kernel_type="gaussian",
        gamma=1)

    print("Test data classified as signal: %f%%" % sess.run(
        tf.reduce_sum(model), feed_dict={input_tensor: test_images,
                                         test_tensor: test_labels}))