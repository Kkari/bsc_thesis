from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from my_rbm import Rbm

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images = mnist.train.images
    test_images = mnist.test.images

    # single_number_train_data_all = np.array(
    #     [mnist.train.images[idx] for idx, one in enumerate(mnist.train.labels) if one[2] == 1])
    #
    # split_point = int(single_number_train_data_all.shape[0] * 0.9)
    # single_number_train_data = single_number_train_data_all[:split_point]
    # single_number_validate_data = single_number_train_data_all[split_point:]
    #
    # single_number_train_labels_all = np.array(
    #     [mnist.train.labels[idx] for idx, one in enumerate(mnist.train.labels) if one[2] == 1])
    # single_number_train_labels = single_number_train_labels_all[:split_point]
    # single_number_validate_labels = single_number_train_labels_all[split_point:]

    ones_train_images = np.array([mnist.train.images[idx] for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])
    ones_train_labels = np.array([mnist.train.labels[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])

    twos_train_images = np.array([mnist.train.images[idx] for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])
    twos_train_labels = np.array([mnist.train.labels[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])

    ones_test_images = np.array([mnist.test.images[idx] for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])
    ones_test_labels = np.array([mnist.test.labels[idx, 1:3] for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])

    twos_test_images = np.array([mnist.test.images[idx] for idx, labels in enumerate(mnist.test.labels) if labels[2] == 1])
    twos_test_labels = np.array([mnist.test.labels[idx, 1:3] for idx, labels in enumerate(mnist.test.labels) if labels[2] == 1])

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

    rbm = Rbm(num_hidden=64, num_epochs=1, num_classes=10, num_features=test_images.shape[1])
    rbm.init_rbm()

    rbm.fit(train_images, test_images)
    rbm.fit_predictor(train_data=mnist.train.images, train_labels=mnist.train.labels,
                      test_data=mnist.test.images, test_labels=mnist.test.labels)
