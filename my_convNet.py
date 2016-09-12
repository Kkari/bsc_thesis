# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Reformat the dataset for the convolutional networks
def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

if __name__ == '__main__':
    depth = 32
    num_channels = 1
    batch_size = 16
    patch_size = 5
    depth = 32
    num_hidden = 64
    num_channels = 1

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # The mnist images have a dimension of 28*28.
    image_size = 28
    # There are 10 labels.
    num_labels = 10
    train_dataset = mnist.train.images
    train_labels = mnist.train.labels

    perm = np.random.permutation(mnist.test.images.shape[0])

    split_point = int(mnist.test.images.shape[0] * 0.1)
    valid_dataset, test_dataset = mnist.test.images[:split_point], mnist.test.images[split_point:]
    valid_labels, test_labels = mnist.test.labels[:split_point], mnist.test.labels[split_point:]
    train_dataset_conv = reformat(train_dataset)
    valid_dataset_conv = reformat(valid_dataset)
    test_dataset_conv = reformat(test_dataset)

    print(train_dataset_conv.shape, train_labels.shape)
    print(valid_dataset_conv.shape, valid_labels.shape)
    print(test_dataset_conv.shape, test_labels.shape)

    if tf.gfile.Exists('./convNetSummaries'):
        tf.gfile.DeleteRecursively('./convNetSummaries')
    tf.gfile.MakeDirs('./convNetSummaries')

    graph = tf.Graph()

    with graph.as_default():
        # Placeholders
        keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('input'):
            # Input data.
            tf_train_batch = tf.placeholder(tf.float32,
                                            shape=(None, image_size, image_size, num_channels),
                                            name='train_batch')

            # The None at the shape argument means that the dimension is not defined,
            tf_train_labels = tf.placeholder(tf.float32,
                                             shape=(None, num_labels),
                                             name='train_labels')

        with tf.name_scope('hidden_layer_1'):
            # Variables.
            h_conv1_weights = weight_variable([patch_size, patch_size, num_channels, depth], name='h_conv1_weights')
            variable_summaries('first_hidden_layer/weights', h_conv1_weights)

            h_conv1_biases = bias_variable([depth], name='h_conv1_biases')
            variable_summaries('first_hidden_layer/biases', h_conv1_biases)

            h_preactivation1 = conv2d(tf_train_batch, h_conv1_weights) + h_conv1_biases
            tf.histogram_summary('first_hidden_layer/pre_activations', h_preactivation1)

            # Define the model:
            # First layer, patches of 5x5 into 32 features
            h_conv1 = tf.nn.relu(conv2d(tf_train_batch, h_conv1_weights) + h_conv1_biases)
            tf.histogram_summary('first_hidden_layer/activations', h_conv1)

            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('hidden_layer_2'):
            h_conv2_weights = weight_variable([patch_size, patch_size, depth, depth * 2], name='h_conv2_weights')
            variable_summaries('second_hidden_layer/weights', h_conv2_weights)

            h_conv2_biases = bias_variable([depth * 2], name='h_conv2_biases')
            variable_summaries('second_hidden_layer/biases', h_conv2_biases)

            h_preactivation2 = conv2d(h_pool1, h_conv2_weights) + h_conv2_biases
            tf.histogram_summary('second_hidden_layer/preactivations', h_preactivation2)

            # Second layer, patches of 5x5 into 64 features
            h_conv2 = tf.nn.relu(h_preactivation2)
            tf.histogram_summary('second_hidden_layer/activations', h_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            conv_image_size = image_size // 4
            # Reshape into the densely connected layer
            h_pool2_flat = tf.reshape(h_pool2, [-1, conv_image_size * conv_image_size * depth * 2])

        with tf.name_scope('fully_connected_layer'):
            fc1_weights = weight_variable([conv_image_size * conv_image_size * depth * 2, num_hidden],
                                          name='fully_connected_layer_weights')
            variable_summaries('fully_connected/weights', fc1_weights)

            fc1_biases = bias_variable([num_hidden], name='fully_connected_layer_biases')
            variable_summaries('fully_connected/biases', fc1_biases)

            # Define the fully connected layer
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_weights) + fc1_biases)
            tf.histogram_summary('fully_connected/activations', h_fc1)

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            tf.histogram_summary('fully_connected/dropout', h_fc1_drop)

        with tf.name_scope('read_out_layer'):
            # Generate the output softmax through the fully connected weights
            output_softmax_weights = weight_variable([num_hidden, num_labels], name='output_softmax_weights')
            variable_summaries('softmax/weights', output_softmax_weights)

            output_softmax_biases = bias_variable([num_labels], name='output_softmax_biases')
            variable_summaries('softmax/biases', output_softmax_biases)

            # Readout layer
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, output_softmax_weights) + output_softmax_biases,
                                   name='output_softmax')

        with tf.name_scope('cross_entropy'):
            # Training computation.
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(y_conv),
                                                          reduction_indices=[1],
                                                          name='xentropy'),
                                           name='xentropy_mean')
            tf.scalar_summary('cross_entropy', cross_entropy)

        with tf.name_scope('optimizer'):
            # Optimizer
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy_measure'):
            # These two lines are measure the accuracy of our model.
            # y_conv is a softmax output, the highest entry is the most probable according to our model
            # (e.g.: [0.7, 0.2, 0.5, 0.5])
            # tf_train_labels are the original labels for the training set.
            # (eg.: [0, 0, 0, 1])
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_train_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter('./convNetSummaries/train', graph)
        test_writer = tf.train.SummaryWriter('./convNetSummaries/train/test')

    config = {
        # 'log_device_placement': True
    }

    with tf.Session(graph=graph, config=tf.ConfigProto(**config)) as sess:
            # Initialize the session variables.
            sess.run(tf.initialize_all_variables())

            for step in range(3001):

                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

                # I should randomize this part a bit more to reduce the possibility of reoccuring batches.
                batch_data = train_dataset_conv[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                if step % 10 == 0:
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={
                                                tf_train_batch: valid_dataset_conv,
                                                tf_train_labels: valid_labels,
                                                keep_prob: 1.0
                                            })
                    test_writer.add_summary(summary, step)
                    print('Accuracy at step %s: %s' % (step, acc))
                if step % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    summary, _ = sess.run([merged, train_step],
                                               options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={tf_train_batch:  batch_data,
                                                          tf_train_labels: batch_labels,
                                                          keep_prob: 1.0})

                    train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                    train_writer.add_summary(summary, step)
                    print('Adding run metadata for', step)

                else:
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={tf_train_batch:  batch_data,
                                                     tf_train_labels: batch_labels,
                                                     keep_prob: 0.5})
                    train_writer.add_summary(summary, step)

            test_accuracy = accuracy.eval(feed_dict={tf_train_batch: test_dataset_conv,
                                                     tf_train_labels: test_labels,
                                                     keep_prob: 1.0})
            test_writer.add_summary(summary, step)
            print("test accuracy %g" % test_accuracy)