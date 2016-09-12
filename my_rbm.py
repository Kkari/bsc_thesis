import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def variable_summaries(var, name):
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


class Rbm:
    """
    My little restricted boltzmann machine class.
    All optimizations are made using Geoffrey Hintons paper:
    "A Practical Guide to Training Restricted Boltzmann Machines"
    """
    def __init__(self, num_hidden=250, visible_unit_type='bin', learning_rate=0.01,
                 std_dev=0.1, num_epochs=10, batch_size=10,
                 gibbs_sampling_steps=1, log_device_placement=False,
                 last_update_prob_hidden=True, weight_decay_rate=0.0001):
        """

        Optimization parameters:
        @param: last_update_prob: Whether the last update use the hidden layer probability as it's state.
                                  it tends to reduce the noise in the training.
        """
        # Model parameters
        self.num_hidden = num_hidden

        if visible_unit_type not in ['gauss', 'bin']:
            raise ValueError('Invalid node type.')

        self.visible_unit_type = visible_unit_type

        # Statistical model parameters
        self.stddev = std_dev
        self.learning_rate = learning_rate

        # Iteration parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gibbs_sampling_steps = gibbs_sampling_steps

        # Optimization parameters
        self.last_update_prob_hidden = last_update_prob_hidden
        self.weight_decay_rate = weight_decay_rate

        # Tensorflow config parameters
        config = {
            'log_device_placement': log_device_placement
        }

        self.g = tf.Graph()
        self.tf_session = tf.Session(graph=self.g, config=tf.ConfigProto(**config))

        # Tensorflow variables
        self.num_features = None
        self.batch = None
        self.h_rand = None
        self.v_rand = None
        self.W = None
        self.h_biases = None
        self.v_biases = None
        self.batch_labels = None
        self.W_prediction = None
        self.biases_prediction = None

    @staticmethod
    def weight_variable(shape, name, initial=None):
        if not initial:
            initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name, initial=None):
        if not initial:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def fit(self, train_dataset, validation_dataset):

        # Define the number of features. (I.e..: The number of columns in the training data)
        self.num_features = train_dataset.shape[1]

        with self.g.as_default():

            with tf.name_scope('input'):
                # Create Placeholder values. ------------------------------------------------------------
                self.batch = tf.placeholder(tf.float32, shape=(None, self.num_features), name='batch_data')

                # This variable will hold a random distribution in every trainings step,
                # it is used to decide the binary state of the hidden variables in the gibbs
                # sampling step.
                self.h_rand = tf.placeholder(tf.float32, [None, self.num_hidden], name='hidden_random')

                # Same for the visible variables.
                self.v_rand = tf.placeholder(tf.float32, [None, self.num_features], name='visible_random')

            # Initialize the weights and the biases. -------------------------------------------------

            # Initialize the bias of the visible unit i to log(p_i/(1-p_i)) where p_i is the proportion of
            # training vectors in which unit i is on.

            init_visible_bias = []
            for i in range(train_dataset.shape[1]):
                p_i = np.count_nonzero(train_dataset[:, i]) / float(train_dataset.shape[0])
                init_visible_bias.append(np.log(p_i/(1-p_i)))

            with tf.name_scope('rbm_variables'):
                with tf.name_scope('rbm_weight_layer'):
                    self.W = Rbm.weight_variable([self.num_features, self.num_hidden],
                                                 name='rbm_weights')
                    variable_summaries(self.W, 'rbm_weights')

                    self.h_biases = Rbm.bias_variable([self.num_hidden],
                                                      name='rbm_hidden_biases')
                    variable_summaries(self.h_biases, 'rbm_weights')

                with tf.name_scope('rbm_visible_biases'):
                    self.v_biases = Rbm.bias_variable([self.num_features],
                                                      initial=init_visible_bias,
                                                      name='rbm_visible_biases')
                    variable_summaries(self.h_biases, 'rbm_weights')

            # Build the model ------------------------------------------------------------------------
            # Perform the first gibbs sampling step.
            last_update = self.gibbs_sampling_steps == 1
            h_prob0, h_state0, v_prob, h_prob1, h_state1 = self.gibbs_sampling_step(
                self.batch,
                self.num_features,
                last_update)

            with tf.name_scope('positive_phase'):
                # Compute the Positive associations. I.e. the when the visible units are
                # clamped to the input.
                if self.visible_unit_type == 'bin':
                    positive = tf.matmul(tf.transpose(v_prob), h_prob0)

                elif self.visible_unit_type == 'gauss':
                    positive = tf.matmul(tf.transpose(self.batch), h_prob0)
                else:
                    raise ValueError('Invalid node type.')

            with tf.name_scope('gibbs_steps'):
                # Initiate the free running gibbs sampling part.
                nn_input = v_prob
                for step in range(self.gibbs_sampling_steps - 1):
                    last_update = step == (self.gibbs_sampling_steps - 1)
                    h_prob0, h_state0, v_prob, h_prob1, h_state1 = self.gibbs_sampling_step(
                        nn_input,
                        self.num_features,
                        last_update)
                    nn_input = v_prob

            with tf.name_scope('negative_phase'):
                # Compute the negative phase, when we reconstruct the visible nodes form
                # the hidden ones.
                negative = tf.matmul(tf.transpose(v_prob), h_prob1)

            with tf.name_scope('Weight_updaters'):
                # Define the update parameters.
                w_upd8 = self.W.assign_add(
                    self.learning_rate * (positive - negative) / self.batch_size) + self.weight_decay_rate * tf.nn.l2_loss(self.W)

                h_bias_upd8 = self.h_biases.assign_add(self.learning_rate *
                                                       tf.reduce_mean(h_prob0 - h_prob1, 0))

                v_bias_upd8 = self.v_biases.assign_add(self.learning_rate *
                                                       tf.reduce_mean(self.batch - v_prob, 0))

            updates = [w_upd8, v_bias_upd8, h_bias_upd8]

            with tf.name_scope('reconstruction_cost_function'):
                # Create a mean square cost function node.
                cost = tf.sqrt(tf.reduce_mean(tf.square(self.batch - v_prob)))
                tf.scalar_summary('reconstruction_error', cost)

            # Train the model. -----------------------------------------------------------------------
            sess = self.tf_session

            sess.run(tf.initialize_all_variables())

            print("Train set dimensions: (%s, %s)" % train_dataset.shape)
            for i in range(self.num_epochs):
                print("epoch: %s" % i)
                np.random.shuffle(train_dataset)
                batches = np.array_split(train_dataset, train_dataset.shape[0] // self.batch_size)

                for j, batch in enumerate(batches):
                    #                     if j % 100 == 0:
                    #                         print("batch_number: %s" % j)
                    sess.run(
                        updates,
                        feed_dict={
                            self.batch: batch,
                            self.h_rand: np.random.rand(batch.shape[0],
                                                        self.num_hidden),
                            self.v_rand: np.random.rand(batch.shape[0],
                                                        batch.shape[1])
                        }
                    )
                rec_loss = sess.run(
                    cost,
                    feed_dict={
                        self.batch: validation_dataset,
                        self.h_rand: np.random.rand(validation_dataset.shape[0],
                                                    self.num_hidden),
                        self.v_rand: np.random.rand(validation_dataset.shape[0],
                                                    validation_dataset.shape[1])
                    })
                print('Reconstruction loss at epoch %s: %s' % (i, rec_loss))

    def dream(self, step_number=10):
        nn_input = self.v_rand

        for step in range(step_number):
            last_update = step == (step_number - 1)
            h_prob0, h_state0, v_prob, hprob1, hstate1 = self.gibbs_sampling_step(
                nn_input,
                self.num_features,
                last_update)
            nn_input = v_prob

        sess = self.tf_session
        result = sess.run(nn_input,
                          feed_dict={
                              self.v_rand: [np.random.rand(self.num_features)]
                          })
        return result

    def fit_predictor(self, train_data, train_labels, test_data, test_labels, validate_data=None, validate_labels=None):
        num_classes = train_labels.shape[1]
        print("number of classes: %s:" % num_classes)
        with self.g.as_default():
            with tf.name_scope('input'):
                self.batch_labels = tf.placeholder(tf.float32, [None, train_labels.shape[1]], name='train_batch_labels')

            with tf.name_scope('sample_rbm_hidden_units'):
                # Create a softmax for class prediction.
                h_prob_predict = tf.nn.sigmoid(tf.matmul(self.batch, self.W) + self.h_biases)

            with tf.name_scope('prediction_weights'):
                self.W_prediction = Rbm.weight_variable([self.num_hidden, num_classes], name='prediction_weights')
                self.biases_prediction = Rbm.bias_variable([num_classes], name='prediction_biases')

            with tf.name_scope('softmax'):
                logits = tf.matmul(h_prob_predict, self.W_prediction) + self.biases_prediction
                y = tf.nn.softmax(logits)

            with tf.name_scope('cross_entropy_predictor'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.batch_labels, name='xentropy')
                with tf.name_scope('total_loss_predictor'):
                    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

            with tf.name_scope('train_predictor'):
                optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=[self.W_prediction, self.biases_prediction])

        with tf.name_scope('accuracy'):
            # These two lines are measure the accuracy of our model.
            # y_conv is a softmax output, the highest entry is the most probable according to our model
            # (e.g.: [0.7, 0.2, 0.5, 0.5])
            # tf_train_labels are the original labels for the training set.
            # (eg.: [0, 0, 0, 1])
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.batch_labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_steps = 3001
        session = self.tf_session

        session.run(tf.initialize_variables([self.W_prediction, self.biases_prediction]))
        print("Initialized fit predictor.")
        for step in range(num_steps):
            index = int(np.random.uniform(0.0, train_data.shape[0] - self.batch_size))
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)

            # Generate a minibatch.
            batch_data = train_data[offset:(offset + self.batch_size), :]
            batch_labels = train_labels[offset:(offset + self.batch_size), :]

            if step % 100 == 0:
                train_accuracy = session.run(accuracy,
                                             feed_dict={self.batch:  batch_data,
                                                        self.batch_labels: batch_labels,
                                                        self.h_rand: np.ones([1, self.num_hidden]),     # We just feed it because we have to.
                                                        self.v_rand: np.ones([1, self.num_features])    # We just feed it because we have to.
                                                        })

                print("step %d, training accuracy %g" % (step, train_accuracy))

            session.run(optimizer,
                        feed_dict={self.batch: batch_data,
                                   self.batch_labels: batch_labels,
                                   self.h_rand: np.random.rand(batch_data.shape[0],
                                                               self.num_hidden),
                                   self.v_rand: np.ones([1, self.num_features])     # We just feed it because we have to.
                                   })

        print("test accuracy %g" % session.run(accuracy,
                                               feed_dict={self.batch: test_data,
                                                          self.batch_labels: test_labels,
                                                          self.h_rand: np.random.rand(test_data.shape[0],
                                                                                      self.num_hidden),
                                                          self.v_rand: np.ones([1, self.num_features])
                                                          }))

    def gibbs_sampling_step(self, visible, num_features, last_update=False):
        # Sample hidden from visible.
        h_prob0 = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.h_biases)
        h_state0 = tf.nn.relu(tf.sign(h_prob0 - self.h_rand))

        # Sample visible from hidden
        visible_activation = tf.matmul(h_prob0, tf.transpose(self.W)) + self.v_biases

        if self.visible_unit_type == 'bin':
            v_probs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            v_probs = tf.truncated_normal((1, num_features), mean=visible_activation, stddev=self.stddev)
        else:
            raise ValueError('Invalid node type.')

        # Sample hidden from visible.
        h_prob1 = tf.nn.sigmoid(tf.matmul(v_probs, self.W) + self.h_biases)

        if last_update and self.last_update_prob_hidden:
            h_state1 = h_prob1
        else:
            h_state1 = tf.nn.relu(tf.sign(h_prob1 - self.h_rand))

        return h_prob0, h_state0, v_probs, h_prob1, h_state1

    def get_model_parameters(self):
        return {
            'W': self.tf_session.run(self.W),
            'h_biases': self.tf_session.run(self.h_biases),
            'v_biases': self.tf_session.run(self.v_biases)
        }

    def __del__(self):
        self.tf_session.close()


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

    ones_train_images = np.array([mnist.train.images[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])
    ones_train_labels = np.array([mnist.train.labels[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[1] == 1])

    twos_train_images = np.array([mnist.train.images[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])
    twos_train_labels = np.array([mnist.train.labels[idx, 1:3] for idx, labels in enumerate(mnist.train.labels) if labels[2] == 1])

    ones_test_images = np.array([mnist.test.images[idx, 1:3] for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])
    ones_test_labels = np.array([mnist.test.labels[idx, 1:3] for idx, labels in enumerate(mnist.test.labels) if labels[1] == 1])

    twos_test_images = np.array([mnist.test.images[idx, 1:3] for idx, labels in enumerate(mnist.test.labels) if labels[2] == 1])
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

    rbm = Rbm(num_hidden=64, num_epochs=4)

    rbm.fit(train_images, test_images)
    rbm.fit_predictor(train_data=train_images, train_labels=train_labels,
                      test_data=test_images, test_labels=test_labels)
