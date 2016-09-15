import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.distributions as tf_dist
import numpy as np


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


class Rbm:
    """
    My little restricted boltzmann machine class.
    All optimizations are made using Geoffrey Hintons paper:
    "A Practical Guide to Training Restricted Boltzmann Machines"
    """
    def __init__(self,  num_features, num_hidden=250, visible_unit_type='bin', learning_rate=0.1,
                 std_dev=0.1, batch_size=10,
                 gibbs_sampling_steps=1, log_device_placement=False,
                 last_update_prob_hidden=True, weight_decay_rate=0.0001,
                 num_classes=0, name='rbm'):
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
        self.batch_size = batch_size
        self.gibbs_sampling_steps = gibbs_sampling_steps

        # Optimization parameters
        self.last_update_prob_hidden = last_update_prob_hidden
        self.weight_decay_rate = weight_decay_rate

        # Tensorflow config parameters
        self.config = {
            'log_device_placement': log_device_placement
        }

        # Tensorflow variables
        self.input_data = None
        self.h_rand = None
        self.v_rand = None
        self.W = None
        self.h_biases = None
        self.v_biases = None
        self.batch_labels = None
        self.W_prediction = None
        self.biases_prediction = None
        self.num_classes = num_classes
        self.num_features = num_features
        self.initialized = False
        self.rbm_trained = False
        self.g = None
        self.rbm_merged = None
        self.optimizer = None
        self.accuracy = None
        self.test_writer = None
        self.train_writer = None
        self.tf_session = None
        self.updates = None
        self.reconstruction_cost = None
        self.all_merged = None

        # Summaries
        self.summariesFolder = './rbm/' + name + 'Summaries'
        if tf.gfile.Exists(self.summariesFolder):
            tf.gfile.DeleteRecursively(self.summariesFolder)
        tf.gfile.MakeDirs(self.summariesFolder)

        print('Number of features: %s' % num_features)
        print('Number of classes: %s' % num_classes)

    def init_rbm(self):

        self.g = tf.Graph()

        with self.g.as_default():
            self.build_rbm()
            self.rbm_merged = tf.merge_all_summaries()

            if self.num_classes != 0:
                self.optimizer, self.accuracy = self.build_predictive_layer()
                self.all_merged = tf.merge_all_summaries()

            self.tf_session = tf.Session(graph=self.g, config=tf.ConfigProto(**self.config))
            self.saver = tf.train.Saver()

        self.train_writer = tf.train.SummaryWriter(self.summariesFolder + '/train', self.g)
        self.test_writer = tf.train.SummaryWriter(self.summariesFolder + '/test')

        self.initialized = True

    @staticmethod
    def weight_variable(shape, name, initial=None):
        if not initial:
            initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name, initial=None):
        if not initial:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    # # Initialize the bias of the visible unit i to log(p_i/(1-p_i)) where p_i is the proportion of
    # # training vectors in which unit i is on.
    #
    # init_visible_bias = []
    # for i in range(train_dataset.shape[1]):
    #     p_i = np.count_nonzero(train_dataset[:, i]) / float(train_dataset.shape[0])
    #     init_visible_bias.append(np.log(p_i / (1 - p_i)))

    def free_energy(self, v_sample):
        wx_b = tf.matmul(v_sample, self.W) + self.h_biases
        print('sample:', v_sample.get_shape())
        print('biases:', self.v_biases.get_shape())
        v_bias_term = tf.matmul(self.input_data, tf.reshape(self.v_biases, [self.num_features, -1]))
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), reduction_indices=1)
        return v_bias_term - hidden_term

    def build_rbm(self, visible_bias_initial=None):
        with tf.name_scope('input'):
            # Create Placeholder values. ------------------------------------------------------------
            self.input_data = tf.placeholder(tf.float32, shape=[None, self.num_features], name='batch_data')

            # This variable will hold a random distribution in every trainings step,
            # it is used to decide the binary state of the hidden variables in the gibbs
            # sampling step.
            self.h_rand = tf.placeholder(tf.float32, shape=[None, self.num_hidden], name='hidden_random')
            self.v_rand = tf.placeholder(tf.float32, shape=[None, self.num_features], name='hidden_random')

        # Initialize the weights and the biases. -------------------------------------------------
        with tf.name_scope('rbm'):
            with tf.name_scope('rbm_weight_layer'):
                self.W = Rbm.weight_variable([self.num_features, self.num_hidden], name='rbm_weights')
                variable_summaries('rbm_weights', self.W)

                self.h_biases = Rbm.bias_variable([self.num_hidden],
                                                  name='rbm_hidden_biases')
                variable_summaries('rbm_hidden_biases', self.h_biases)

            with tf.name_scope('rbm_visible_biases'):
                self.v_biases = Rbm.bias_variable([self.num_features],
                                                  initial=visible_bias_initial,
                                                  name='rbm_visible_biases')
                variable_summaries('rbm_visible_biases', self.v_biases)

            # Build the model ------------------------------------------------------------------------
            # Perform the first gibbs sampling step.

            with tf.name_scope('first_gibbs_sampling_step'):
                last_update = self.gibbs_sampling_steps == 1
                h_prob0, h_state0, v_prob, v_state, h_prob1, h_state1 = self.gibbs_sampling_step(
                    self.input_data,
                    self.num_features,
                    last_update)

            with tf.name_scope('positive_phase'):
                # Compute the Positive associations. I.e. the when the visible units are
                # clamped to the input.
                # TODO: Or vprob and hprob0???
                positive = tf.matmul(tf.transpose(self.input_data), h_state0, name='positive_phase_bin')
                tf.histogram_summary('rbm/positive_phase_bin', positive)

            with tf.name_scope('gibbs_steps'):
                # Initiate the free running gibbs sampling part.
                nn_input = v_prob
                for step in range(self.gibbs_sampling_steps - 1):
                    last_update = step == (self.gibbs_sampling_steps - 1)
                    h_prob0, h_state0, v_prob, v_state, h_prob1, h_state1 = self.gibbs_sampling_step(
                        nn_input,
                        self.num_features,
                        last_update)
                    nn_input = v_prob

            with tf.name_scope('negative_phase'):
                # Compute the negative phase, when we reconstruct the visible nodes form
                # the hidden ones.
                # TODO: Or h_prob1?
                negative = tf.matmul(tf.transpose(v_prob), h_state1, name='negative_phase')
                tf.histogram_summary('rbm/negative_phase_gauss', negative)

                cost = tf.reduce_mean(self.free_energy(self.input_data)) - tf.reduce_mean(self.free_energy(v_state))
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost, var_list=[
                                                                                                 self.W,
                                                                                                 self.h_biases,
                                                                                                 self.v_biases
                                                                                                 ])

                self.updates = optimizer
            with tf.name_scope('reconstruction_cost_function'):
                # Create a mean square cost function node.
                self.reconstruction_cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - v_prob)), name='reconstruction_cost')
                tf.scalar_summary('reconstruction_error', self.reconstruction_cost)

    def build_predictive_layer(self):
        with tf.name_scope('softmax_layer'):
            with tf.name_scope('input'):
                self.batch_labels = tf.placeholder(tf.float32, [None, self.num_classes],
                                                   name='train_batch_labels')

            with tf.name_scope('sample_rbm_hidden_units'):
                # Create a softmax for class prediction.
                h_prob_predict = tf.nn.sigmoid(tf.matmul(self.input_data, self.W) + self.h_biases)

            with tf.name_scope('prediction_weights'):
                self.W_prediction = Rbm.weight_variable([self.num_hidden, self.num_classes], name='prediction_weights')
                #self.W_prediction = Rbm.weight_variable([784, self.num_classes], name='prediction_weights')
                tf.histogram_summary('rbm/prediction_weights', self.W_prediction)
                self.biases_prediction = Rbm.bias_variable([self.num_classes], name='prediction_biases')
                tf.histogram_summary('rbm/prediction_biases', self.biases_prediction)

            with tf.name_scope('softmax'):
                logits = tf.matmul(h_prob_predict, self.W_prediction) + self.biases_prediction
                #logits = tf.matmul(self.input_data, self.W_prediction) + self.biases_prediction
                self.prediction_y = tf.nn.softmax(logits, name='prediction_softmax')

        with tf.name_scope('prediction_optimizer'):
            with tf.name_scope('cross_entropy_predictor'):
                print('logit shape: ', logits.get_shape())
                print('batch_labels shape: ', self.batch_labels.get_shape())
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.batch_labels,
                                                                        name='xentropy')
                with tf.name_scope('xentropy_mean'):
                    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                tf.scalar_summary('xentropy_mean', loss)

            with tf.name_scope('train_predictor'):
                optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=[self.W_prediction,
                                                                                              self.biases_prediction],
                                                                              name='Prediction_Optimizer')

        with tf.name_scope('accuracy'):
            # These two lines are measure the accuracy of our model.
            # y_conv is a softmax output, the highest entry is the most probable according to our model
            # (e.g.: [0.7, 0.2, 0.5, 0.5])
            # tf_train_labels are the original labels for the training set.
            # (eg.: [0, 0, 0, 1])
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.prediction_y, 1), tf.argmax(self.batch_labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return optimizer, accuracy

    def fit(self, train_dataset, validation_dataset, num_epochs=10):
        if not self.initialized:
            raise AssertionError('network is not built yet!')

        with self.g.as_default():
            # Train the model. -----------------------------------------------------------------------
            sess = self.tf_session
            sess.run(tf.initialize_all_variables())

            print("Train set dimensions: (%s, %s)" % train_dataset.shape)
            for i in range(num_epochs):
                print("epoch: %s" % i)
                permutation = np.random.permutation(train_dataset.shape[0])
                train_dataset_permuted = train_dataset[permutation]

                batches = np.array_split(train_dataset_permuted, train_dataset.shape[0] // self.batch_size)
                # print('Batches shape: %s' % len(batches))
                for j, batch in enumerate(batches):
                    # print('batch shape: ', batch.shape)
                    #                     if j % 100 == 0:
                    #                         print("batch_number: %s" % j)
                    sess.run(
                        self.updates,
                        feed_dict={
                            self.input_data: batch,
                            self.h_rand: np.random.rand(batch.shape[0],
                                                        self.num_hidden),
                            self.v_rand: np.random.rand(batch.shape[0], batch.shape[1])
                        }
                    )

                    if j % 10 == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        rec_loss, summary = sess.run(
                            [self.reconstruction_cost, self.rbm_merged],
                            run_metadata=run_metadata,
                            options=run_options,
                            feed_dict={
                                self.input_data: validation_dataset,
                                self.h_rand: np.random.rand(validation_dataset.shape[0],
                                                            self.num_hidden),
                                self.v_rand: np.random.rand(validation_dataset.shape[0], validation_dataset.shape[1])
                            })
                        self.train_writer.add_run_metadata(run_metadata, 'step_ep%d_step%d' % (i, j))
                        self.train_writer.add_summary(summary, i)

                    if j % 100 == 0:
                        print('Reconstruction loss at epoch %s step %s: %s' % (i, j, rec_loss))
            self.saver.save(sess, './rbmModelTrained')

    def dream(self, step_number=10):
        nn_input = self.v_rand

        for step in range(step_number):
            last_update = step == (step_number - 1)
            h_prob0, h_state0, v_prob, v_state, hprob1, hstate1 = self.gibbs_sampling_step(
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

    def fit_predictor(self, train_data, train_labels, test_data, test_labels, num_steps=3000):

        with self.g.as_default():
            if not self.initialized:
                raise AssertionError('network is not built yet!')

            session = self.tf_session

            session.run(tf.initialize_variables([self.W_prediction, self.biases_prediction]))
            print("Initialized fit predictor.")
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)

                # Generate a minibatch.
                batch_data = train_data[offset:(offset + self.batch_size), :]
                batch_labels = train_labels[offset:(offset + self.batch_size), :]

                if step % 10 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    train_accuracy, summary = session.run([self.accuracy, self.all_merged],
                                                          run_metadata=run_metadata,
                                                          options=run_options,
                                                          feed_dict={self.input_data:  batch_data,
                                                                     self.batch_labels: batch_labels,
                                                                     self.h_rand: np.ones([1, self.num_hidden]),     # We just feed it because we have to.
                                                                     self.v_rand: np.random.rand(
                                                                         batch_data.shape[0],
                                                                         batch_data.shape[1])
                                                                     })
                    self.train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                    self.train_writer.add_summary(summary, step)
                    print('Adding run metadata for', step)
                    print("step %d, training accuracy %g" % (step, train_accuracy))

                session.run(self.optimizer,
                            feed_dict={self.input_data: batch_data,
                                       self.batch_labels: batch_labels,
                                       self.h_rand: np.random.rand(batch_data.shape[0],
                                                                   self.num_hidden)
                                       })

            print("test accuracy %g" % session.run(self.accuracy,
                                                   feed_dict={self.input_data: test_data,
                                                              self.batch_labels: test_labels,
                                                              self.h_rand: np.random.rand(test_data.shape[0],
                                                                                          self.num_hidden),
                                                              self.v_rand: np.random.rand(
                                                                  test_data.shape[0],
                                                                  test_data.shape[1])
                                                              }))

    def predict(self, data):
        predicted_value = self.tf_session.run(self.prediction_y,
                                              feed_dict={self.input_data: data,
                                                         self.batch_labels: np.ones([1, self.num_classes]),
                                                         self.h_rand: np.ones([1, self.num_hidden])
                                                         })
        return predicted_value

    def gibbs_sampling_step(self, visible, num_features, last_update=False, step='1'):
        with tf.name_scope('gibbs_sampling_step_round_' + step):
            with tf.name_scope('first_hidden_sampling'):
                # Sample hidden from visible.
                h_prob0 = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.h_biases, name='h_prob0_step_' + step)
                h_state0 = tf.nn.relu(tf.sign(h_prob0 - self.h_rand))

            with tf.name_scope('sample_visible_from_hidden'):
                # Sample visible from hidden
                visible_activation = tf.matmul(h_prob0, tf.transpose(self.W)) + self.v_biases
                v_probs = tf.nn.sigmoid(visible_activation, name='v_prob_bin_step_' + step)
                v_state = tf.nn.relu(tf.sign(v_probs - self.v_rand))

            with tf.name_scope('second_hidden_sampling'):
                # Sample hidden from visible.
                h_prob1 = tf.nn.sigmoid(tf.matmul(v_probs, self.W) + self.h_biases, name='h_prob1_step_' + step)

                if last_update and self.last_update_prob_hidden:
                    h_state1 = h_prob1
                else:
                    h_state1 = tf.nn.relu(tf.sign(h_prob1 - self.h_rand))

        return h_prob0, h_state0, v_probs, v_state, h_prob1, h_state1

    def get_model_parameters(self):
        return {
            'W': self.tf_session.run(self.W),
            'h_biases': self.tf_session.run(self.h_biases),
            'v_biases': self.tf_session.run(self.v_biases)
        }

    def __del__(self):
        self.tf_session.close()
