import tensorflow as tf


def cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs):
    """Tiles two tensors in perpendicular dimensions."""
    expanded_a = tf.expand_dims(tensor_a, 1)
    expanded_b = tf.expand_dims(tensor_b, 0)
    tiled_a = tf.tile(expanded_a, tf.constant([1, b_inputs, 1]))
    tiled_b = tf.tile(expanded_b, tf.constant([a_inputs, 1, 1]))

    return [tiled_a, tiled_b]


def linear_kernel(tensor_a, a_inputs, tensor_b, b_inputs):
    """Returns the linear kernel (dot product) matrix of two matrices of vectors
    element-wise."""
    cross = cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs)

    kernel = tf.reduce_sum(tf.mul(cross[0], cross[1]), reduction_indices=2)

    return kernel


def gaussian_kernel(tensor_a, a_inputs, tensor_b, b_inputs, gamma):
    """Returns the Gaussian kernel matrix of two matrices of vectors
    element-wise."""
    cross = cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs)

    kernel = tf.exp(tf.mul(tf.reduce_sum(tf.square(
        tf.sub(cross[0], cross[1])), reduction_indices=2),
        tf.neg(tf.constant(gamma, dtype=tf.float32))))

    return kernel


def cost(train_data, train_labels, inputs, kernel_type="gaussian", C=1, gamma=1):
    """Returns the kernelised cost to be minimised."""
    beta = tf.Variable(tf.zeros([inputs, 1]))
    offset = tf.Variable(tf.zeros([1]))

    if kernel_type == "linear":
        kernel = linear_kernel(train_data, inputs, train_data, inputs)
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel(train_data, inputs, train_data, inputs, gamma)

    kernel_matmul = tf.matmul(tf.matmul(beta, kernel, transpose_a=True), beta)
    first_term = tf.reshape(tf.div(kernel_matmul, tf.constant([2.0])), [1])

    t = tf.add(tf.matmul(kernel, beta, transpose_a=True), offset)
    one_minus_yt = tf.sub(tf.ones([1]), tf.mul(train_labels, t))
    linear_loss = tf.reduce_max(tf.concat(1, [one_minus_yt, tf.zeros_like(one_minus_yt)]), reduction_indices=1)
    second_term = tf.mul(tf.constant([C], dtype=tf.float32), linear_loss)

    cost_function = tf.add(first_term, second_term)

    return beta, offset, cost_function


def decide(training, training_instances, testing, testing_instances,
           beta, offset, kernel_type="gaussian", gamma=1):
    """Tests a set of test instances."""

    if kernel_type == "linear":
        kernel = linear_kernel(
            testing, testing_instances, training, training_instances)
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel(
            testing, testing_instances, training, training_instances, gamma)

    return tf.sign(tf.add(tf.matmul(kernel, beta), offset))