import tensorflow as tf


def convLayer(nameLayer,x, n_channels, n_kernels, size_kernels, strides = (1,1,1,1)):
	# COnvolutional layer is defined
	with tf.name_scope(nameLayer):
		w = tf.Variable(tf.truncated_normal(shape = [size_kernels, size_kernels, n_channels, n_kernels], mean = 0, stddev = 0.1, name = nameLayer))
		b = tf.Variable(tf.zeros(n_kernels))
		conv_op = tf.nn.conv2d(x, w, strides = strides, padding = 'VALID')
		activation = tf.nn.relu(conv_op) + b

		# define summaries
		tf.summary.histogram("weights", w)
		tf.summary.histogram("bias", b)
		tf.summary.histogram("activations", activation)

		return conv_op, activation


def poolLayer(nameLayer, x, ksize, strides= (1,1,1,1)):
	# max pool layer is defined
	with tf.name_scope(nameLayer):
		return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding="SAME")
		


def hiddenLayer(nameLayer, x, in_neurons, out_neurons):
	# Hidden Layer is define
	with tf.name_scope(nameLayer):
		w = tf.Variable(tf.truncated_normal([in_neurons, out_neurons], stddev = 0.1, name = "w"))
		b = tf.Variable(tf.truncated_normal(tf.zeros(out_neurons)))
		activation = tf.matmul(x ,w) + b

		tf.summary.histogram("weights", w)
		tf.summary.histogram("bias", b)
		tf.summary.histogram("activattions", activation)

	return activation



def dropoutLayer(nameLayer, x, keep_prob = 0.4):
	# DropoutLayer 
	with tf.name_scope(nameLayer):
		return tf.nn.dropout(x, keep_prob = keep_prob, name = "dropout")


