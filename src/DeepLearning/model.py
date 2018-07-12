import tensorflow as tf
from DeepLearning.layers import *

def convolutionalModel(n_channels, image_dim, kernels, outputbatch_size = 30):

    x = tf.placeholder(dtype = tf.float32, shape = [None, image_dim, image_dim, n_channels], name = "x")
    y = tf.placeholder(dtype = tf.float32, shape = None, name = "y")

    # start defining layers
    conv_op1, conv1 = convLayer(nameLayer = "conv1",x = x, n_channels = n_channels, n_kernels = kernels[0], size_kernels=[5,5], strides = (1,1,1,1))
    pool1 = poolLayer(nameLayer = "pool1", x = conv1, ksize=[2,2])

    # flattened input
    new_dimension = (image_dim - 2)/2 # yo creo que se le restan dos porque en la convolucion pierdes los bordes de la imagen
    flatten = hiddenLayer(nameLayer = "hidden", x = pool1)

    tf.onehot()





