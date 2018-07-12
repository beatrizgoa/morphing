import tensorflow as tf
from DeepLearning.layers import *


# Define global variables
IMAGE_DIMENSION = None
N_CHANNELS = None
N_CLASSES = None



def convolutionalModel(x, conv_kernels, hidden_neurons):
    # x is a placeholder

    # start defining layers
    conv_op1, conv1 = convLayer(nameLayer = "conv1",x = x, n_channels = N_CHANNELS, n_kernels = conv_kernels[0], size_kernels=[5,5], strides = (1,1,1,1))
    pool1 = poolLayer(nameLayer = "pool1", x = conv1, ksize=[2,2])

    # flattened input
    new_dimension = (IMAGE_DIMENSION - 2)/2 # yo creo que se le restan dos porque en la convolucion pierdes los bordes de la imagen
    flat_input = tf.reshape(pool1, shape=[-1, new_dimension * new_dimension * conv_kernels[0]])
    hidden1 = hiddenLayer(nameLayer = "hidden", x = flat_input, out_neurons = hidden_neurons[0], in_neurons=new_dimension*new_dimension*conv_kernels[0])

    # logits
    logits = hiddenLayer(nameLayer = "logits", x = hidden1, out_neurons = N_CLASSES, in_neurons=hidden_neurons[0])

    return logits



def trainingProcess(conv_kernels, hidden_neurons):

    x = tf.placeholder(dtype = tf.float32, shape = [None, ], name = "x")
    y = tf.placeholder(dtype = tf.float32, shape = None, name = "y")

    # Load model
    logits = convolutionalModel(x, conv_kernels, hidden_neurons)
