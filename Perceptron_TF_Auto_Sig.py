import numpy as np                   # import numpy module
from matplotlib import pyplot as plt # import matplotlib module
import tensorflow as tf              # import tensorflow module

def perceptron(X, T, W=None): 
    for i in range(8):
        Wt = tf.transpose(W)
        Xreshape = tf.reshape(tensor=X[i,:], shape=[2, 1])
        A = T[i]*tf.nn.sigmoid(tf.matmul(Wt, Xreshape))
        if (A <= 0):
            LReshape = tf.reshape(lr*T[i]*X[i,:], shape=[2, 1])
            W = tf.add(W, LReshape)
            #print(W)
    return W

def plot_hyperplane2d(X, T, w):
    for i in range(8):
        if (T[i] == 1):
            plt.plot(X[i, 0], X[i, 1],'og')
        else:
            plt.plot(X[i, 0], X[i, 1],'or')
        if (w[1] != 0):
            xlim = plt.gca().get_xlim()
            slope = -w[0] / w[1]
            bias = 0.0
            plt.plot(xlim, [xlim[0] * slope + bias, xlim[1] * slope + bias], 'b')
        else:
            ylim = plt.gca().get_ylim()
            plt.plot([0, 0], ylim, 'b')

lr = 0.1 # lr = learning rate
lrn = 1 # lrn = learning rate for using in filename
for i in range(9):
    filename = "fsiglr0%d.png" %lrn
    print("Filename:", filename)
    print("Learning rate:", lr)
    # Create random numbers with different seed
    GuassRandom1 = tf.random.Generator.from_seed(1)
    GuassRandom2 = tf.random.Generator.from_seed(2)
    # Generate input vector : X1 with center@[-1,-1] and X2 with center@[1, 1]
    X1 = tf.Variable(GuassRandom1.normal(shape=[4, 2], mean=0.0, stddev=0.5)) + tf.constant([-1.0, -1.0])
    X2 = tf.Variable(GuassRandom2.normal(shape=[4, 2], mean=0.0, stddev=0.5)) + tf.constant([1.0, 1.0])
    X = tf.concat(values=[X1, X2], axis=0)
    # Generate label vector for output data
    T1 = tf.ones(shape=[4,1], dtype=tf.float32)
    T2 = -1.0*tf.ones(shape=[4,1], dtype=tf.float32)
    T = tf.concat(values=[T1, T2], axis=0)
    # Initialize random weight vector : W
    W = tf.Variable(GuassRandom1.normal(shape=[2, 1]))
    print("Original: W = \n",W.numpy())
    # Perform Learning process using the Perceptron
    W = perceptron(X, T, W)
    print("After Learning: W = \n",W.numpy())
    # Plot hyperplane to examine how accurate perceptron perform learning proess
    plot_hyperplane2d(X, T, W)
    plt.savefig(filename)
    plt.show()
    lr = lr + 0.1
    lrn= lrn + 1