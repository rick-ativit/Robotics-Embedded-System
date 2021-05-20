from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from numpy import where
from numpy import argmax
import numpy as np
from timeit import default_timer as timer


# Step 1: Load the machine dataset
df = read_csv("MachineData_Exam.csv", header=None)
# Split data into input (X) and output (y) columns
X, y = df.values[1:35001, :-1], df.values[1:35001, -1]
# ensure all data are floating point values
X = X.astype('float32')
y = y.astype('float32')

# Step 2: Preparing training dataset for training and testing of DNN Network
def prepare_data():
    # split into train and test datasets
    x = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    n_features = X_train.shape[1]
    print('Number of feature %.1f' % n_features)
    return X_train, y_train, X_test, y_test

# Step 3: Construct Deep Neural Network (DNN) model
def fit_model(X_train, y_train, X_test, y_test, initializer, batch, name):
    # define DNN network model
    model = Sequential()
    # Add an input layer: dimensionality of the output space = 24 hidden units
    model.add(Dense(24, activation='relu', kernel_initializer=initializer, input_shape=(9,)))
    # Add a Hidden layer: output of this layer is arrays of shape = 24 hidden unit 
    model.add(Dense(24, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 7.
    model.add(Dense(7, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = 0.01, momentum = 0.5)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, batch_size=batch)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title(name, pad=-40)
    plt.ylim([0.5,1.2]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')

# Main Program: Step 3 
# prepare training and testing datasets -------------------------------------------------
X_train, y_train, X_test, y_test = prepare_data()
initializer = [initializers.Zeros(), initializers.RandomUniform(seed=1), initializers.GlorotUniform(seed=1), initializers.he_uniform()]
name = ["Initializer = Zeros","Initializer = RandomUniform","Initializer = GlorotUniform","Initializer = he_uniform"]
batch = 32
plt.figure(figsize=[12.8,10])
for i in range (len(initializer)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, initializer[i], batch, name[i])
    
# show learning curves
plt.savefig('step3.png')
plt.show()

# Main Program: Step 4 
# prepare training and testing datasets -------------------------------------------------
X_train, y_train, X_test, y_test = prepare_data()
initializer = initializers.he_uniform()
name = ["Batch Gradient Descent","Mini-Batch Gradient Descent 64","Mini-Batch Gradient Descent 128","Stochastic Gradient Descent"]
batch = [len(X_train),64,128,1]
plt.figure(figsize=[12.8,10])
for i in range (len(name)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training DNN model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, initializer, batch[i], name[i])
    
# show learning curves
plt.savefig('step4.png')
plt.show()