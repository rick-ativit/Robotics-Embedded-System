from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from keras import backend
from numpy import where
from numpy import argmax
import numpy as np
from timeit import default_timer as timer
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau

# Part 0: monitor the learning rate
class LearningRateMonitor(Callback):
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()
 
    # end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
        # get and store the learning rate
        optimizer = self.model.optimizer
        lrate = float(backend.get_value(self.model.optimizer.lr))
        self.lrates.append(lrate)

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
def fit_model_lr(X_train, y_train, X_test, y_test, lrate, name):
    # define DNN network model
    model = Sequential()
    # Using he_uniform for weight initiliazation
    initializer = initializers.he_uniform()

    # Add an input layer: dimensionality of the output space = 24 hidden units
    model.add(Dense(24, activation='relu', kernel_initializer=initializer, input_shape=(9,)))
    # Add a Hidden layer: output of this layer is arrays of shape = 24 hidden unit 
    model.add(Dense(24, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 7.
    model.add(Dense(7, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = lrate, momentum = 0.5)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, batch_size=32)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title(name, pad=-40)
    plt.ylim([0.5,1.2]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')
    
def fit_model_mm(X_train, y_train, X_test, y_test, momen, name):
    # define DNN network model
    model = Sequential()
    # Using Zero for weight initiliazation
    initializer = initializers.he_uniform()

    # Add an input layer: dimensionality of the output space = 24 hidden units
    model.add(Dense(24, activation='relu', kernel_initializer=initializer, input_shape=(9,)))
    # Add a Hidden layer: output of this layer is arrays of shape = 24 hidden unit 
    model.add(Dense(24, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 3.
    model.add(Dense(7, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = 0.01, momentum = momen)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, batch_size=32)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title(name, pad=-40)
    plt.ylim([0.5,1.2]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')
    
def fit_model_lrs(X_train, y_train, X_test, y_test, patience, name):
    # define DNN network model
    model = Sequential()
    # Using Zero for weight initiliazation
    initializer = initializers.he_uniform()

    # Add an input layer: dimensionality of the output space = 24 hidden units
    model.add(Dense(24, activation='relu', kernel_initializer=initializer, input_shape=(9,)))
    # Add a Hidden layer: output of this layer is arrays of shape = 24 hidden unit 
    model.add(Dense(24, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 3.
    model.add(Dense(7, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = 0.01, momentum = 0.5)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_delta=1E-7)
    lrm = LearningRateMonitor()
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, callbacks=[rlrp, lrm])
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title(name, pad=-40)
    plt.ylim([0.5,1.2]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')
    
def fit_model_node(X_train, y_train, X_test, y_test, node, name):
    # define DNN network model
    model = Sequential()
    # Using Zero for weight initiliazation
    initializer = initializers.he_uniform()

    # Add an input layer: dimensionality of the output space = varying hidden units
    model.add(Dense(node, activation='relu', kernel_initializer=initializer, input_shape=(9,)))
    # Add a Hidden layer: output of this layer is arrays of shape = varying hidden unit 
    model.add(Dense(node, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 3.
    model.add(Dense(7, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = 0.01, momentum = 0.5)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, batch_size=32)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title(name, pad=-40)
    plt.ylim([0.5,1.2]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')

# Main Program (Learning rate):  
# prepare training and testing datasets -------------------------------------------------
print("Learning rate:")
X_train, y_train, X_test, y_test = prepare_data()
lrate = [0.01,0.05,0.1,0.5]
name = ["lr = 0.01","lr = 0.05","lr = 0.1","lr = 0.5"]
plt.figure(figsize=[12.8,10])
for i in range (len(lrate)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model_lr(X_train, y_train, X_test, y_test, lrate[i], name[i])
    
# show learning curves
plt.savefig('step5_lr.png')
#plt.show()

# Main Program (Momentum):  
# prepare training and testing datasets -------------------------------------------------
print("Momentum:")
X_train, y_train, X_test, y_test = prepare_data()
momen = [0.5,0.7,0.8,0.9]
name = ["momentum = 0.5","momentum = 0.7","momentum = 0.8","momentum = 0.9"]
plt.figure(figsize=[12.8,10])
for i in range (len(momen)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model_mm(X_train, y_train, X_test, y_test, momen[i], name[i])
    
# show learning curves
plt.savefig('step5_mm.png')
#plt.show()

# Main Program (Learning Rate Schedule):  
# prepare training and testing datasets -------------------------------------------------
print("Learning rate schedule:")
X_train, y_train, X_test, y_test = prepare_data()
patiences = [2, 5, 10, 15]
name = ["patiences = 2","patiences = 5","patiences = 10","patiences = 15"]
plt.figure(figsize=[12.8,10])
for i in range (len(patiences)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model_lrs(X_train, y_train, X_test, y_test, patiences[i], name[i])
    
# show learning curves
plt.savefig('step5_lrs.png')
#plt.show()

# Main Program (Nodes):  
# prepare training and testing datasets -------------------------------------------------
print("Nodes:")
X_train, y_train, X_test, y_test = prepare_data()
node = [18, 24, 36, 64]
name = ["node = 18","mnode = 24","node = 36","node = 64"]
plt.figure(figsize=[12.8,10])
for i in range (len(node)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model_node(X_train, y_train, X_test, y_test, node[i], name[i])
    
# show learning curves
plt.savefig('step5_node.png')
#plt.show()

