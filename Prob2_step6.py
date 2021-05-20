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
from sklearn.decomposition import PCA
import pandas as pd

# Step 1: Load the machine dataset
Colfeatures = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'Class']
df = read_csv("MachineData_Exam.csv", header=None, names=Colfeatures)
# Split data into input (X) and output (y) columns
X, y = df.values[1:35001, :-1], df.values[1:35001, -1]
print('y:',y)
# ensure all data are floating point values
X = X.astype('float32')
print('X:',X)
#y = LabelEncoder().fit_transform(y)
y = y.astype('float32')
print("y:",y)

# Step 2: Preparing training dataset for training and testing of DNN Network

def prepare_data():
    # split into train and test datasets
    x = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    n_features = X_train.shape[1]
    print('Number of feature %.1f' % n_features)
    return X_train, y_train, X_test, y_test

# Step 3: Construct Deep Neural Network (DNN) model
def fit_model(X_train, y_train, X_test, y_test, initializer, batch, name, sh):
    # define DNN network model
    model = Sequential()
    # Using Zero for weight initiliazation
    #initializer = initializers.GlorotUniform(seed=1)

    # Add an input layer: dimensionality of the output space = 18 hidden units
    model.add(Dense(24, activation='relu', kernel_initializer=initializer, input_shape=sh))
    # Add two Hidden layer: output of this layer is arrays of shape = 18 hidden unit 
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

# Main Program (Step 4):
# prepare training and testing datasets -------------------------------------------------
X_train, y_train, X_test, y_test = prepare_data()
print(X_test.shape)
initializer = initializers.he_uniform()
name = "Batch Gradient Descent"
sh = (9,)
batch = len(X_train)
plt.figure(figsize=[12.8,10])
fit_model(X_train, y_train, X_test, y_test, initializer, batch, name, sh)
    
# show learning curves
plt.savefig('step4_BGD.png')
plt.show()

# PCA: Projecting the original 9-D data into 3-D principal components --------------------------
pca = PCA(n_components=3)
x = StandardScaler().fit_transform(X)
principalComponents = pca.fit_transform(x)
#print(pca.explained_variance_ratio_)

# Displaying PrincipalComponents: just the 3 main dimensions of variation
principalDf = pd.DataFrame(data = principalComponents, columns =
                           ['principal component 1', 'principal component 2', 'principal component 3'])
# Concatenating DataFrame along axis = 1 (or column).
finalDf = pd.concat([principalDf, df[['Class']]], axis = 1)
# Plotting 3 dimensional data --------------------------------------------------------------
fig = plt.figure(1,figsize=[12.8,10])

ax = fig.add_subplot(111, projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_zlabel('Principal Component 3', fontsize = 12)
ax.set_title('Principal Component Analysis of Machine Dataset', fontsize = 12)

Classes = ['0', '1', '2', '3', '4', '5', '6']
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
for Class, color in zip(Classes,colors):
    indicesToKeep = finalDf['Class'] == Class
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color, s = 30)
ax.legend(Classes)
ax.grid()
plt.savefig('PCA.png')
plt.show()

# Main Program (Step 4 repeat):
# prepare training and testing datasets -------------------------------------------------
X = principalComponents
X = X.astype('float32')
X_train, y_train, X_test, y_test = prepare_data()
print(X_test.shape)
initializer = initializers.he_uniform()
name = ["Batch = original", "Batch = 64", "Batch = 128", "Batch = 1"]
sh = (9,3)
batch = [len(X_train), 64, 128, 1]
plt.figure(figsize=[12.8,10])
for i in range (len(batch)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, initializer, batch[i], name[i], sh)
    
# show learning curves
plt.savefig('step4_after.png')
plt.show()