from sklearn.datasets import make_blobs# mlp for multiclass classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from numpy import where
from numpy import argmax

# Part 1: Create 2D random dataset & display with scatter plot
X, y = make_blobs(n_samples=1400, centers=3,n_features=2, cluster_std=2.0, random_state=1)

plt.figure(1)
# Display dataset with scatter plot for each class value
for class_value in range(3):
    # select indices of points with the class label
    row_ix = where(y == class_value)
    # scatter plot for points with a different color
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
plt.xlabel(r"$x_1$"), plt.ylabel(r"$x_2$") 
plt.show() # show scatter plot

# Part 2: Preparing training dataset and testing dataset
def prepare_data():
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    n_features = X_train.shape[1]
    print('Number of feature %.1f' % n_features)
    return X_train, y_train, X_test, y_test

# Part 3: Training the MLP network & evaluate its accuracy for both Train/Test datasets
def fit_model(X_train, y_train, X_test, y_test, lrate, momentum, nodes):
    # define MLP network model
    model = Sequential()
    # Add a Hidden layer: dimensionality of the output space = 25 hidden units
    model.add(Dense(nodes, activation='relu', kernel_initializer='he_normal', input_shape=(2,)))
    # Add an output layer: ending network with a Dense layer of size 1.
    model.add(Dense(3, activation='sigmoid'))

    # compile the MLP network  ------------------------------------------------------------------------
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr=lrate, momentum=momentum)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples, in batches of 32 sample
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                           epochs=200, batch_size=32, verbose=0)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    # plot learning curves
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title('lrates='+str(lrate)+'Momentum='+str(momentum))
    plt.ylim([0.2,1.0]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')

# Main Program Step 3: (Nodes = 25) 
# prepare training and testing datasets
X_train, y_train, X_test, y_test = prepare_data()
# create learning curves for different learning rates
lrate = [1E1, 1E0, 1E-1, 1E-2]
momentum = 0.0
nodes = 25
for i in range(len(lrate)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, lrate[i],momentum, nodes)
# show learning curves
plt.show()

# Main Program Step 3: (Nodes = 50)  
# prepare training and testing datasets
X_train, y_train, X_test, y_test = prepare_data()
# create learning curves for different learning rates
lrate = [1E1, 1E0, 1E-1, 1E-2]
momentum = 0.0
nodes = 50
for i in range(len(lrate)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, lrate[i],momentum, nodes)
# show learning curves
plt.show()

# Main Program Step 4:  
# prepare training and testing datasets
X_train, y_train, X_test, y_test = prepare_data()
# create learning curves for different learning rates
lrate = 0.1
momentum = [0.0, 0.5, 0.9, 0.99]
nodes = 25
for i in range(len(momentum)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, lrate,momentum[i], nodes)
# show learning curves
plt.show()