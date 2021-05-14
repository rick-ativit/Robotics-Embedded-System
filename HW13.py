from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timeit import default_timer as timer

# Part 1: Load Breast Cancer Dataset ----------------------------------------------------------
breast = load_breast_cancer()
# This data has 569 samples with thirty features, and each sample has a label associated with
breast_data = breast.data # To fetch the data, calling .data
print("Data dimension: ",breast_data.shape)
# Assign original data into input (X)
X = breast_data
X = X.astype('float32') # ensure all data are floating point values
breast_labels = breast.target # To fetch the labels, calling .target
print("Target dimension: ",breast_labels.shape)
y = breast_labels # assign output (y) for class label

# Part 2: Construct Breast Cancer Data Table --------------------------------------------------
# Reshaping the breast_labels to concatenate it with the breast_data
# so that you can finally create a DataFrame which will have both the data and labels.
labels = np.reshape(breast_labels,(569,1))
# After reshaping labels, we can concatenate data and class labels along the second axis,
final_breast_data = np.concatenate([breast_data,labels],axis=1)
# Creating the DataFrame of the final data to represent the data in a tabular fashion.
breast_dataset = pd.DataFrame(final_breast_data)
# print the features that are there in the breast cancer dataset!
features = breast.feature_names
print("features = ", features)
# Adding the label field to feature arrays
features_labels = np.append(features,'label')

breast_dataset.columns = features_labels
# Since the original labels are in 0,1 format, you will change the labels to
# benign and malignant using ".replace" function. You will use "inplace=True"
# which will modify the dataframe breastdataset.
breast_dataset['label'].replace(0, 'Benign', inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
print("\n",breast_dataset.head())

# Part 3: Use StandardScaler to standardize the dataset*s features onto unit scale ------------
# (mean = 0 and variance = 1) to optimal performance of machine learning algorithms
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
print(np.mean(x),np.std(x))

feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
print(normalised_breast.tail())

# Step 3: Preparing training dataset and testing dataset
def prepare_data():
    # Using first 66% of sample for training dataset
    n_train = 376 # 66% of 569
    X_train, X_test = X[:n_train, :], X[n_train:, :]
    # Using remaining 33% for testing dataset
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test

# Step 4: Construct Deep Neural Network (DNN) model
def fit_model(X_train, y_train, X_test, y_test, lrate, col):
    # define DNN network model
    model = Sequential()
    # Using GlorotUniform for weight initiliazation
    initializer = initializers.GlorotUniform(seed=1)

    # Add an input layer: dimensionality of the output space = 10 hidden units
    model.add(Dense(10, activation='relu', kernel_initializer=initializer, input_shape=(376,col)))
    # Add two Hidden layer: output of this layer is arrays of shape = 45 hidden unit 
    model.add(Dense(45, activation='relu', kernel_initializer=initializer))
    model.add(Dense(45, activation='relu', kernel_initializer=initializer))
    # Add an output layer: ending network with a Dense layer of size 3.
    model.add(Dense(3, activation='softmax'))

    # compile the DNN network  ------------------------------------------------------------------------
    Tstart = timer()
    # The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels
    opt = SGD(lr = lrate, momentum = 0.9)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the MLP network ------------------------------------------------------------------------------
    # Train the model for 150 epochs or iterations over all the samples
    historyFit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=0, batch_size=35)
    # evaluate the model performance-----------------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    Tend = timer(); deltaT = Tend - Tstart
    print('Training deltaT = ',deltaT,' sec') 
    plt.plot(historyFit.history['accuracy'], label='train', color='b')
    plt.plot(historyFit.history['val_accuracy'], label='test', color='r')
    plt.title('Learning Rate = '+str(lrate), pad=-40)
    plt.ylim([0.2,1.1]), plt.xlabel('Epochs'), plt.ylabel('Accuracy')

# Main Program:  
# prepare training and testing datasets -------------------------------------------------
col = 30
X_train, y_train, X_test, y_test = prepare_data()
# create learning curves for different learning rates
learning_rates = [0.5, 0.1, 0.075, 0.05]
plt.figure(figsize=[12.8,10])
for i in range(len(learning_rates)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, learning_rates[i], col)
# show learning curves
plt.savefig('MSGD.png')
plt.show()

# Step 5: Projecting the original 30-D data into 2-D principal components --------------------------
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
#print(pca.explained_variance_ratio_)

# Displaying PrincipalComponents: just the 2 main dimensions of variation
principalDf = pd.DataFrame(data = principalComponents, columns =
                           ['principal component 1', 'principal component 2'])
# Concatenating DataFrame along axis = 1 (or column).
finalDf = pd.concat([principalDf, breast_dataset[['label']]], axis = 1)
print(finalDf)
# Part 4: Plotting 2 dimensional data --------------------------------------------------------------
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_title('Principal Component Analysis of Breast Cancer Dataset', fontsize = 12)

labels = ['Benign', 'Malignant']
colors = ['r', 'g']
for label, color in zip(labels,colors):
    indicesToKeep = finalDf['label'] == label
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color, s = 30)
ax.legend(labels)
plt.show()

# Main Program:  
# prepare training and testing datasets -------------------------------------------------
print('X:',X)
X = principalComponents
X = X.astype('float32')
col = 2
print('finalDf:',X)
X_train, y_train, X_test, y_test = prepare_data()
# create learning curves for different learning rates
learning_rates = [0.5, 0.1, 0.075, 0.05]
plt.figure(figsize=[12.8,10])
for i in range(len(learning_rates)):
    # assign different plot number
    plot_no = (i+1)
    plt.subplot(2,2,plot_no)
    # Training MLP model and plot learning curves for a learning rate
    fit_model(X_train, y_train, X_test, y_test, learning_rates[i], col)
# show learning curves
plt.savefig('PCA_MSGD.png')
plt.show()