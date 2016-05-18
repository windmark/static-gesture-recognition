# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn import cross_validation, datasets, neighbors, metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib




class Knn:
  featureFile = ''
  trainedModel = None
  n_neighbors = 5
  dict = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}

  def train(self, featureFile):
    self.featureFile = featureFile
    label_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = (0,), dtype = str)
    input_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(input_vector, label_vector)
    self.trainedModel = kNNClassifier
    return self.trainedModel

  def classify(self, featureVector):
    predictedLabel = self.trainedModel.predict(featureVector)
    return self.dict[predictedLabel[0]]

  def saveModel(self, file):
    joblib.dump(self.trainedModel, file)

  def loadModel(self, file):
    self.trainedModel = joblib.load(file)

  def validateModel(self):
    percentSplit = 0.7
    n_neighbors = 2

    label_vector = np.loadtxt(self.featureFile, delimiter = ', ', usecols = (0,), dtype = str)
    input_vector = np.loadtxt(self.featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)

    trainData, testData, trainLabels, testLabels = cross_validation.train_test_split(input_vector, 
          label_vector, test_size=(1.0-percentSplit), random_state=0)

    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    kNNClassifier.fit(trainData, trainLabels) 
    predictedLabels = kNNClassifier.predict(testData)
    
    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(testLabels, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabels, predictedLabels))
    print(' ')
    print('Split Validation training :: Done.\n')





class NeuralNetwork:
  trainedModel = None

  def train(self, featureFile):
    # Build computation graph by creating nodes for input images and target output classes
    # 10 elements input
    # 8 output classes

    # Network input 
    x = tf.placeholder(tf.float32, [None, 10])

    # Network weights
    W = tf.Variable(tf.zeros([10, 8]))

    # Network bias
    b = tf.Variable(tf.zeros([8]))

    # Regression model implementation
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # To implement cross-entropy, a new placeholder is needed to input the correct answers
    y_ = tf.placeholder(tf.float32, [None, 8])

    # Cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Train the model
    learnRate = 0.9

    #Train using gradient descent
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)