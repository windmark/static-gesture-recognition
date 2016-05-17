# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn import cross_validation, datasets, neighbors, metrics
from sklearn.cross_validation import train_test_split

def train_nn(inputData, outputData):
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

  '''Train using gradient descent'''
  train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)

  '''Train using RMSProp'''
  #train_step = tf.train.RMSPropOptimizer(learnRate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')

  '''Train using AdamOptimizer'''
  #train_step = tf.train.AdamOptimizer(learnRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

  # Add accuracy checking nodes
  tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

  # Init variables
  init = tf.initialize_all_variables()

  sess = tf.Session()
  sess.run(init)

  # Run each training operation with 1000 training examples
  k=[]
  saved=0
  for i in range(350):
    #sess.run(train_step, feed_dict={x: x_train, y_: y_train})
    #result = sess.run(tf_accuracy, feed_dict={x: x_test, y_: y_test})
    sess.run(train_step, feed_dict={x: inputData, y_: outputData})
    result = sess.run(tf_accuracy, feed_dict={x: inputData, y_: outputData})
    if (i % 25 == 0):
      print("Run {},{}".format(i,result))
    k.append(result)

  # Evaluate model
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  k=np.array(k)
  print(np.where(k==k.max()))
  print("Max accuracy: {}".format(k.max()))
  print(' ')
  print('NN training Validation :: Done.\n')

def train_splitval(inputData, outputData, percentSplit = 0.7, n_neighbors = 2):

    trainData, testData, trainLabels, testLabels = cross_validation.train_test_split(inputData, 
          outputData, 
          test_size=(1.0-percentSplit), 
          random_state=0)

    # k-NearestNeighbour Classifier instance
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

    # train the model
    kNNClassifier.fit(trainData, trainLabels) 

    predictedLabels = kNNClassifier.predict(testData)
    
    #Display classifier results
    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(testLabels, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabels, predictedLabels))
    print(' ')
    print('Split Validation training :: Done.\n')


''' MAIN '''

label_vector_strings = np.loadtxt('features.txt', 
    delimiter = ', ',
    usecols = (0,),
    dtype = str)

input_vector = np.loadtxt('features.txt', 
    delimiter = ', ',
    usecols = range(1,11),
    dtype = float)

dict = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}

label_vector = []
for n in range(0,len(label_vector_strings)):
  index = dict[label_vector_strings[n]]
  label_array_representation = [0, 0, 0, 0, 0, 0, 0, 0]
  label_array_representation[index] = 1
  label_vector.append(label_array_representation)

''' TRAIN ANN '''
train_nn(input_vector, label_vector)

''' TRAIN kNN '''
train_splitval(input_vector, label_vector_strings)