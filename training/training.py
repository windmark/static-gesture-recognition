# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn import cross_validation, datasets, neighbors, metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib



dict = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}

''' KNN CLASS '''
class Knn:
  featureFile = ''
  trainedModel = None
  n_neighbors = 2

  def __loadData__(self, featureFile):
    label_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = (0,), dtype = str)
    input_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)
    return (label_vector, input_vector)

  def train(self, featureFile):
    self.featureFile = featureFile
    (label_vector, input_vector) = self.__loadData__(featureFile)

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(input_vector, label_vector)
    self.trainedModel = kNNClassifier
    return self.trainedModel

  def classify(self, featureVector):
    predicted = self.trainedModel.predict(np.array(featureVector).reshape(1,-1))
    label = dict[predicted[0]]
    return label

  def saveModel(self, file):
    joblib.dump(self.trainedModel, file)

  def loadModel(self, file):
    self.trainedModel = joblib.load(file)

  def externalValidateModel(self, separateFeatureFile):
    (label_vector, input_vector) = self.__loadData__(self.featureFile)    
    (test_label_vector, test_input_vector) = self.__loadData__(separateFeatureFile)

    predictedLabels = self.trainedModel.predict(test_input_vector)

    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(test_label_vector, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label_vector, predictedLabels))
    print('Split Validation training :: Done.\n')



  def splitValidateModel(self):
    percentSplit = 0.7

    (label_vector, input_vector) = self.__loadData__(self.featureFile)

    trainData, testData, trainLabels, testLabels = cross_validation.train_test_split(input_vector, 
          label_vector, test_size=(1.0-percentSplit))

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(trainData, trainLabels) 
    predictedLabels = kNNClassifier.predict(testData)
    
    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(testLabels, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabels, predictedLabels))
    print('Split Validation training :: Done.\n')

  def crossValidateModel(self):
    (label_vector, input_vector) = self.__loadData__(self.featureFile)
    kFold = 5

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    scores = cross_validation.cross_val_score(kNNClassifier, input_vector, label_vector, cv = kFold)
    
    print("\n----- k-fold Cross Validation -----")
    print(scores)
    print("Average: ", sum(scores) / len(scores))



''' NEURAL NETWORK '''
class NeuralNetwork:

  trainedModel = None
  featureFile = ''
  dict = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}

  def __loadData__(self, featureFile):
    label_vector_strings = np.loadtxt(featureFile, delimiter = ', ', usecols = (0,), dtype = str)
    input_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)

    label_vector = []
    for n in range(0,len(label_vector_strings)):
      index = self.dict[label_vector_strings[n]]
      label_array_representation = [0, 0, 0, 0, 0, 0, 0, 0]
      label_array_representation[index] = 1
      label_vector.append(label_array_representation)
    return (label_vector, input_vector)

  def saveModel(self, file):
    saver = tf.train.Saver()
    save_path = saver.save(self.trainedModel, ("%s.ckpt" % file))
    print("Model saved in file: %s" % save_path)

  def loadModel(self, file):
    saver = tf.train.Saver()
    saver.restore(self.trainedModel, ("%s.ckpt" % file))
    print("Model restored.")

  def validateModel(self):
    (label_vector, input_vector) = self.__loadData__(self.featureFile)

  def trainMLP(self, featureFile):
    (label_vector, input_vector) = self.__loadData__(featureFile)

    percent_split = 0.7
    trX, teX, trY, teY = cross_validation.train_test_split(input_vector, 
              label_vector, test_size=(1.0-percent_split), random_state=0)

    n_inputs = 10
    n_outputs = 8

    X = tf.placeholder("float", [None, n_inputs])
    Y = tf.placeholder("float", [None, n_outputs])

    w_h = tf.Variable(tf.random_normal([n_inputs, 10], stddev=0.01))
    w_o = tf.Variable(tf.random_normal([10, n_outputs], stddev=0.01))

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    py_x = tf.matmul(h, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    #train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    predict_op = tf.argmax(py_x, 1)
      
    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        for i in range(10000):
            sess.run(train_op, feed_dict={X: trX, Y: trY,
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})
            if (i % 250 == 0):
              print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                             p_keep_input: 1.0,
                                                             p_keep_hidden: 1.0})))

  def trainSoftmax(self, featureFile):
    self.featureFile = featureFile
    (label_vector, input_vector) = self.__loadData__(featureFile)
    # Build computation graph by creating nodes for input images and target output classes
    n_inputs = 10
    n_outputs = 8

    # Network input 
    x = tf.placeholder(tf.float32, [None, n_inputs])

    # Network weights
    W = tf.Variable(tf.zeros([n_inputs, n_outputs]))

    # Network bias
    b = tf.Variable(tf.zeros([n_outputs]))

    # Regression model implementation
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # To implement cross-entropy, a new placeholder is needed to input the correct answers
    y_ = tf.placeholder(tf.float32, [None, n_outputs])

    # Cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Train the model
    learnRate = 0.01

    #Train using gradient descent
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)

    # Add accuracy checking nodes
    tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

    # Init variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    # Add ops to save and restore all the variables.
    #saver = tf.train.Saver()

    # Run each training operation with 1000 training examples
    k=[]
    saved=0
    for i in range(10000):
      #sess.run(train_step, feed_dict={x: x_train, y_: y_train})
      #result = sess.run(tf_accuracy, feed_dict={x: x_test, y_: y_test})
      sess.run(train_step, feed_dict={x: input_vector, y_: label_vector})
      result = sess.run(tf_accuracy, feed_dict={x: input_vector, y_: label_vector})
      if (i % 25 == 0):
        print("Run {},{}".format(i,result))
      k.append(result)

    # Evaluate model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    k=np.array(k)
    print(np.where(k==k.max()))
    #sess.saveModele.save(sess, 'my-model', global_step=0)
    #save_path = saver.save(sess, "model.ckpt")
    #print("Model saved in file: %s" % save_path)
    print("Max accuracy: {}".format(k.max()))
    print(' ')
    print('NN training Validation :: Done.\n')
    print ' '

    self.trainedModel = sess
    return self.trainedModel

    #saver.restore(sess, "model.ckpt")
    #print("Model restored.")
