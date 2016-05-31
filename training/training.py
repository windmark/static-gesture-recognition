# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn import cross_validation, datasets, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import matplotlib.colors as clr



DICT = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}
COLOR_MAP = {'INIT': 'olivedrab', 'ALCOHOL': 'gold', 'NON-ALCOHOL': 'black', 'FOOD': 'tomato', \
            'UNDO': 'orange', 'CHECKOUT': 'red', 'CASH': 'magenta', 'CREDIT': 'royalblue'}


'''
Reads the data of featureFile, expecting a format of labels and input data.
'''
def loadData(featureFile):
  label_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = (0,), dtype = str)
  input_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)
  return (label_vector, input_vector)


''' KNN CLASS '''
class Knn:
  featureFile = ''
  trainedModel = None
  n_neighbors = 2
  percentSplit = 0.7

  '''
  Trains the model based on the given featureFile and sets it as the new model of the class. 
  '''
  def train(self, featureFile):
    self.featureFile = featureFile
    (label_vector, input_vector) = loadData(featureFile)

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(input_vector, label_vector)
    self.trainedModel = kNNClassifier
    return self.trainedModel

  '''
  Classifies a given feature vector and returns the label.
  '''
  def classify(self, featureVector):
    predicted = self.trainedModel.predict(np.array(featureVector).reshape(1,-1))
    label = DICT[predicted[0]]
    return label

  '''
  Saves the current model to file.
  '''
  def saveModel(self, file):
    joblib.dump(self.trainedModel, file)

  '''
  Loads the model in file and sets it as the new model of the class.
  '''
  def loadModel(self, file):
    self.trainedModel = joblib.load(file)

  '''
  Validate the model based on external data, that has not been trained on.
  '''
  def externalValidateModel(self, separateFeatureFile):
    (label_vector, input_vector) = loadData(self.featureFile)    
    (test_label_vector, test_input_vector) = loadData(separateFeatureFile)

    predictedLabels = self.trainedModel.predict(test_input_vector)

    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(test_label_vector, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label_vector, predictedLabels))
    print('Split Validation training :: Done.\n')

  '''
  Performs split validation on the model. Call with visualizePredictions=True to plot the predictions.
  '''
  def splitValidateModel(self, visualizePredictions = False):
    (label_vector, input_vector) = loadData(self.featureFile)

    indexArray = range(0, len(input_vector))
    trainData, testData, trainLabels, expectedLabels, trainIndices, testIndices = \
      cross_validation.train_test_split(input_vector, label_vector, indexArray, test_size=(1.0 - self.percentSplit))

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(trainData, trainLabels) 
    predictedLabels = kNNClassifier.predict(testData)
    
    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(expectedLabels, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expectedLabels, predictedLabels))
    print('Split Validation training :: Done.\n')

    if visualizePredictions:
      self.__visualizePredictedDataset__(input_vector, testIndices, predictedLabels, expectedLabels)

  '''
  Perform cross validatation on the model.
  '''  
  def crossValidateModel(self):
    (label_vector, input_vector) = loadData(self.featureFile)
    kFold = 5

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    scores = cross_validation.cross_val_score(kNNClassifier, input_vector, label_vector, cv = kFold)
    
    print("\n----- k-fold Cross Validation -----")
    print(scores)
    print("Average: ", sum(scores) / len(scores))

  '''
  Visualize the dataset. Plot new features by calling with specific featureData file,
  and save it to an image file by specifying fileName.
  '''
  def visualizeData(self, featureData = '', fileName = ''):
    if featureData == '':
      (label_vector, input_vector) = loadData(self.featureFile)
    else:
      (label_vector, input_vector) = loadData(featureData)

    pca = PCA(n_components = 2)
    X_trans = pca.fit_transform(input_vector)

    plt.figure()
    colorArray = []
    for n in range(0, len(input_vector)):
      colorArray.append(COLOR_MAP[label_vector[n]])

    plt.scatter(X_trans[:,0], X_trans[:,1], c = colorArray)
    if fileName == '':
      plt.show()
    else:
      plt.savefig(fileName)
      print "Plot saved as " + fileName + ".png"

  '''
  Visualized the predicted dataset, based on results from a split validation.
  '''
  def __visualizePredictedDataset__(self, data, testIndices, predictedLabels, expectedLabels):
    pca = PCA(n_components = 2)
    X_trans = pca.fit_transform(data)

    plt.figure()
    colorArray = []

    print("----- Wrong predictions -----")
    for n in range(0, len(data)):
      if n in testIndices:
        if predictedLabels[testIndices.index(n)] != expectedLabels[testIndices.index(n)]:
          colorArray.append('red')
          print("Expected", expectedLabels[testIndices.index(n)], 
                    "Predicted", predictedLabels[testIndices.index(n)])
        else:
          colorArray.append('olivedrab')
      else:
        colorArray.append('white')

    plt.scatter(X_trans[:,0], X_trans[:,1], c = colorArray)
    plt.show()

  '''
  Train the model iteratively using sample steps of n_datapoints. 
  '''
  def trainLimited(self, featureFile, n_datapoints):
    (label_vector, input_vector) = loadData(featureFile)

    trainData, testData, trainLabels, testLabels = \
      cross_validation.train_test_split(input_vector, label_vector, test_size=(0))

    n_totalrows = int((len(label_vector)/n_datapoints))
    for n in range(0, n_totalrows):
      limited_label_vector = trainLabels[0: (n+1) * n_datapoints]
      limited_input_vector = trainData[0: (n+1) * n_datapoints]

      kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
      kNNClassifier.fit(limited_input_vector, limited_label_vector)

      scores = cross_validation.cross_val_score(kNNClassifier, limited_input_vector, limited_label_vector, cv = 5)
      print '%f on %d datapoints' % ((sum(scores) / len(scores)), len(limited_label_vector))






''' NEURAL NETWORK '''
class NeuralNetwork:
  trainedModel = None
  featureFile = ''

  '''
  Reads the data from featureFile and processes it to the right format.
  '''
  def __loadData__(self, featureFile):
    (label_vector_strings, input_vector) = loadData(featureFile)

    label_vector = []
    for n in range(0,len(label_vector_strings)):
      index = DICT[label_vector_strings[n]]
      label_array_representation = [0, 0, 0, 0, 0, 0, 0, 0]
      label_array_representation[index] = 1
      label_vector.append(label_array_representation)
    return (label_vector, input_vector)

  '''
  Saves the current model to file.
  '''
  def saveModel(self, file):
    saver = tf.train.Saver()
    save_path = saver.save(self.trainedModel, ("%s.ckpt" % file))
    print("Model saved in file: %s" % save_path)

  '''
  Loads the model in file and sets it as the new model of the class.
  '''
  def loadModel(self, file):
    saver = tf.train.Saver()
    saver.restore(self.trainedModel, ("%s.ckpt" % file))
    print("Model restored.")

  '''
  Trains the MLP model based on the given featureFile and sets it as the new model of the class. 
  '''
  def trainMLP(self, featureFile):
    self.featureFile = featureFile
    (label_vector, input_vector) = self.__loadData__(featureFile)
    self.trainMLPWithData(input_vector, label_vector)

  '''
  Trains the MLP model based on the given input data, making it possible to
  specify the steps between prints.
  '''
  def trainMLPWithData(self, input_vector, label_vector, printSteps = 250):
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

    learnRate = 0.01
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    #train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cost)

    # Add accuracy checking nodes
    tf_correct_prediction = tf.equal(tf.argmax(py_x,1), tf.argmax(teY,1))
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

    # Init variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    k=[]
    for i in range(10000):
        sess.run(train_step, feed_dict={X: trX, Y: trY, p_keep_input: 0.8, p_keep_hidden: 0.5})
        result = sess.run(tf_accuracy, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0})
        # Save data
        k.append(result)
        if (i % printSteps == 0):
          print("Run {},{}".format(i,result))

    k=np.array(k)
    print("Max accuracy: {}".format(k.max()))
    print(('MLP training with %s datapoints :: Done \n\n') % (len(input_vector)))

    self.trainedModel = sess
    return (self.trainedModel, k.max())
  
  '''
  Train the MLP model iteratively using sample steps of n_datapoints. 
  '''
  def trainLimitedMLP(self, featureFile, n_datapoints):
    (label_vector, input_vector) = self.__loadData__(featureFile)

    n_totalrows = int((len(label_vector)/n_datapoints))
    k=[]
    for n in range(0, n_totalrows):
      trainData, testData, trainLabels, testLabels = \
        cross_validation.train_test_split(input_vector, label_vector, test_size=(0.2))

      limited_label_vector = trainLabels[0: (n+1) * n_datapoints]
      limited_input_vector = trainData[0: (n+1) * n_datapoints]

      average = []
      for a in range(0,5):
        _, maxVal = self.trainMLPWithData(limited_input_vector, limited_label_vector, 1000)
        average.append(maxVal)

      averageMaxVal = sum(average) / len(average)
      print 'Total Average Value: %s \n\n' % (averageMaxVal)
      average = []
      k.append(averageMaxVal)

    print('Limited MLP training result -------------')
    for i in range (0,len(k)):
        print '%f on %d datapoints' % (k[i], n_datapoints * (i+1))
    print '------------------------------------------'

  '''
  Trains the Softmax model based on the given featureFile and sets it as the new model of the class. 
  '''
  def trainSoftmax(self, featureFile, n_datapoints):
    self.featureFile = featureFile
    (label_vector, input_vector) = self.__loadData__(featureFile)
    self.trainSoftmaxWithData(input_vector, label_vector)

  def trainSoftmaxWithData(self, input_vector, label_vector, printSteps = 250):
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

    learnRate = 0.01

    #Train using gradient descent
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)
    #train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    # Add accuracy checking nodes
    tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

    # Init variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    # Split data
    percent_split = 0.7
    trX, teX, trY, teY = cross_validation.train_test_split(input_vector, 
              label_vector, test_size=(1.0-percent_split), random_state=0)

    # Run each training operation with 1000 training examples
    k=[]
    for i in range(10000):
      sess.run(train_step, feed_dict={x: trX, y_: trY})
      result = sess.run(tf_accuracy, feed_dict={x: teX, y_: teY})
      # Save result
      k.append(result)
      if (i % printSteps == 0):
        print("Run {},{}".format(i,result))

    # Evaluate model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    k=np.array(k)
    print("Max accuracy: {}".format(k.max()))
    print(('Softmax training with %s datapoints :: Done \n\n') % (len(input_vector)))

    self.trainedModel = sess
    return (self.trainedModel, k.max())

  '''
  Trains the Softmax model based on the given input data, making it possible to
  specify the steps between prints.
  '''
  def trainLimitedSoftmax(self, featureFile, n_datapoints):
    (label_vector, input_vector) = self.__loadData__(featureFile)

    n_totalrows = int((len(label_vector)/n_datapoints))
    k=[]
    trainData, testData, trainLabels, testLabels = \
        cross_validation.train_test_split(input_vector, label_vector, test_size=(0.2))

    for n in range(0, n_totalrows):

      limited_label_vector = trainLabels[0: (n+1) * n_datapoints]
      limited_input_vector = trainData[0: (n+1) * n_datapoints]

      _, maxVal = self.trainSoftmaxWithData(limited_input_vector, limited_label_vector, 1000)

      print 'Total Average Value: %s \n\n' % (maxVal)
      k.append(maxVal)

    print('Limited Softmax training result ----------')
    for i in range (0,len(k)):
      print '%f on %d datapoints' % (k[i], (n_datapoints * (i+1)))
    print '------------------------------------------'
