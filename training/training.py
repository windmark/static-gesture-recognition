# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn import cross_validation, datasets, neighbors, metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


dict = {'INIT': 0, 'ALCOHOL': 1, 'NON-ALCOHOL': 2, 'FOOD': 3, 'UNDO': 4, 'CHECKOUT': 5, 'CASH': 6, 'CREDIT': 7}

class Knn:
  trainedModel = None
  n_neighbors = 5

  def train(self, featureFile):
    label_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = (0,), dtype = str)
    input_vector = np.loadtxt(featureFile, delimiter = ', ', usecols = range(1,11), dtype = float)

    kNNClassifier = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
    kNNClassifier.fit(input_vector, label_vector)
    self.trainedModel = kNNClassifier
    return self.trainedModel

  def classify(self, featureVector):
    predictedLabel = self.trainedModel.predict(featureVector)
    return predictedLabel[0]

  def saveModel(self, file):
    joblib.dump(self.trainedModel, file)