# Imports
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

# Taking iris data from sklearn
from sklearn import datasets
# 
#
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn import cross_validation

    
matrix_length = (len(np.loadtxt('input.txt', delimiter = ',', dtype = float)[0]))
input_vector = [None] * matrix_length
label_vector = [None] * matrix_length

for n in range(0,matrix_length):
    feature_vector = np.loadtxt('input.txt', 
        delimiter = ',',
        usecols = range(n,n+1),
        dtype = float)
    input_vector[n] = feature_vector

for n in range(0,matrix_length):
    feature_vector = np.loadtxt('label.txt', 
        delimiter = ',',
        usecols = range(n,n+1),
        dtype = float)
    label_vector[n] = feature_vector


# Build computation graph by creating nodes for input images and target output classes
# 10 elements input
# 8 output classes

# network input 
x = tf.placeholder(tf.float32, [None, 50])

# network weights
W = tf.Variable(tf.zeros([50, 4]))

# network bias
b = tf.Variable(tf.zeros([4]))

# Regression model implementation
y = tf.nn.softmax(tf.matmul(x, W) + b)

# To implement cross-entropy, a new placeholder is needed to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 4])

# Cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Train the model
learnRate = 0.5
train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)

# Add accuracy checking nodes
tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

# Init variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Split for early stopping
percentSplit = 0.7
x_train, x_test, y_train, y_test = cross_validation.train_test_split(input_vector, 
            label_vector, 
            test_size=(1.0-percentSplit), 
            random_state=0)


# Run each training operation with 1000 training examples
k=[]
saved=0
for i in range(5000):
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
print("Max accuracy: {}".format(k.max()))
