#!/bin/python

__author__ = 'ashwin'

'''
Basic tf model to run linear regression.
In the end prints final fit quality.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

#To silence environment warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Setting up the data
trainX = np.linspace(-1,1,101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

#Setting up the edges placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name = 'weights')
y_model = tf.multiply(X,w)

#Declaring model
cost = (tf.pow(Y-y_model,2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

#setup graph
graph = tf.get_default_graph()

#print all available operations 
#for op in graph.get_operations():
#   print op.name 

with tf.Session() as sess:
   sess.run(init)
   #Iterate 100 times to optimize
   for i in range(0,1):
      for (x,y) in zip(trainX, trainY):
         loss_func_value = tf.reduce_sum(Y - y_model)
         _, loss_func_value = sess.run([train_op, loss_func_value], feed_dict = {X: x, Y: y})
      print "Iteration : %s, cost = %s"%(i, loss_func_value)
   print sess.run(w)
   final_w = sess.run(w)

#fitted values
plt.plot(trainX, trainY, 'r')
plt.plot(trainX, trainX*final_w, 'b')
plt.show()
