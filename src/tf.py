#!/bin/python

'''
Tutorial for training basic tf with linear regression
Inspired from : http://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow/tensorflow-tutorial/ 

BASIC OPERANDI :
nodes : operations
arcs/edges : tensors
Nodes and edges combine to form graph.
a Graph is a blue print which gets initialized in a session.
Sessions can be run on devices (gpu, group of gpu's, cpu , group of cpu's etc.)

'''

import tensorflow as tf
import os

#To silence environment warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Setup graph - blue print for entire graph - operation
#Graph is the backbone of TensorFlow and every computation/operation/variables reside on the graph. We will use it later.
graph = tf.get_default_graph()

#To list all operations available on the graph
print "All available operations for this graph"
for op in graph.get_operations(): 
    print(op.name)

#Graph is like a blue print, but a session is like a session or an execution of the blue print.
'''
A graph is used to define operations, but the operations are only run within a session. Graphs and sessions are created independently of each other. You can imagine graph to be similar to a blueprint, and a session to be similar to a construction site.
'''

'''
Open / close a session like this : 
sess=tf.Session()
... your code ...
... your code ...
sess.close()
'''
'''
OR : run it in a block, good practice : 
with tf.Session() as sess:
   sess.run(f)
'''

#1. declare constants like this : 
a = tf.constant(1.0)
print a

#This is different from other programming languages like python
#can't print/access constant unless you run it inside a session

#2. declare variables like this : 
b = tf.Variable(2.0,name="test_var")

#To initialize all variables together : 
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
   print "constant"
   print sess.run(a)
   print "First initialize variables"
   sess.run(init_op)
   print "variables"
   print sess.run(b)

print "All available operations ..."
for op in graph.get_operations():
   print op.name

#3. placeholders are data sink's waiting for data to get populated. Typically used for training data.

a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.multiply(a, b)
 
#Typically we load feed_dict from somewhere else, may be reading from a training data folder. etc
#For simplicity, we have put values in feed_dict here
feed_dict ={a:2,b:3}
print "Multiplication of data using placeholders"
with tf.Session() as sess:
   print(sess.run(y,feed_dict))

