#!/bin/python
__author__ = 'ashwin'

'''
Simple model which uses tf for some basic math operations 
'''

import tensorflow as tf
import os

#To silence environment warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

w=tf.Variable(tf.random_normal([784, 10], stddev=0.01)) 
b = tf.Variable([10,20,30,40,50,60],name='t')

#To initialize all variables together : 
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
   print "Activating variables"
   sess.run(init_op)
   print "Printing variable w"
   print sess.run(w)
   print "Printing mean of b"
   print sess.run(tf.reduce_mean(b))


a=[ [0.1, 4,  0.3  ],
    [20,  2,       3   ]
  ]


b = tf.Variable(a,name='b')

#To initialize all variables together : 
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
   print "argmax of new a"
   sess.run(init_op)
   print sess.run(tf.argmax(b,0))
