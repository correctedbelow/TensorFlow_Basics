import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


x = tf.placeholder(tf.float32)
W = tf.Variable([0.1], dtype=tf.float32) # A single weight
b = tf.Variable([-0.3], dtype=tf.float32)
y = x * W + b

session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!
print(session.run(y, {x: 5.0}))