import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


x = tf.placeholder(tf.float32)
W = tf.Variable([[0.1], [0.2]], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
y = tf.matmul(x, W) + b
session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!
print(session.run(y, {x: [[5.0, 7.0]]}))