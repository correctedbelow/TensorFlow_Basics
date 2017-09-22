import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


x = tf.placeholder(tf.float32, shape=(None, 2))
h = tf.layers.dense(x, units=2, activation=tf.sigmoid, use_bias=True)
y = tf.layers.dense(h, units=1, use_bias=True)
session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!
print(session.run(y, {x: [[5.0, 7.0]]}))