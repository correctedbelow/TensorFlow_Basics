import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


x = tf.placeholder(tf.float32)
W1 = tf.Variable([[0.1], [0.2]], dtype=tf.float32)
b1 = tf.Variable([-0.3, 0.4], dtype=tf.float32)
W2 = tf.Variable([[0.5], [-0.6]], dtype=tf.float32)
b2 = tf.Variable([0.7], dtype=tf.float32)
h = tf.matmul(x, W1) + b1
h = tf.sigmoid(h)
y = tf.matmul(h, W2) + b2

session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!
print(session.run(y, {x: [[5.0, 7.0]]}))