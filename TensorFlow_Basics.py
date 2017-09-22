import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


c1 = tf.constant(3.0)
c2 = tf.constant(7.0)
p = c1 * c2

session = tf.Session()
print(session.run([c1, c2, p]))