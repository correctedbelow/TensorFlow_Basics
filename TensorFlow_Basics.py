import sys
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


c1 = tf.constant(3.0)
c2 = tf.constant(7.0)
p = c1 * c2
print(p)
