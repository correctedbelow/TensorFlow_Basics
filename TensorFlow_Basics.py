import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))


x = tf.placeholder(tf.float32, shape=(None, 2))
h = tf.layers.dense(x, units=2, activation=tf.sigmoid, use_bias=True,
                    kernel_initializer=tf.constant_initializer([[0.1], [0.2]]), # Shape doesn't matter
                    bias_initializer=tf.constant_initializer([-0.3, 0.4])
                    )
y = tf.layers.dense(h, units=1, use_bias=True,
                    kernel_initializer=tf.constant_initializer([0.5, -0.6]), # See, it doesn't care about the shape
                    bias_initializer=tf.constant_initializer([0.7])
                    )
session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!

xor_inputs = [
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]
       ]
xor_outputs = [
       [0],
       [1],
       [1],
       [0]
       ]

prediction = session.run(y, {x: xor_inputs})
print(prediction)
error = prediction - xor_outputs
print('error:', error)
squared_error = (prediction - xor_outputs) * (prediction - xor_outputs)
mean_squared_error = sum(squared_error) / len(error)
print('mean_squared_error:', mean_squared_error)
