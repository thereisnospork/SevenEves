'''rewrite of MNIST'''

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):  #start fitting method, inputs last value/placeholder (?)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  #imports data from tf examples

    x = tf.placeholder(tf.float32,[None,784])  #y = Wx + b. MODEL.  784 is length of vector (tensor), indifinite length of input (hence None)
    W = tf.Variable(tf.zeros([784, 10])) #catagorization, 0-9, for each array pixel (784 total) init to 0
    b = tf.Variable(tf.zeros([10])) # intercept, 0-9
    y = tf.matmul(x, W) + b #product of fitting model. matrix multiplacation, plus intercept b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help = 'Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv= [sys.argv[0]] + unparsed)
