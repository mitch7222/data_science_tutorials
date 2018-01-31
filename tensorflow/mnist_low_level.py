import os
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import shutil
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import IMAGES
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None
LOG_DIR = '/tmp/mnist/test/1'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='W')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='B')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def expand_data_set(data):
    # generate translated Dataset
    i_r, i_c = data.images.shape
    l_r, l_c = data.labels.shape
    gen_images = np.ndarray((4 * i_r, i_c), dtype=np.float32)
    gen_labels = np.ndarray((4 * l_r, l_c))

    # Shift right
    gen_images[0:i_r, 1:] = data.images[:, :-1]
    gen_labels[0:l_r, :] = data.labels
    for i in range(28):
        gen_images[0:i_r, i * 28] = 0

    # Shift left
    gen_images[i_r:2 * i_r, :-1] = data.images[:, 1:]
    gen_labels[i_r:2 * i_r, :] = data.labels
    for i in range(28):
        gen_images[i_r:2 * i_r, i * 28 + 27] = 0

    # Shift up
    gen_images[2 * i_r:3 * i_r, :-28] = data.images[:, 28:]
    gen_labels[2 * i_r:3 * i_r, :] = data.labels
    gen_images[2 * i_r:3 * i_r, :28] = 0

    # Shift down
    gen_images[3 * i_r:4 * i_r, 28:] = data.images[:, :-28]
    gen_labels[3 * i_r:4 * i_r, :] = data.labels
    gen_images[3 * i_r:4 * i_r, :28] = 0

    # Convert back to pixel values
    gen_images = gen_images * 255

    return DataSet(gen_images, gen_labels, reshape=False, one_hot=True)

def main(arg_in):
    print(arg_in)

    print(LOG_DIR)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    mnist_t = expand_data_set(mnist.train)

    # Input/Output
    x = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('x', x_image)

    # Layer 1 Convolution
    with tf.name_scope('Layer_1_Convolution') as scope:
        num_nodes = 32
        l = 28
        x_image = tf.reshape(x, [-1, l, l, 1])

        W_conv1 = weight_variable([5, 5, 1, num_nodes])
        b_conv1 = bias_variable([num_nodes])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        prev_num_nodes = num_nodes
        prev_out = h_pool1

        with tf.name_scope('Weight') as scope:
            variable_summaries(W_conv1)
        with tf.name_scope('Bias') as scope:
            variable_summaries(b_conv1)

    # Layer 2 Convolution
    with tf.name_scope('Layer_2_Convolution') as scope:
        num_nodes = 32
        l = 14
        prev_out = tf.reshape(prev_out, [-1, l, l, prev_num_nodes])

        W_conv2 = weight_variable([5, 5, prev_num_nodes, num_nodes])
        b_conv2 = bias_variable([num_nodes])

        h_conv2 = tf.nn.relu(conv2d(prev_out, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        prev_num_nodes = num_nodes
        prev_out = h_pool2

        with tf.name_scope('Weight') as scope:
            variable_summaries(W_conv1)
        with tf.name_scope('Bias') as scope:
            variable_summaries(b_conv1)

    # Fully Connected
    with tf.name_scope('Fully_connected') as scope:
        num_nodes = 1024
        l = 7
        in_flat = tf.reshape(prev_out, [-1, l * l * prev_num_nodes])
        W_fc1 = weight_variable([l * l * prev_num_nodes, num_nodes])
        b_fc1 = bias_variable([num_nodes])
        h_fc1 = tf.nn.relu(tf.matmul(in_flat, W_fc1) + b_fc1)
        prev_num_nodes = num_nodes
        prev_out = h_fc1

    # Apply Dropout on Fully Connected
    with tf.name_scope('Dropout') as scope:
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(prev_out, keep_prob)
        prev_out = h_fc1_drop

    with tf.name_scope('Readout') as scope:
        num_nodes = 10
        W_fc_out = weight_variable([prev_num_nodes, num_nodes])
        b_fc_out = bias_variable([num_nodes])
        dense = tf.matmul(prev_out, W_fc_out) + b_fc_out
        prev_out = dense

    with tf.name_scope('Softmax') as scope:
        softmax = tf.exp(prev_out) / tf.reduce_sum(tf.exp(prev_out))

    y_conv = dense

    with tf.name_scope('Cost') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=dense))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train') as scope:
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    with tf.name_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        data = mnist_t
        # data = mnist.train
        print('Processing {} samples'.format(data.num_examples))
        for i in range(10000):
            batch = data.next_batch(100)
            if i % 100 == 0:
                validation_batch = (mnist.test.images, mnist.test.labels)
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})
                # saver = tf.train.Saver()
                # saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)

                print('step {}, training accuracy {}, test accuracy:{}'.format(i, train_accuracy, test_accuracy))
            if i % 10 == 0 and 0:
                tests = 1
                e_y = y_conv.eval(feed_dict={x: (batch[0])[:tests], y_: (batch[1])[:tests], keep_prob: 1.0})
                a_t = (batch[1])[:tests]
                for i in range(len(e_y[0])):
                    print('{}: {:8.2f} - {:4.0f}'.format(i, e_y[0][i], a_t[0][i]))
                print()
            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                writer.add_summary(s, i)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('test accuracy {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    writer.close()
    print("Done!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



