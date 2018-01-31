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


def main(arg_in):
    print(arg_in)

    print(LOG_DIR)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    data = mnist.train
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

    # COnvert back to pixel values
    gen_images = gen_images * 255

    mnist_t = DataSet(gen_images, gen_labels, reshape=False, one_hot=True)

    sess = tf.InteractiveSession()

    # Input/Output
    x = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    input_layer = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('x', input_layer)

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    tf_is_training = tf.placeholder_with_default(True, shape=())
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=tf_is_training)
    logits = tf.layers.dense(inputs=dropout, units=10)

    with tf.name_scope('Softmax') as scope:
        softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))

    y_conv = logits

    with tf.name_scope('Cost') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits))
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
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], tf_is_training: False})
                test_accuracy = accuracy.eval(
                    feed_dict={x: validation_batch[0], y_: validation_batch[1], tf_is_training: False})
                # saver = tf.train.Saver()
                # saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)

                print('step {}, training accuracy {}, test accuracy:{}'.format(i, train_accuracy, test_accuracy))
            if i % 10 == 0 and 0:
                tests = 1
                e_y = y_conv.eval(feed_dict={x: (batch[0])[:tests], y_: (batch[1])[:tests], tf_is_training: False})
                a_t = (batch[1])[:tests]
                for i in range(len(e_y[0])):
                    print('{}: {:8.2f} - {:4.0f}'.format(i, e_y[0][i], a_t[0][i]))
                print()
            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict={x: batch[0], y_: batch[1], tf_is_training: False})
                writer.add_summary(s, i)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], tf_is_training: True})
        print('test accuracy {}'.format(
            accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, tf_is_training: False})))
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

