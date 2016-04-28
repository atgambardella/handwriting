# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import random

import numpy
from PIL import Image
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import control_flow_ops

WORK_DIRECTORY = 'data'
#BORDERLESS_SIZE = 40
IMAGE_SIZE = 32
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 100
IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_LABELS*IMAGES_PER_CLASS
TEST_SIZE = int(NUM_IMAGES*0.2)
VALIDATION_SIZE = int(int(NUM_IMAGES*0.8)*0.2)  # Size of the validation set.
TRAIN_SIZE = NUM_IMAGES - TEST_SIZE - VALIDATION_SIZE
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

# Hyperparameters
"""
base_learning_rates = [0.001, 0.01, 0.1]
decay_rates = [0.5, 0.75, 0.9, 1]
fc_depths = [128, 256, 512, 1024]
conv_depths = [16, 32, 64]
filter_sizes = [3, 5]
dropout_rates = [0.5, 0.75, 1]

base_learning_rate = random.choice(base_learning_rates)
decay_rate = random.choice(decay_rates)
fc_depth = random.choice(fc_depths)
conv_depth = random.choice(conv_depths)
filter_size = random.choice(filter_sizes)
dropout_rate = random.choice(dropout_rates)
l2_reg = random.choice([True, False])
#"""

base_learning_rate = 0.001
decay_rate = 0.9
conv_depth = 64
filter_size = 5
dropout_rate = 0.75

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

def extract_data_and_labels(top_level="data/handwriting_chinese_100_classes/"):
  from matplotlib import pylab as plt
  data = numpy.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=numpy.float32)
  labels = numpy.zeros(NUM_IMAGES, dtype=numpy.int64)
  for i, label in enumerate(os.listdir(top_level)):
    for j, filename in enumerate(os.listdir(os.path.join(top_level, label))):
      img = Image.open(os.path.join(top_level, label, filename))
      """
      img = img.resize((BORDERLESS_SIZE, BORDERLESS_SIZE), Image.ANTIALIAS)
      img_with_border = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
      img_with_border.paste(img, ((IMAGE_SIZE - BORDERLESS_SIZE)//2, (IMAGE_SIZE - BORDERLESS_SIZE)//2))
      img_with_border = numpy.asarray(img_with_border, dtype=numpy.float32)
      data[i*IMAGES_PER_CLASS + j, :, :, 0] = img_with_border
      """
      img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
      data[i*IMAGES_PER_CLASS + j, :, :, 0] = img
      #The last 6 classes are shifted up by 162
      int_label = int(label, 16) - int("B0A1", 16) if int(label, 16) <= int("B0FE", 16) else int(label, 16) - int("B0A1", 16) - 162
      labels[i*IMAGES_PER_CLASS + j] = int_label
  data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  return data, labels

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def shuffle_in_unison_inplace(a, b):
  """
  http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
  """
  assert len(a) == len(b)
  p = numpy.random.permutation(len(a))
  return a[p], b[p]

def count_classes(labels):
  from collections import Counter
  c = Counter()
  for label in labels:
    c[label] += 1
  for k, v in c.iteritems():
    print(k)
    print(v)
  print()

def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
    """
    http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/34634291#34634291
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
            name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        """
        mean, var = control_flow_ops.cond(phase_train,
            mean_var_with_update,
            lambda: (ema_mean, ema_var))
        """
        mean, var = (ema_mean, ema_var) if phase_train else mean_var_with_update()
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, 1e-3, affine)
    return normed

def main(argv=None):  # pylint: disable=unused-argument
  print("Hyperparameters:")
  print("base_learning_rate =", base_learning_rate)
  print("decay_rate =", decay_rate)
  print("conv_depth =", conv_depth)
  print("filter_size =", filter_size)
  print("dropout_rate =", dropout_rate)
  # Extract it into numpy arrays.
  train_data, train_labels = extract_data_and_labels()
  train_data, train_labels = shuffle_in_unison_inplace(train_data, train_labels)
  test_data, test_labels = train_data[:TEST_SIZE, ...], train_labels[:TEST_SIZE]
  train_data, train_labels = train_data[TEST_SIZE:, ...], train_labels[TEST_SIZE:]
  validation_data, validation_labels = train_data[:VALIDATION_SIZE, ...], train_labels[:VALIDATION_SIZE]
  train_data, train_labels = train_data[VALIDATION_SIZE:, ...], train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  #map(count_classes, [train_labels, validation_labels, test_labels])
  assert(TRAIN_SIZE == train_labels.shape[0])
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([filter_size, filter_size, NUM_CHANNELS, conv_depth],
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.constant(0.1, shape=[conv_depth]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([filter_size, filter_size, conv_depth, conv_depth*2],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv_depth*2]))
  conv3_weights = tf.Variable(
      tf.truncated_normal([filter_size, filter_size, conv_depth*2, conv_depth*4],
                          stddev=0.1,
                          seed=SEED))
  conv3_biases = tf.Variable(tf.constant(0.1, shape=[conv_depth*4]))
  """
  conv4_weights = tf.Variable(
      tf.truncated_normal([filter_size, filter_size, conv_depth*4, conv_depth*8],
                          stddev=0.1,
                          seed=SEED))
  conv4_biases = tf.Variable(tf.constant(0.1, shape=[conv_depth*8]))
  """
  softmax_weights = tf.Variable(
    tf.truncated_normal([4096, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  softmax_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    if train:
      relu = tf.nn.dropout(relu, dropout_rate, seed=SEED)
    conv = tf.nn.conv2d(relu,
                        conv2_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    if train:
      relu = tf.nn.dropout(relu, dropout_rate, seed=SEED)
    conv = tf.nn.conv2d(relu,
                        conv3_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
    if train:
      relu = tf.nn.dropout(relu, dropout_rate, seed=SEED)
    """
    conv = tf.nn.conv2d(relu,
                        conv4_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
    if train:
      relu = tf.nn.dropout(relu, dropout_rate, seed=SEED)
    """
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    relu_shape = relu.get_shape().as_list()
    reshape = tf.reshape(
        relu,
        [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    return tf.matmul(reshape, softmax_weights) + softmax_biases
  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))
  """
  if l2_reg:
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
  """
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule.
  learning_rate = tf.train.exponential_decay(
      base_learning_rate,  # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      TRAIN_SIZE,          # Decay step.
      decay_rate,          # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions
  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * TRAIN_SIZE) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      #"""
      if step%(TRAIN_SIZE//BATCH_SIZE) == 0:
        print("shuffling data")
        train_data, train_labels = shuffle_in_unison_inplace(train_data, train_labels)
      #"""
      offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / TRAIN_SIZE,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.6f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.6f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.6f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)

if __name__ == '__main__':
  tf.app.run()
