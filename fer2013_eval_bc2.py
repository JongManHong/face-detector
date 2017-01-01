"""Evaluation for FER2013."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy import misc

import fer2013_2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/fer2013_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 3589,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, logits, labels, top_k_op, summary_op):
  # print("Called eval_once ...")
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
#      print("Checkpoint file path:", ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/fer2013_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_input_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_input_size
      step = 0
      time.sleep(1)

      # print("step = %d, num_iter = %d  " % (step, num_iter))

      emotion_dict = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad'}

      while step < num_iter and not coord.should_stop():
        # print("Inside while ...")
        result1, result2  = sess.run([logits, labels])
        #label = sess.run(labels)
        # print('Step:', step, 'result',result1, 'Label:', result2)
        c = sess.run(tf.arg_max(result1, 1))
        
        
        emotion_dict = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Disgust', 5: 'Surprise', 6; 'Neutral'}
        print("-----------------------------------------------------")
        print('LABEL FOR INPUT IMAGE:', result1, '->', c, emotion_dict[c[0]])
        print("-----------------------------------------------------")
        
        step += 1
        return c, result1

      # print("Exited while! Next...")

      # Compute precision @ 1.
      precision = true_count / step
      # print('Summary -- Step:', step, 'Accurcy:',true_count * 100.0 / step * 1.0, )
      # print('%s: total:%d true:%d precision @ 1 = %.3f' % (datetime.now(), total_sample_count, true_count, precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(img1):
 
  img =  np.array(img1).astype(np.float32)/255.
  
  with tf.Graph().as_default():
    # Get images and labels for FER2013.
    eval_data = FLAGS.eval_data == 'test'
    
    img = tf.image.rgb_to_grayscale(img)

    distorted_image = tf.image.resize_image_with_crop_or_pad(img, 32, 32)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)  
    float_image = tf.reshape(float_image, [1, 32, 32, 1])
    

    
    logits = fer2013_2.inference(float_image)

    # Calculate predictions.
    labels = tf.reshape(tf.constant(18), [1])
    

    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        fer2013_2.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

  
   
    return eval_once(saver, summary_writer, logits, labels, top_k_op, summary_op)
