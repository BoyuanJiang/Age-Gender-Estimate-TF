# This file used to transfer a pre-trained model's
# weight(it's network architecture may be different from ours) to our model

import tensorflow as tf
import os

def restore_from_source(sess,source_path):
    s_saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(source_path)
    if ckpt and ckpt.model_checkpoint_path:
        s_saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore and continue training!")
        return sess
    else:
        raise IOError("Not found source model")


def _init_all_uninitialized_variables(sess):
    uninitialized_variables = sess.run(tf.report_uninitialized_variables())
    init_op = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(uninitialized_variables)])
    sess.run(init_op)
    init_op = tf.variables_initializer([v for v in tf.local_variables() if v.name.split(':')[0] in set(uninitialized_variables)])
    sess.run(init_op)

def save_to_target(sess,target_path,max_to_keep=5):
    t_saver = tf.train.Saver(max_to_keep=max_to_keep)
    _init_all_uninitialized_variables(sess)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    save_path = t_saver.save(sess, target_path+"model.ckpt",global_step=0)
    print("Model saved in file: %s" % save_path)
    return sess,t_saver
