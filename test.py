import argparse
import os

import numpy as np
import tensorflow as tf

import inception_resnet_v1
from utils import inputs, get_files_name


def test_once(image_path, batch_size, model_checkpoint_path):
    with tf.Graph().as_default():
        sess = tf.Session()
        images, age_labels, gender_labels, file_paths = inputs(
            path=get_files_name(image_path),
            batch_size=batch_size,
            num_epochs=1, allow_smaller_final_batch=True)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

        gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                              logits=gender_logits)
        gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)
        total_loss = tf.add_n(
            [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
            name="total_loss")

        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        prob_age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        abs_age_error = tf.losses.absolute_difference(prob_age, age_labels)

        prob_gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        mean_error_age, mean_gender_acc, mean_loss = [], [], []
        try:
            while not coord.should_stop():
                prob_gender_val, real_gender, prob_age_val, real_age, image_val, gender_acc_val, abs_age_error_val, cross_entropy_mean_val, file_names = sess.run(
                    [prob_gender, gender_labels, prob_age, age_labels, images, gender_acc, abs_age_error, total_loss,
                     file_paths], {train_mode: False})
                mean_error_age.append(abs_age_error_val)
                mean_gender_acc.append(gender_acc_val)
                mean_loss.append(cross_entropy_mean_val)
                print("Age_MAE:%.2f,Gender_Acc:%.2f%%,Loss:%.2f" % (
                    abs_age_error_val, gender_acc_val * 100, cross_entropy_mean_val))
        except tf.errors.OutOfRangeError:
            print('!!!TESTING DONE!!!')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()
        return prob_age_val, real_age, prob_gender_val, real_gender, image_val, np.mean(mean_error_age), np.mean(
            mean_gender_acc), np.mean(mean_loss), file_names


def choose_best_model(model_path, image_path, batch_size):
    ckpt = tf.train.get_checkpoint_state(model_path)
    best_gender_acc, best_gender_idx = 0.0, 0
    best_age_mae, best_age_idx = 100.0, 0
    result = []
    for idx in range(len(ckpt.all_model_checkpoint_paths)):
        print("restore model %d!" % idx)
        _, _, _, _, _, mean_error_age, mean_gender_acc, mean_loss, _ = test_once(image_path, batch_size,
                                                                                 ckpt.all_model_checkpoint_paths[idx], )
        result.append([ckpt.all_model_checkpoint_paths[idx], mean_error_age, mean_gender_acc])
        if mean_gender_acc > best_gender_acc:
            best_gender_acc, best_gender_idx = mean_gender_acc, idx
        if mean_error_age < best_age_mae:
            best_age_mae, best_age_idx = mean_error_age, idx
    return best_gender_acc, ckpt.all_model_checkpoint_paths[best_gender_idx], best_age_mae, \
           ckpt.all_model_checkpoint_paths[best_age_idx], result


def main(model_path, image_path, batch_size):
    best_gender_acc, gender_model, best_age_mae, age_model, result = choose_best_model(model_path, image_path,
                                                                                       batch_size)
    return best_gender_acc, gender_model, best_age_mae, age_model, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="./data/test", help="Testset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--model_path", type=str, default="./models/", help="Model path")
    parser.add_argument("--choose_best", action="store_true", default=False,
                        help="If you use this flag,will test all models under model path and return the best one.")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.choose_best:
        best_gender_acc, gender_model, best_age_mae, age_model, result = main(args.model_path, args.images,
                                                                              args.batch_size)
        print("Age_MAE:%.2f,Gender_Acc:%.2f%%,Age_model:%s,Gender_model:%s" % (
            best_age_mae, best_gender_acc * 100, age_model, gender_model))
    else:
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            _, _, _, _, _, mean_error_age, mean_gender_acc, mean_loss, _ = test_once(args.images,
                                                                                     args.batch_size,
                                                                                     ckpt.model_checkpoint_path)
            print("Age_MAE:%.2f,Gender_Acc:%.2f%%,Loss:%.2f" % (mean_error_age, mean_gender_acc * 100, mean_loss))
        else:
            raise IOError("Not found a pretrained model")
