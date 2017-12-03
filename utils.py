import tensorflow as tf
import os


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'age': tf.FixedLenFeature([], tf.int64),
            'gender': tf.FixedLenFeature([], tf.int64),
            'file_name': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    # image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    # image = tf.image.resize_images(image, [64, 64])
    # image = tf.cast(image, tf.uint8)
    # image.set_shape([mnist.IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = image * (1. / 255) - 0.5

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([160 * 160 * 3])
    image = tf.reshape(image, [160, 160, 3])
    image = tf.reverse_v2(image, [-1])
    image = tf.image.per_image_standardization(image)
    # image = tf.cast(image,tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    age = features['age']
    gender = features['gender']
    file_path = features['file_name']
    return image, age, gender, file_path


def inputs(path, batch_size, num_epochs, allow_smaller_final_batch=False):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    # filename = os.path.join(FLAGS.train_dir,
    #                       TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            path, num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, age, gender, file_path = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels, genders, file_paths = tf.train.shuffle_batch(
            [image, age, gender, file_path], batch_size=batch_size, num_threads=12,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000, allow_smaller_final_batch=allow_smaller_final_batch)

        return images, sparse_labels, genders, file_paths


def get_files_name(path):
    list = os.listdir(path)
    result = []
    for line in list:
        file_path = os.path.join(path, line)
        if os.path.isfile(file_path):
            result.append(file_path)
    return result

    # def choose_best_model(sess, model_path):
    #     ckpt = tf.train.get_checkpoint_state(model_path)
    #     best_gender_acc,best_gender_idx = 0.0,0
    #     best_age_mae,best_age_idx = 100.0,0
    #     for idx in range(len(ckpt.all_model_checkpoint_paths)):
    #         print("restore model %d!" % idx)
    #         _, _, _, _, _, mean_error_age, mean_gender_acc, mean_loss, _, sess=test_once(sess,ckpt.all_model_checkpoint_paths[idx])
    #         if mean_gender_acc>best_gender_acc:
    #             best_gender_acc,best_gender_idx = mean_gender_acc,idx
    #         if mean_error_age<best_age_mae:
    #             best_age_mae,best_age_idx = mean_error_age,idx
    #     return best_gender_acc,ckpt.all_model_checkpoint_paths[best_gender_idx],best_age_mae,ckpt.all_model_checkpoint_paths[best_age_idx],sess
