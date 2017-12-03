import tensorflow as tf
import inception_resnet_v1


sess = tf.Session()

images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
train_mode = tf.placeholder(tf.bool)
age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                             phase_train=train_mode,
                                                             weight_decay=1e-5)
gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state("./models/")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    pass

saver.save(sess,"./models/savedmodel.ckpt")
# builder = tf.saved_model.builder.SavedModelBuilder("./models/save")
# inputs = {'images': tf.saved_model.utils.build_tensor_info(images_pl),
#             'train_mode': tf.saved_model.utils.build_tensor_info(train_mode)}
# outputs = {'ages' : tf.saved_model.utils.build_tensor_info(age),
#            'genders':tf.saved_model.utils.build_tensor_info(gender)}
#
# signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'sig_name')
# builder.add_meta_graph_and_variables(sess, ['inception'], {'signature':signature})
# builder.save()