# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime
from scipy.io import loadmat

import tensorflow as tf

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import pandas as pd
import numpy as np
import skimage.io as io
from tqdm import tqdm

from sklearn.model_selection import train_test_split
FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    file_name = data_set.file_name
    genders = data_set.gender
    ages = data_set.age
    face_score = data_set.score
    second_face_score = data_set.second_score
    num_examples = data_set.shape[0]
    base_dir = "data/imdb_crop"

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=64)

    error=0
    total=0
    # if images.shape[0] != num_examples:
    #     raise ValueError('Images size %d does not match label size %d.' %
    #                      (images.shape[0], num_examples))
    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shape[3]

    filename = os.path.join(name + '.tfrecords')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in tqdm(range(num_examples)):
            if face_score[index] < 0.75:
                continue
            # if (~np.isnan(second_face_score[index])) and second_face_score[index] > 0.0:
            #     continue
            if ~(0 <= ages[index] <= 100):
                continue

            if np.isnan(genders[index]):
                continue

            try:
                # image_raw = io.imread(os.path.join(base_dir,file_names[index])).tostring()
                # image_raw = open(os.path.join(base_dir,str(file_name[index][0]))).read()

                # load the input image, resize it, and convert it to grayscale
                image = cv2.imread(os.path.join(base_dir,str(file_name[index][0])),cv2.IMREAD_COLOR)
                image = imutils.resize(image, width=256)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 2)
                if len(rects)!=1:
                    continue
                else:
                    image_raw = fa.align(image, gray, rects[0])
                    image_raw = image_raw.tostring()
            except IOError: #some files seem not exist in face_data dir
                error = error+1
                pass
            # image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                # 'height': _int64_feature(rows),
                # 'width': _int64_feature(cols),
                # 'depth': _int64_feature(depth),
                'age': _int64_feature(int(ages[index])),
                'gender':_int64_feature(int(genders[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
            total = total+1
    print("There are ",error," missing pictures" )
    print("Found" ,total, "valid faces")


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
            "second_score": second_face_score}
    dataset = pd.DataFrame(data)
    return dataset


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def main(unused_argv):
    # Get the data.
    # data_sets = pd.read_csv("gender_age_train.txt", header=None, sep=" ")
    # data_sets.columns = ["file_name", "gender", "age"]
    data_sets = get_meta('./data/imdb_crop/imdb.mat','imdb')
    # data_sets = data_sets[data_sets.age >= 0]
    # data_sets = data_sets[data_sets.age <= 100]

    train_sets,test_sets = train_test_split(data_sets,train_size=0.001,random_state=2017)
    train_sets.reset_index(drop=True, inplace=True)
    test_sets.reset_index(drop=True, inplace=True)
    # data_sets = mnist.read_data_sets(FLAGS.directory,
    #                                  dtype=tf.uint8,
    #                                  reshape=False,
    #                                  validation_size=FLAGS.validation_size)

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_sets, 'train')
    convert_to(test_sets,'test')
    # convert_to(data_sets.validation, 'validation')
    # convert_to(data_sets.test, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--directory',
    #     type=str,
    #     default='/tmp/data',
    #     help='Directory to download data files and write the converted result'
    # )
    # parser.add_argument(
    #     '--validation_size',
    #     type=int,
    #     default=5000,
    #     help="""\
    #   Number of examples to separate from the training data for the validation
    #   set.\
    #   """
    # )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
