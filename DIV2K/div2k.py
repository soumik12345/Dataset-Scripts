import numpy as np
from glob import glob
from tqdm import tqdm
import os, random, math
import tensorflow as tf
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TFRecordCreator:

    def __init__(self, hr_images_path, lr_images_path):
        self.hr_images_path = hr_images_path
        self.lr_images_path = lr_images_path
    
    def _byte_feature(self, val):
        val = val.numpy() if isinstance(val, type(tf.constant(0))) else val
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[val]
            )
        )
    
    def _float_feature(self, val):
        return tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[val]
            )
        )
    
    def _int64_feature(self, val):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[val]
            )
        )
    
    def make_binary_example(self, image_name, hr_img_str, lr_img_str):
        feature = {
            'image/img_name': self._byte_feature(image_name),
            'image/hr_image': self._byte_feature(hr_img_str),
            'image/lr_image': self._byte_feature(lr_img_str)
        }
        return tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
    
    def make_tfrecord_file(self, output_path):
        samples = []
        for hr_img_path in glob(self.hr_images_path + '/*.png'):
            image_name = os.path.basename(hr_img_path).replace('.png', '')
            lr_img_path = os.path.join(self.lr_images_path, image_name + 'x4.png')
            samples.append((
                image_name,
                hr_img_path,
                lr_img_path
            ))
        random.shuffle(samples)
        with tf.io.TFRecordWriter(output_path) as writer:
            for img_name, hr_img_path, lr_img_path in tqdm(samples):
                hr_img_str = open(hr_img_path, 'rb').read()
                lr_img_str = open(lr_img_path, 'rb').read()
                tf_example = self.make_binary_example(
                    str.encode(img_name),
                    hr_img_str, lr_img_str
                )
                writer.write(tf_example.SerializeToString())



class SRTfrecordDataset:

    def __init__(self, gt_size, scale=4, apply_flip=True, apply_rotation=True):
        self.gt_size = gt_size
        self.scale = scale
        self.apply_flip = apply_flip
        self.apply_rotation = apply_rotation
    
    def apply_random_crop(self, lr_image, hr_image):
        lr_image_shape = tf.shape(lr_image)
        hr_image_shape = tf.shape(hr_image)
        gt_shape = (
            self.gt_size,
            self.gt_size,
            3
        )
        lr_shape = (
            self.gt_size // self.scale,
            self.gt_size // self.scale,
            3
        )
        limit = lr_image_shape - lr_shape + 1
        offset = tf.random.uniform(
            tf.shape(lr_image_shape),
            dtype=tf.int32, maxval=tf.int32.max
        ) % limit
        lr_image = tf.slice(lr_image, offset, lr_shape)
        hr_image = tf.slice(hr_image, offset * self.scale, gt_shape)
        return lr_image, hr_image
    
    def normalize(self, lr_image, hr_image):
        lr_image = tf.cast(lr_image, dtype=tf.float32)
        hr_image = tf.cast(hr_image, dtype=tf.float32)
        return lr_image * 2.0 - 1.0, hr_image  * 2.0 - 1.0
    
    def parse_tfrecord(self, tfrecord_file):
        def parse(tfrecord):
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/hr_image': tf.io.FixedLenFeature([], tf.string),
                'image/lr_image': tf.io.FixedLenFeature([], tf.string)
            }
            x = tf.io.parse_single_example(tfrecord, features)
            lr_image = tf.image.decode_png(x['image/lr_image'], channels=3)
            hr_image = tf.image.decode_png(x['image/hr_image'], channels=3)
            lr_image, hr_image = self.apply_random_crop(lr_image, hr_image)
            lr_image, hr_image = self.normalize(lr_image, hr_image)
            return lr_image, hr_image
        return parse
    
    def get_dataset(self, tfrecord_file, batch_size, buffer_size):
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(
            self.parse_tfrecord(tfrecord_file),
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            Augmentation.flip_lr,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            Augmentation.flip_ud,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

class Augmentation:

    def __init__(self):
        pass
    
    @staticmethod
    def flip_lr(img_1, img_2):
        do_flip = tf.random.uniform([]) > 0.5
        img_1 = tf.cond(do_flip, lambda: tf.image.flip_left_right(img_1), lambda: img_1)
        img_2 = tf.cond(do_flip, lambda: tf.image.flip_left_right(img_2), lambda: img_2)
        return img_1, img_2
    
    @staticmethod
    def flip_ud(img_1, img_2):
        do_flip = tf.random.uniform([]) > 0.5
        img_1 = tf.cond(do_flip, lambda: tf.image.flip_up_down(img_1), lambda: img_1)
        img_2 = tf.cond(do_flip, lambda: tf.image.flip_up_down(img_2), lambda: img_2)
        return img_1, img_2
    
    @staticmethod
    def rotate_90(img_1, img_2):
        do_rotation = tf.random.uniform([])
        angle = tf.random.uniform([]).numpy() * 3
        img_1 = tf.cond(do_rotation, lambda: tf.image.rot90(img_1, k=math.ceil(angle)), lambda: img_1)
        img_2 = tf.cond(do_rotation, lambda: tf.image.rot90(img_2, k=math.ceil(angle)), lambda: img_2)
        return img_1, img_2