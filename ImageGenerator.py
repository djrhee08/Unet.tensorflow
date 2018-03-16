# Created on Wed May 31 14:48:46 2017
# @author: Frederik Kratzert
# @modified by Dong Joo Rhee

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

# TODO: Change this to tf.data.Dataset
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, dir_path, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            dir_path: Path to the files.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Whether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """
        self.num_classes = num_classes

        self.img_path = dir_path + '/image'
        self.mask_path = dir_path + '/mask'

        # TODO: img_path and mask_path should be matched and reorganized here!!!



        # number of samples in the dataset
        self.data_size = len(self.mask_path)

        # initial shuffling of the image and mask lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_path = convert_to_tensor(self.img_path, dtype=dtypes.string)
        self.mask_path = convert_to_tensor(self.mask_path, dtype=dtypes.string)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_path, self.mask_path))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8, output_buffer_size=100*batch_size)

        elif mode == 'validation':
            data = data.map(self._parse_function_inference, num_threads=8, output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        # NEWLY ADDED! 'prefetch'
        data = data.prefetch(buffer_size=buffer_size)

        self.data = data


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths."""
        img_path = self.img_path
        mask_path = self.mask_path
        permutation = np.random.permutation(self.data_size)
        self.img_path = []
        self.mask_path = []
        for i in permutation:
            self.img_paths.append(img_path[i])
            self.mask_path.append(mask_path[i])

    def _parse_function_train(self, img_fname, mask_fname):
        """Input parser for samples of the training set."""
        # load and preprocess the image
        img_string = tf.read_file(img_fname)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        """
        Data augmentation + 3 channel addition comes here.
        """

        # load and preprocess the image
        mask_string = tf.read_file(mask_fname)
        # TODO: Change this part!! not decode_png... (maybe just omit it?)
        mask_decoded = tf.image.decode_png(mask_string, channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, [224, 224])

        return img_resized, mask_resized

    def _parse_function_inference(self, img_fname, mask_fname):
        """Input parser for samples of the validation/test set."""
        # load and preprocess the image
        img_string = tf.read_file(img_fname)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])

        # load and preprocess the mask
        mask_string = tf.read_file(mask_fname)
        # TODO: Change this part!! not decode_png... (maybe just omit it?)
        mask_decoded = tf.image.decode_png(mask_string, channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, [224, 224])

        return img_resized, mask_resized
