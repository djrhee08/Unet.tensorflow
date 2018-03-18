# Created on Wed May 31 14:48:46 2017
# @author: Frederik Kratzert
# @modified by Dong Joo Rhee

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import glob
import dicom_utils.dicomPreprocess as dpp
import skimage.transform as sktf


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, dir_path, mode, rotation_status, rotation_angle, batch_size, shuffle=True, buffer_size=1):
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
            shuffle: Whether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """
        self.img_path = dir_path + '/image'
        self.mask_path = dir_path + '/mask'
        self.mask_list = []
        self.img_list = []
        self.rotation_status = rotation_status
        self.rotation_angle = rotation_angle

        # TODO 1: img_path and mask_path should be matched and reorganized here!!!
        # TODO 2: Check what buffer size is.
        for file in glob.glob(self.mask_path + "/*.npy"):
            self.mask_list.append(file)

        for file in glob.glob(self.img_path + "/*.npy"):
            self.img_list.append(file)

        # number of samples in the dataset
        self.data_size = len(self.mask_list)

        # initial shuffling of the image and mask lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.image_list = tf.constant(self.img_list)
        self.mask_list = tf.constant(self.mask_list)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_list, self.mask_list))

        data = data.map(lambda img_fname, mask_fname : tuple(tf.py_func(self._read_py_function, [img_fname, mask_fname],
                                                                        [tf.double, tf.float64])), num_parallel_calls=4)


        # Set image/mask types to be uint16, bool respectively. Then, convert them into
        # float32/float32 (? think about it) for feeding

        data = data.repeat()
        data = data.batch(batch_size)

        if mode == 'training':
            data = data.map(self.train_function, num_parallel_calls=4)
            #  Currently have some problem + not compatible with what I try to do..
            #data = data.apply(tf.contrib.data.map_and_batch(map_func=self._parse_function_train, batch_size=batch_size))
        elif mode == 'validation':
            data = data.map(self.validation_function, num_parallel_calls=4)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        data = data.prefetch(buffer_size=buffer_size)

        self.data = data

        """
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        """

    def _read_py_function(self, img_fname, mask_fname):
        #print(img_fname.decode('utf-8'))
        img_fname = img_fname.decode('utf-8')
        mask_fname = mask_fname.decode('utf-8')
        #print(img_fname, mask_fname)

        image = np.load(img_fname)
        mask = np.load(mask_fname)
        mask = dpp.resize_2d(mask, (472,472))

        if self.rotation_status == True:
            aug_img = image[np.newaxis, :, :]
            aug_mask = mask[np.newaxis, :, :]
            for i in range(len(self.rotation_angle)):
                rotate_img = dpp.rotate(image, self.rotation_angle[i])
                aug_img = np.append(aug_img, rotate_img[np.newaxis, :, :], axis=0)
                aug_mask = np.append(aug_mask, mask[np.newaxis, :, :], axis=0)
        else:
            aug_img = image
            aug_mask = mask

        return aug_img, aug_mask


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths."""
        img_path = self.img_list
        mask_path = self.mask_list
        permutation = np.random.permutation(self.data_size)
        self.img_list = []
        self.mask_list = []
        for i in permutation:
            self.img_list.append(img_path[i])
            self.mask_list.append(mask_path[i])


    def train_function(self, img, mask):

        """
        #Data augmentation + 3 channel addition comes here.
        """

        img = tf.reshape(img, shape=(-1, 512, 512))
        img = tf.expand_dims(img, -1)

        # TODO : Change 472 -> dynamic variable
        #print(tf.shape(mask), tf.shape(mask)[2])
        mask = tf.reshape(mask, shape=(-1, 472, 472))
        #mask = tf.image.resize_images(mask, [tf.shape(mask)[0], 472, 472])
        mask = tf.expand_dims(mask, -1)

        return img, mask

    def validation_function(self, img, mask):

        """
        #Data augmentation + 3 channel addition comes here.
        """

        return img, mask
