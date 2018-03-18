from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt

from ImageGenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator

import TensorflowUtils as utils
import datetime
from six.moves import xrange
import time
import os

import unet_utils as util
from unet_layer import (weight_variable, weight_variable_devonc, bias_variable, conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2, cross_entropy)
from collections import OrderedDict

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_string('optimization', "dice", "optimization mode: cross_entropy/ dice")
tf.flags.DEFINE_string('data_option', "normal", "data mode: normal/ fast")

MAX_ITERATION = int(9000)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if FLAGS.data_option == "fast":
    dir_image = 'image_fast'
    dir_mask = 'mask_fast'
else:
    dir_image = 'image'
    dir_mask = 'mask'


# Create "logs/" directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")
    print("Directory 'logs' created")


# NUM_OF_CLASSES = the number of segmentation classes + 1 (1 for none for anything)
NUM_OF_CLASSES = 2
IMAGE_SIZE = 512
LOGITS_SIZE = 472

def dice(mask1, mask2, smooth=1e-6):
    print(mask1.shape, mask2.shape)
    print(mask1.flatten().max(), mask2.flatten().max())
    print(np.sum(mask1.flatten()), np.sum(mask2.flatten()))

    mul = mask1 * mask2
    print(mul.flatten().max(), np.sum(mul.flatten()))
    inse = np.sum(mul.flatten())

    l = np.sum(mask1.flatten())
    r = np.sum(mask2.flatten())

    print("l, r, intersection : ", l, r, inse)

    dice_coeff = (2.* inse + smooth) / (l + r + smooth)

    return round(dice_coeff,3)


def u_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2):
    """
       Creates a new convolutional unet for the given parametrization.

       :param x: input tensor, shape [?,nx,ny,channels]
       :param keep_prob: dropout probability tensor
       :param channels: number of channels in the input image
       :param n_class: number of output labels
       :param layers: number of layers in the net
       :param features_root: number of features in the first layer
       :param filter_size: size of the convolution filter
       :param pool_size: size of the max pooling operation
       :param summaries: Flag if summaries should be created
       """

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2 ** layer * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)

        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])

        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size -= 4
        if layer < layers - 1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):
        features = 2 ** (layer + 1) * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))

        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
        bd = bias_variable([features // 2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat

        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
        b1 = bias_variable([features // 2])
        b2 = bias_variable([features // 2])

        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    #return output_map, variables, int(in_size - size)
    return output_map



def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)



def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")

    if FLAGS.optimization == "cross_entropy":
        annotation = tf.placeholder(tf.int32, shape=[None, LOGITS_SIZE, LOGITS_SIZE, 1], name="annotation")   # For cross entropy
        logits = u_net(x=image,keep_prob=0.75,channels=1,n_class=2)

        label = tf.squeeze(annotation, squeeze_dims=[3])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label,name="entropy")) # For softmax

        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy"))  # For softmax

    elif FLAGS.optimization == "dice":
        annotation = tf.placeholder(tf.int32, shape=[None, LOGITS_SIZE, LOGITS_SIZE, 1], name="annotation")  # For DICE
        logits = u_net(x=image,keep_prob=0.75,channels=1,n_class=2)

        # pred_annotation (argmax) is not differentiable so it cannot be optimized. So in loss, we need to use logits instead of pred_annotation!
        logits = tf.nn.softmax(logits) # axis = -1 default
        logits2 = tf.slice(logits, [0,0,0,1],[-1,LOGITS_SIZE,LOGITS_SIZE,1])
        loss = 1 - tl.cost.dice_coe(logits2, tf.cast(annotation, dtype=tf.float32))


    total_var = tf.trainable_variables()
    # ========================================
    # To limit the training range
    # scope_name = 'inference'
    # trainable_var = [var for var in total_var if scope_name in var.name]
    # ========================================

    # Train all model
    trainable_var = total_var

    train_op = train(loss, trainable_var)


    # All the variables defined HERE -------------------------------
    dir_path = 'AP/Train'

    batch_size = 3

    opt_crop = False
    crop_shape = (224, 224)
    opt_resize = False
    resize_shape = (224, 224)
    rotation_status = True
    rotation_angle = [-5, 5]
    # --------------------------------------------------------------

    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(dir_path=dir_path, mode='training', rotation_status=rotation_status,
                                     rotation_angle=rotation_angle, batch_size=batch_size)
        iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(tr_data.data)

    sess = tf.Session()
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) # CPU ONLY

    print("Setting up Saver...")
    saver = tf.train.Saver()
    #summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        print("Start training")
        start = time.time()
        train_loss_list = []
        x_train = []
        validation_loss_list = []
        x_validation = []

        sess.run(training_init_op)

        # for itr in xrange(MAX_ITERATION):
        for itr in xrange(MAX_ITERATION): # about 12 hours of work / 2000
            train_images, train_annotations = sess.run(next_batch)

            # Reshape the annotation as the output (mask) has different dimension with the input
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.75}
            sess.run(train_op, feed_dict=feed_dict)

            if (itr+1) % 20 == 0:
                #train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_loss_list.append(train_loss)
                x_train.append(itr+1)
                #summary_writer.add_summary(summary_str, itr)


            end = time.time()
            print("Iteration #", itr+1, ",", np.int32(end - start), "s")

        saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr+1)

        # Draw loss functions
        plt.plot(x_train,train_loss_list,label='train')
        plt.title("loss functions")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.ylim(ymin=min(train_loss_list))
        plt.ylim(ymax=max(train_loss_list)*1.1)
        plt.legend()
        plt.savefig("loss_functions.png")

    # Need to add another mode to draw the contour based on image only.
    elif FLAGS.mode == "test":
        print("Setting up test data...")
        img_dir_name = '..\H&N_CTONLY'
        test_batch_size = 10
        test_index = 5
        ind = 0
        test_records = dicom_batchImage.read_DICOMbatchImage(dir_name=img_dir_name, opt_resize=opt_resize,
                                                             resize_shape=resize_shape, opt_crop=opt_crop, crop_shape=crop_shape)

        test_annotations = np.zeros([test_batch_size,224,224,1]) # fake input

        for index in range(test_index):
            print("Start creating data")
            test_images = test_records.next_batch(batch_size=test_batch_size)
            pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations, keep_probability: 1.0})
            pred = np.squeeze(pred, axis=3)

            print("Start saving data")
            for itr in range(test_batch_size):
                plt.subplot(121)
                plt.imshow(test_images[itr, :, :, 0], cmap='gray')
                plt.title("image")
                plt.subplot(122)
                plt.imshow(pred[itr], cmap='gray')
                plt.title("pred mask")
                plt.savefig(FLAGS.logs_dir + "/Prediction_test" + str(ind) + ".png")
                print("Test iteration : ", ind)
                ind += 1

    elif FLAGS.mode == "visualize":
        print("Setting up validation data...")
        validation_records = dicom_batch.read_DICOM(dir_name=dir_name + 'validation_set', dir_image=dir_image, dir_mask=dir_mask,
                                                    contour_name=contour_name, opt_resize=opt_resize, resize_shape=resize_shape,
                                                    opt_crop=opt_crop, crop_shape=crop_shape, rotation=False,
                                                    rotation_angle=rotation_angle, bitsampling=False,
                                                    bitsampling_bit=bitsampling_bit)

        dice_array = []
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Save the image for display. Use matplotlib to draw this.
        for itr in range(20):
            valid_images, valid_annotations = validation_records.next_batch(batch_size=1)

            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
            pred = np.squeeze(pred, axis=3)


            print(valid_images.shape, valid_annotations.shape, pred.shape)
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            dice_coeff = dice(valid_annotations[0], pred[0])

            dice_array.append(dice_coeff)
            print("min max of prediction : ", pred.flatten().min(), pred.flatten().max())
            print("min max of validation : ", valid_annotations.flatten().min(), valid_annotations.flatten().max())
            print("DICE : ", dice_coeff)
            print(valid_annotations.shape)


            # Save images
            plt.subplot(131)
            plt.imshow(valid_images[0, :, :, 0], cmap='gray')
            plt.title("image")
            plt.subplot(132)
            plt.imshow(valid_annotations[0,:,:], cmap='gray')
            plt.title("mask original")
            plt.subplot(133)
            plt.imshow(pred[0], cmap='gray')
            plt.title("mask predicted")
            plt.suptitle("DICE : " + str(dice_coeff))

            plt.savefig(FLAGS.logs_dir + "/Prediction_validation" + str(itr) + ".png")
            # plt.show()

        plt.figure()
        plt.hist(dice_array,bins)
        plt.xlabel('Dice')
        plt.ylabel('frequency')
        plt.title('Dice coefficient distribution of validation dataset')
        plt.savefig(FLAGS.logs_dir + "/dice histogram" + ".png")

if __name__ == "__main__":
    tf.app.run()
