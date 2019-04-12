# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import atanh
from functools import partial
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import instance_norm, layer_norm

from models.sparse_image_warp import sparse_image_warp


batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
 

def leaky_relu(x):
    return tf.maximum(0.2*x, x)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def padding(x, pad, pad_type='reflect'):
    if pad_type == 'zero' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == 'reflect' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
    else:
        raise ValueError('Unknown pad type: {}'.format(pad_type))

def conv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)

def deconv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d_transpose(x, *args, **kwargs)


def discriminator(images, num_classes, bottleneck_size=512, keep_prob=1.0, phase_train=True,
            weight_decay=0.0, reuse=None, scope='Discriminator'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=leaky_relu,
                        normalizer_fn=None,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope(scope, [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):

                print('{} input shape:'.format(scope), [dim.value for dim in images.shape])

                net =conv(images, 32, kernel_size=4, stride=2, scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])
                
                net = conv(net, 64, kernel_size=4, stride=2, scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])

                net = conv(net, 128, kernel_size=4, stride=2, scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])
 
             
                net = conv(net, 256, kernel_size=4, stride=2, scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])

                net = conv(net, 512, kernel_size=4, stride=2, scope='conv5')
                print('module_5 shape:', [dim.value for dim in net.shape])


                # Patch Discrminator
                patch5_logits = slim.conv2d(net, 3, 1, activation_fn=None, normalizer_fn=None, scope='patch5_logits')
                patch_logits = tf.reshape(patch5_logits, [-1,3])

              
                # Global Discriminator
                net = slim.flatten(net)
                prelogits = slim.fully_connected(net, bottleneck_size, scope='Bottleneck',
                                        weights_initializer=slim.xavier_initializer(), 
                                        activation_fn=None, normalizer_fn=None)
                prelogits = tf.nn.l2_normalize(prelogits, dim=1)
                print('latent shape:', [dim.value for dim in prelogits.shape])

                logits = slim.fully_connected(prelogits, num_classes, scope='Logits',
                                    activation_fn=None, normalizer_fn=None)

                return patch_logits, logits


def encoder(images, style_size=8, keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Encoders'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        # weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected],
                    normalizer_fn=layer_norm, normalizer_params=None):
                    print('{} input shape:'.format(scope), [dim.value for dim in images.shape])

                    batch_size = tf.shape(images)[0]
                    k = 64


                    with tf.variable_scope('StyleEncoder'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                            normalizer_fn=None, normalizer_params=None):
                            
                            print('-- StyleEncoder')

                            net = images

                            net = conv(net, k, 7, stride=1, pad=3, scope='conv0')
                            print('module conv0 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 2*k, 4, stride=2, scope='conv1')
                            print('module conv1 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 4*k, 4, stride=2, scope='conv2')
                            print('module conv2 shape:', [dim.value for dim in net.shape])
     

                            encoded_style = net

                            net = slim.avg_pool2d(net, net.shape[1:3], padding='VALID', scope='global_pool')
                            net = slim.flatten(net)

                            style_vec = slim.fully_connected(net, style_size, activation_fn=None, normalizer_fn=None, scope='fc1')
                            print('module fc1 shape:', [dim.value for dim in net.shape])
                            style_vec = tf.identity(style_vec, name='style_vec')


                    #  Transform textures
                    with tf.variable_scope('ContentEncoder'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_fn=instance_norm, normalizer_params=None):
                            print('-- ContentEncoder')
                            net = images

                            net = conv(net, k, 7, stride=1, pad=3, scope='conv0')
                            print('module conv0 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 2*k, 4, stride=2, scope='conv1')
                            print('module conv1 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 4*k, 4, stride=2, scope='conv2')
                            print('module conv2 shape:', [dim.value for dim in net.shape])
                            
                            for i in range(3):
                                net_ = conv(net, 4*k, 3, scope='res{}_0'.format(i))
                                net += conv(net_, 4*k, 3, activation_fn=None, biases_initializer=None, scope='res{}_1'.format(i))
                                print('module res{} shape:'.format(i), [dim.value for dim in net.shape])

                            encoded = net
                        
                    return encoded, style_vec


def decoder(encoded, scales, styles, texture_only=False, style_size=8, image_size=(112,112),
        keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Decoder'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        # weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected],
                    normalizer_fn=layer_norm, normalizer_params=None):
                    print('{} input shape:'.format(scope), [dim.value for dim in encoded.shape])
                        
                    batch_size = tf.shape(encoded)[0]
                    h, w = tuple(image_size)
                    k = 64
    
                    with tf.variable_scope('StyleController'):

                        if styles is None:
                            styles = tf.random_normal((batch_size, style_size))

                        net = tf.identity(styles, name='input_style')

                        net = slim.fully_connected(net, 128, scope='fc2')
                        print('module fc2 shape:', [dim.value for dim in net.shape])

                        net = slim.fully_connected(net, 128, scope='fc3')
                        print('module fc3 shape:', [dim.value for dim in net.shape])

                        gamma = slim.fully_connected(net, 4*k, activation_fn=None, normalizer_fn=None, scope='fc4')
                        gamma = tf.reshape(gamma, [-1, 1, 1, 4*k], name='gamma')
                        print('gamma shape:', [dim.value for dim in gamma.shape])

                        beta = slim.fully_connected(net, 4*k, activation_fn=None, normalizer_fn=None, scope='fc5')
                        beta = tf.reshape(beta, [-1, 1, 1, 4*k], name='beta')
                        print('beta shape:', [dim.value for dim in beta.shape])


                    
                    with tf.variable_scope('Decoder'):
                        print('-- Decoder')
                        net = encoded

                        adain = lambda x : gamma * instance_norm(x, center=False, scale=False) + beta

                        with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                                    normalizer_fn=adain, normalizer_params=None):
                            for i in range(3):
                                net_ = conv(net, 4*k, 3, scope='res{}_0'.format(i))
                                net += conv(net_, 4*k, 3, activation_fn=None, biases_initializer=None, scope='res{}_1'.format(i))
                                print('module res{} shape:'.format(i), [dim.value for dim in net.shape])

               
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_fn=layer_norm, normalizer_params=None):
                            net = upscale2d(net, 2)
                            net = conv(net, 2*k, 5, pad=2, scope='deconv1_1')
                            print('module deconv1 shape:', [dim.value for dim in net.shape])

                            net = upscale2d(net, 2)
                            net = conv(net, k, 5, pad=2, scope='deconv2_1')

                        net = conv(net, 3, 7, pad=3, activation_fn=None, normalizer_fn=None, 
                                    weights_initializer=tf.constant_initializer(0.0), scope='conv_image')
                        images_rendered = tf.nn.tanh(net, name='images_rendered')
                        print('images_rendered shape:', [dim.value for dim in images_rendered.shape])

                    if texture_only:
                        return images_rendered                        

                    with tf.variable_scope('WarpController'):

                        print('-- WarpController')

                        net = encoded
                        warp_input = tf.identity(images_rendered, name='warp_input')

                        net = slim.flatten(net)

                        net = slim.fully_connected(net, 128, scope='fc1')
                        print('module fc1 shape:', [dim.value for dim in net.shape])

                        num_ldmark = 16

                        # Predict the control points
                        ldmark_mean = (np.random.normal(0,50, (num_ldmark,2)) + np.array([[0.5*h,0.5*w]])).flatten()
                        ldmark_mean = tf.Variable(ldmark_mean.astype(np.float32), name='ldmark_mean')
                        print('ldmark_mean shape:', [dim.value for dim in ldmark_mean.shape])

                        ldmark_pred = slim.fully_connected(net, num_ldmark*2, 
                            weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                            normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='fc_ldmark')
                        ldmark_pred = ldmark_pred + ldmark_mean
                        print('ldmark_pred shape:', [dim.value for dim in ldmark_pred.shape])
                        ldmark_pred = tf.identity(ldmark_pred, name='ldmark_pred')
                 

                        # Predict the displacements
                        ldmark_diff = slim.fully_connected(net, num_ldmark*2, 
                            normalizer_fn=None,  activation_fn=None, scope='fc_diff')
                        print('ldmark_diff shape:', [dim.value for dim in ldmark_diff.shape])
                        ldmark_diff = tf.identity(ldmark_diff, name='ldmark_diff')
                        ldmark_diff = tf.identity(tf.reshape(scales,[-1,1]) * ldmark_diff, name='ldmark_diff_scaled')



                        src_pts = tf.reshape(ldmark_pred, [-1, num_ldmark ,2])
                        dst_pts = tf.reshape(ldmark_pred + ldmark_diff, [-1, num_ldmark, 2])

                        diff_norm = tf.reduce_mean(tf.norm(src_pts-dst_pts, axis=[1,2]))
                        # tf.summary.scalar('diff_norm', diff_norm)
                        # tf.summary.scalar('mark', ldmark_pred[0,0])

                        images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts,
                                regularization_weight = 1e-6, num_boundary_points=0)
                        dense_flow = tf.identity(dense_flow, name='dense_flow')

                return images_transformed, images_rendered, ldmark_pred, ldmark_diff
 

