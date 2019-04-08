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

# import nntools.tensorflow.watcher as tfwatcher

from nets.sparse_image_warp import sparse_image_warp

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
    '92': ([4, 12, 24, 4], [64, 128, 256, 512]),
    '116': ([4, 16, 32, 4], [64, 128, 256, 512]),
}

batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
batch_norm_params_std = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': False,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}   

def parametric_relu(x):
    with tf.variable_scope('p_re_lu'):
        if x.shape.ndims >= 3:
            shape = (1, 1, x.get_shape()[-1])
        else:
            shape = (x.get_shape()[-1],)
        alphas = tf.get_variable('alpha', shape,
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * tf.minimum(x, 0)
    return pos + neg

def prelu_keras(x):
    if len(x.shape) == 4: 
        return tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
    else:
        return tf.keras.layers.PReLU().apply(x)

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

def conv_module(net, num_res_layers, num_kernels, trans_kernel_size=3, trans_stride=2,
                    reuse=None, scope=None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):

        net = slim.conv2d(net, num_kernels, kernel_size=trans_kernel_size, stride=trans_stride, padding='SAME',
                weights_initializer=slim.xavier_initializer())
        shortcut = net
        for i in range(num_res_layers):
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            print('| ---- block_%d' % i)
            net = net + shortcut
            shortcut = net
    return net


def discriminator(images, styles, num_classes, bottleneck_size=512, keep_probability=1.0, phase_train=True,
            weight_decay=0.0, model_version=None, reuse=None, scope_name='Discriminator'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=leaky_relu,
                        normalizer_fn=None,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope(scope_name, [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):

                print('{} input shape:'.format(scope_name), [dim.value for dim in images.shape])

                net =conv(images, 32, kernel_size=4, stride=2, scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])
                
                net = conv(net, 64, kernel_size=4, stride=2, scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])

                net = conv(net, 128, kernel_size=4, stride=2, scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])
 
                patch_logits = []

                def concat(net_map, style_vec):
                    h, w = net_map.shape[1].value, net_map.shape[2].value
                    style_vec = tf.reshape(style_vec, [-1,1,1,32])
                    style_map = tf.tile(style_vec, [1,h,w,1])
                    return tf.concat([net_map, style_map], axis=3)

                patch3_styles = slim.fully_connected(styles, 32, scope='patch3_styles')
                net_ = concat(net, patch3_styles)
                # patch3_logits = slim.conv2d(net, 3, 1, activation_fn=None, normalizer_fn=None, scope='patch3_logits')
                # patch_logits.append(tf.reshape(patch3_logits, [-1,3]))
             
                net = conv(net, 256, kernel_size=4, stride=2, scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])

                patch4_styles = slim.fully_connected(styles, 32, scope='patch4_styles')
                net_ = concat(net, patch4_styles)
                # patch4_logits = slim.conv2d(net, 3, 1, activation_fn=None, normalizer_fn=None, scope='patch4_logits')
                # patch_logits.append(tf.reshape(patch4_logits, [-1,3]))

                net = conv(net, 512, kernel_size=4, stride=2, scope='conv5')
                print('module_5 shape:', [dim.value for dim in net.shape])

                patch5_styles = slim.fully_connected(styles, 32, scope='patch5_styles')
                net_ = concat(net, patch5_styles)
                patch5_logits = slim.conv2d(net, 3, 1, activation_fn=None, normalizer_fn=None, scope='patch5_logits')
                patch_logits.append(tf.reshape(patch5_logits, [-1,3]))
              
                net = slim.flatten(net)
                prelogits = slim.fully_connected(net, bottleneck_size, scope='Bottleneck',
                                        weights_initializer=slim.xavier_initializer(), 
                                        activation_fn=None, normalizer_fn=None)
                prelogits = tf.nn.l2_normalize(prelogits, dim=1)
                print('latent shape:', [dim.value for dim in prelogits.shape])

                logits = slim.fully_connected(prelogits, num_classes, scope='Logits',
                                    activation_fn=None, normalizer_fn=None)

                return patch_logits, logits




def generator(images, scales, encoder_only=False, 
        style_only=False, random_style=True, no_shape=False, styles=None, 
        spherical_latent=False, keep_probability=1.0, phase_train=True, weight_decay=0.0, 
        reuse=None, model_version=None, scope_name='Generator'):
    with tf.variable_scope(scope_name, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        # weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected],
                    normalizer_fn=layer_norm, normalizer_params=None):

                    print('{} input shape:'.format(scope_name), [dim.value for dim in images.shape])

                    batch_size = tf.shape(images)[0]
                    h, w = (images.shape[1].value, images.shape[2].value)
                    k = 64


                    with tf.variable_scope('StyleNet'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                            normalizer_fn=None, normalizer_params=None):
                            
                            print('-- StyleNet')

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

                            style_vec = slim.fully_connected(net, 8, activation_fn=None, normalizer_fn=None, scope='fc1')
                            print('module fc1 shape:', [dim.value for dim in net.shape])
                            style_vec = tf.identity(style_vec, name='style_vec')

                            if spherical_latent:
                                style_vec = tf.nn.l2_normalize(style_vec, axis=1)

                            if style_only:
                                return style_vec

                            if random_style:
                                style_vec = tf.random_normal((batch_size, 8))

                                if spherical_latent:
                                    style_vec = tf.nn.l2_normalize(style_vec, axis=1)

                            if styles is not None:
                                net = styles
                            else:
                                net = style_vec
 
                            net = tf.identity(net, name='input_style')

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


                    #  Transform textures
                    with tf.variable_scope('Encoder'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_fn=instance_norm, normalizer_params=None):
                            print('-- Encoder')
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
                        
                    if encoder_only:
                        return encoded, style_vec

                    with tf.variable_scope('Decoder'):
                        print('-- Decoder')
                        net = encoded

                        final_params = {'activation_fn': None, 'normalizer_fn': None, 
                                    'weights_initializer': tf.constant_initializer(0.0)}

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

                        net = conv(net, 3, 7, pad=3, scope='conv_image', **final_params)
                        images_rendered = tf.nn.tanh(net, name='images_rendered')
                        print('images_rendered shape:', [dim.value for dim in images_rendered.shape])


                    if no_shape:
                        return images_rendered, style_vec

                    # Transform landmarks
                    with tf.variable_scope('ShapeNet'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_fn=layer_norm, normalizer_params=None):
                            print('-- StyleNet')

                            net = encoded

                            net = slim.flatten(net)

                            net = slim.fully_connected(net, 128, scope='fc1')
                            print('module fc1 shape:', [dim.value for dim in net.shape])

                            num_ldmark = 16

                            ldmark_mean = (np.random.normal(0,50, (num_ldmark,2)) + np.array([[0.5*h,0.5*w]])).flatten()

                            ldmark_mean = tf.Variable(ldmark_mean.astype(np.float32), name='ldmark_mean')
                            print('ldmark_mean shape:', [dim.value for dim in ldmark_mean.shape])

                            ldmark_pred = slim.fully_connected(net, num_ldmark*2, 
                                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                                normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='fc_ldmark')
                            ldmark_pred = ldmark_pred + ldmark_mean
                            print('ldmark_pred shape:', [dim.value for dim in ldmark_pred.shape])
                            ldmark_pred = tf.identity(ldmark_pred, name='ldmark_pred')
                     


                            ldmark_diff = slim.fully_connected(net, num_ldmark*2, 
                                # weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                                normalizer_fn=None,  activation_fn=None, scope='fc_diff')
                            print('ldmark_diff shape:', [dim.value for dim in ldmark_diff.shape])
                            ldmark_diff = tf.identity(ldmark_diff, name='ldmark_diff')
                            ldmark_diff = tf.identity(tf.reshape(scales,[-1,1]) * ldmark_diff, name='ldmark_diff_scaled')

                            src_pts = tf.reshape(ldmark_pred, [-1, num_ldmark ,2])
                            dst_pts = tf.reshape(ldmark_pred + ldmark_diff, [-1, num_ldmark, 2])


                            diff_norm = tf.reduce_mean(tf.norm(src_pts-dst_pts, axis=[1,2]))
                            tf.summary.scalar('diff_norm', diff_norm)
                            tf.summary.scalar('mark', ldmark_pred[0,0])

                            warp_input = tf.identity(images_rendered, name='warp_input')

                            images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts,
                                    regularization_weight = 1e-6, num_boundary_points=0)
                            dense_flow = tf.identity(dense_flow, name='dense_flow')

                            raw_images_transformed, _ = sparse_image_warp(images, src_pts, dst_pts,
                                    regularization_weight = 1e-6, num_boundary_points=0)
                            raw_images_transformed = tf.identity(raw_images_transformed, name='raw_images_transformed')

                    return images_transformed, images_rendered, style_vec, ldmark_pred, ldmark_diff

