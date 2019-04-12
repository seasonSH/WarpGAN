"""Main implementation class of WarpGAN
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import imp
import time
from functools import partial

import numpy as np
import tensorflow as tf


class WarpGAN:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes=None):
        '''
            Initialize the graph from scratch according to config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.images_A = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images_A')
                self.images_B = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images_B')
                self.labels_A = tf.placeholder(tf.int32, shape=[None], name='labels_A')
                self.labels_B = tf.placeholder(tf.int32, shape=[None], name='labels_B')
                self.scales_A = tf.placeholder(tf.float32, shape=[None], name='scales_A')
                self.scales_B = tf.placeholder(tf.float32, shape=[None], name='scales_B')

                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                self.setup_network_model(config, num_classes)

                # Build generator
                encode_A, styles_A = self.encoder(self.images_A)
                encode_B, styles_B = self.encoder(self.images_B)

                deform_BA, render_BA, ldmark_pred, ldmark_diff = self.decoder(encode_B, self.scales_B, None)
                render_AA = self.decoder(encode_A, self.scales_A, styles_A, texture_only=True)
                render_BB = self.decoder(encode_B, self.scales_B, styles_B, texture_only=True)

                self.styles_A = tf.identity(styles_A, name='styles_A')
                self.styles_B = tf.identity(styles_B, name='styles_B')
                self.deform_BA = tf.identity(deform_BA, name='deform_BA')
                self.ldmark_pred = tf.identity(ldmark_pred, name='ldmark_pred')
                self.ldmark_diff = tf.identity(ldmark_diff, name='ldmark_diff')


                # Build discriminator for real images
                patch_logits_A, logits_A = self.discriminator(self.images_A)
                patch_logits_B, logits_B = self.discriminator(self.images_B)
                patch_logits_BA, logits_BA = self.discriminator(deform_BA)                          

                # Show images in TensorBoard
                image_grid_A = tf.stack([self.images_A, render_AA], axis=1)[:1]
                image_grid_B = tf.stack([self.images_B, render_BB], axis=1)[:1]
                image_grid_BA = tf.stack([self.images_B, deform_BA], axis=1)[:1]
                image_grid = tf.concat([image_grid_A, image_grid_B, image_grid_BA], axis=0)
                image_grid = tf.reshape(image_grid, [-1] + list(self.images_A.shape[1:]))
                image_grid = self.image_grid(image_grid, (3,2))
                tf.summary.image('image_grid', image_grid)


                # Build all losses
                self.watch_list = {}
                loss_list_G  = []
                loss_list_D  = []
               
                # Advesarial loss for deform_BA
                loss_D, loss_G = self.cls_adv_loss(logits_A, logits_B, logits_BA,
                    self.labels_A, self.labels_B, self.labels_B, num_classes)
                loss_D, loss_G = config.coef_adv*loss_D, config.coef_adv*loss_G

                self.watch_list['LDg'] = loss_D
                self.watch_list['LGg'] = loss_G
                loss_list_D.append(loss_D)
                loss_list_G.append(loss_G)

                # Patch Advesarial loss for deform_BA
                loss_D, loss_G = self.patch_adv_loss(patch_logits_A, patch_logits_B, patch_logits_BA)
                loss_D, loss_G = config.coef_patch_adv*loss_D, config.coef_patch_adv*loss_G

                self.watch_list['LDp'] = loss_D
                self.watch_list['LGp'] = loss_G
                loss_list_D.append(loss_D)
                loss_list_G.append(loss_G)

                # Identity Mapping (Reconstruction) loss
                loss_idt_A = tf.reduce_mean(tf.abs(render_AA - self.images_A), name='idt_loss_A')
                loss_idt_A = config.coef_idt * loss_idt_A

                loss_idt_B = tf.reduce_mean(tf.abs(render_BB - self.images_B), name='idt_loss_B')
                loss_idt_B = config.coef_idt * loss_idt_B

                self.watch_list['idtA'] = loss_idt_A
                self.watch_list['idtB'] = loss_idt_B
                loss_list_G.append(loss_idt_A+loss_idt_B)


                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                self.watch_list['reg_loss'] = reg_loss
                loss_list_G.append(reg_loss)
                loss_list_D.append(reg_loss)


                loss_G = tf.add_n(loss_list_G, name='loss_G')
                grads_G = tf.gradients(loss_G, self.G_vars)

                loss_D = tf.add_n(loss_list_D, name='loss_D')
                grads_D = tf.gradients(loss_D, self.D_vars)

                # Training Operaters
                train_ops = []

                opt_G = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
                opt_D = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
                apply_G_gradient_op = opt_G.apply_gradients(list(zip(grads_G, self.G_vars)))
                apply_D_gradient_op = opt_D.apply_gradients(list(zip(grads_D, self.D_vars)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_G_gradient_op, apply_D_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Collect TF summary
                for k,v in self.watch_list.items():
                    tf.summary.scalar('losses/' + k, v)
                tf.summary.scalar('learning_rate', self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=99)
 
               

    def setup_network_model(self, config, num_classes):
        network_models = imp.load_source('network_models', config.network)
        self.encoder = partial(network_models.encoder, 
            style_size = config.style_size,
            keep_prob = self.keep_prob,
            phase_train = self.phase_train,
            weight_decay = config.weight_decay, 
            reuse=tf.AUTO_REUSE, scope = 'Encoder')
        self.decoder = partial(network_models.decoder, 
            style_size = config.style_size,
            image_size = config.image_size,
            keep_prob = self.keep_prob,
            phase_train = self.phase_train,
            weight_decay = config.weight_decay, 
            reuse=tf.AUTO_REUSE, scope = 'Decoder')
        self.discriminator = partial(network_models.discriminator,
            num_classes = 3*num_classes,
            bottleneck_size = config.bottleneck_size,
            keep_prob = self.keep_prob,
            phase_train = self.phase_train,
            weight_decay = config.weight_decay, 
            reuse=tf.AUTO_REUSE, scope = 'Discriminator')

        return

    @property
    def G_vars(self):
        vars_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        vars_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
        return vars_encoder + vars_decoder
        
    @property
    def D_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    def patch_adv_loss(self, logits_A, logits_B, logits_BA):
        with tf.name_scope('PatchAdvLoss'):
            labels_D_A = tf.zeros(tf.shape(logits_A)[0:1], dtype=tf.int32)
            labels_D_B = tf.ones(tf.shape(logits_B)[0:1], dtype=tf.int32)
            labels_D_BA = tf.ones(tf.shape(logits_BA)[0:1], dtype=tf.int32) * 2
            labels_G_BA = tf.zeros(tf.shape(logits_BA)[0:1], dtype=tf.int32)
            loss_D_A = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_A, labels=labels_D_A))
            loss_D_B = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_B, labels=labels_D_B))
            loss_D_BA = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_BA, labels=labels_D_BA))
            loss_G = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_BA, labels=labels_G_BA))
            loss_D = loss_D_A + loss_D_B + loss_D_BA

            return loss_D, loss_G
            
    def cls_adv_loss(self, logits_A, logits_B, logits_BA, labels_A, labels_B, labels_BA, num_classes):
        with tf.name_scope('ClsAdvLoss'):
            loss_D_A = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_A, labels=labels_A))
            loss_D_B = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_B, labels=labels_B+num_classes))
            loss_D_BA = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_BA, labels=labels_BA+2*num_classes))

            loss_D = loss_D_A + loss_D_B + loss_D_BA

            loss_G_BA = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=logits_BA, labels=labels_BA))

            loss_G = loss_G_BA

        return loss_D, loss_G
        
                
    def image_grid(self, images, size):
        m, n = size
        h, w, c = images.shape[1:4]
        h, w, c = h.value, w.value, c.value
        images = tf.reshape(images, [m, n, h, w, c])
        images = tf.transpose(images, [0, 2, 1, 3, 4])
        image_grid = tf.reshape(images, [1, m*h, n*w, c])
        return image_grid

    def save_model(self, model_dir, global_step):
        with self.sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...')
            self.saver.save(self.sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                self.saver.export_meta_graph(metagraph_path)

    def restore_model(self, model_dir, restore_scopes=None):
        var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with self.sess.graph.as_default():
            if restore_scopes is not None:
                var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
            model_dir = os.path.expanduser(model_dir)
            ckpt_file = tf.train.latest_checkpoint(model_dir)

            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            self.images_A = self.graph.get_tensor_by_name('images_A:0')
            self.images_B = self.graph.get_tensor_by_name('images_B:0')
            self.scales_A = self.graph.get_tensor_by_name('scales_A:0')
            self.scales_B = self.graph.get_tensor_by_name('scales_B:0')
            self.styles_A = self.graph.get_tensor_by_name('styles_A:0')
            self.styles_B = self.graph.get_tensor_by_name('styles_B:0')
            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.deform_BA = self.graph.get_tensor_by_name('deform_BA:0')
            # self.ldmark_pred = self.graph.get_tensor_by_name('ldmark_pred:0')
            # self.ldmark_diff = self.graph.get_tensor_by_name('ldmark_diff:0')
            self.input_style = self.graph.get_tensor_by_name('Decoder/StyleController/input_style:0')
            self.warp_input = self.graph.get_tensor_by_name('Decoder/WarpController/warp_input:0')



    def train(self, images_batch, labels_batch, switch_batch, learning_rate, keep_prob):
        images_A = images_batch[~switch_batch]
        images_B = images_batch[switch_batch]
        labels_A = labels_batch[~switch_batch]
        labels_B = labels_batch[switch_batch]
        scales_A = np.ones((images_A.shape[0]))
        scales_B = np.ones((images_B.shape[0]))
        feed_dict = {   self.images_A: images_A,
                        self.images_B: images_B,
                        self.labels_A: labels_A,
                        self.labels_B: labels_B,
                        self.scales_A: scales_A,
                        self.scales_B: scales_B,
                        self.learning_rate: learning_rate,
                        self.keep_prob: keep_prob,
                        self.phase_train: True,}
        _, wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict = feed_dict)

        step = self.sess.run(self.global_step)

        return wl, sm, step

    def generate_BA(self, images, scales, batch_size, styles=None, visualization=False):
        num_images = images.shape[0]
        h, w, c = tuple(self.deform_BA.shape[1:])
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        #ldmark_pred = np.ndarray((num_images, self.ldmark_pred.shape[1].value), dtype=np.float32)
        #ldmark_diff = np.ndarray((num_images, self.ldmark_pred.shape[1].value), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            indices = slice(start_idx, end_idx)
            images_B = images[indices]
            scales_B = scales[indices]
            feed_dict = {   self.images_B: images_B,          
                            self.scales_B: scales_B,
                            self.phase_train: False,
                            self.keep_prob: 1.0}
            if styles is not None:
                feed_dict[self.input_style] = styles[indices]

            if visualization:
                result[indices], ldmark_pred[indices], ldmark_diff[indices] = \
                     self.sess.run([self.deform_BA, self.ldmark_pred, self.ldmark_diff], feed_dict=feed_dict)
            else:
                result[indices] = self.sess.run(self.deform_BA, feed_dict=feed_dict)

        if visualization:
            return result, ldmark_pred, ldmark_diff
        else:
            return result


    def get_styles(self, images, batch_size):    
        num_images = images.shape[0]
        h, w, c = tuple(self.deform_BA.shape[1:])
        styles = np.ndarray((num_images, self.input_style.shape[1].value), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            indices = slice(start_idx, end_idx)
            images_B = images[indices]
            feed_dict = {self.images_B: images_B,           
                        self.phase_train: False,
                        self.keep_prob: 1.0}
            styles[indices] = self.sess.run(self.styles_B, feed_dict=feed_dict)

        return styles


