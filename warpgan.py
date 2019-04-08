"""Main training file for face recognition
"""

import os
import sys
import imp
import time
from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import tflib


class WarpGAN:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes=None):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.images_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images_batch')
                self.labels_placeholder = tf.placeholder(tf.int32, shape=[None], name='labels_batch')
                self.scales_placeholder = tf.placeholder(tf.float32, shape=[None], name='scales_batch')
                self.switch_placeholder = tf.placeholder(tf.bool, shape=[None], name='switch_batch')
                self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                self.setup_network_model(config, num_classes)

                images_splits = tf.split(self.images_placeholder, config.num_gpus)
                labels_splits = tf.split(self.labels_placeholder, config.num_gpus)
                scales_splits = tf.split(self.scales_placeholder, config.num_gpus)
                switch_splits = tf.split(self.switch_placeholder, config.num_gpus)

                grads_G_splits = []
                grads_D_splits = []
                average_dict = {}
                def insert_dict(k,v):
                    if k in average_dict: average_dict[k].append(v)
                    else: average_dict[k] = [v]
                        
                for device in range(config.num_gpus):
                    scope_name = 'device_{}'.format(device)
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=device>0):
                            with tf.device('/gpu:{}'.format(device)):

                                # Collect input tensors for this branch
                                images = tf.identity(images_splits[device], name='images')
                                labels = tf.identity(labels_splits[device], name='labels')
                                scales = tf.identity(scales_splits[device], name='scales')
                                switch = tf.identity(switch_splits[device], name='switch')

                                images_A = tf.identity(tf.boolean_mask(images, tf.logical_not(switch)), name="images_A")
                                images_B = tf.identity(tf.boolean_mask(images, switch), name="images_B")
                                labels_A = tf.identity(tf.boolean_mask(labels, tf.logical_not(switch)), name="labels_A")
                                labels_B = tf.identity(tf.boolean_mask(labels, switch), name="labels_B")
                                scales_A = tf.identity(tf.boolean_mask(scales, tf.logical_not(switch)), name="scales_A")
                                scales_B = tf.identity(tf.boolean_mask(scales, switch), name="scales_B")


                                # Build generator

                                encode_A, styles_A = self.encoder(images_A)
                                encode_B, styles_B = self.encoder(images_B)

                                deform_BA, render_BA, ldmark_pred, ldmark_diff = self.decoder(encode_B, scales_B, None)
                                render_AA = self.decoder(encode_A, scales_A, styles_A, texture_only=True)
                                render_BB = self.decoder(encode_B, scales_B, styles_B, texture_only=True)

                                styles_A = tf.identity(styles_A, name='styles_A')
                                styles_B = tf.identity(styles_B, name='styles_B')
                                deform_BA = tf.identity(deform_BA, name='deform_BA')
                                render_BA = tf.identity(render_BA, name='render_BA')
                                render_AA = tf.identity(render_AA, name='render_AA')
                                render_BB = tf.identity(render_BB, name='render_BB')

 
                                # Build discriminator for real images
                                patch_logits_A, logits_A = self.discriminator(images_A)
                                patch_logits_B, logits_B = self.discriminator(images_B)
                                patch_logits_BA, logits_BA = self.discriminator(deform_BA)                          

                                if device == 0:
                                    image_grid_A = tf.stack([images_A, render_AA], axis=1)[:1]
                                    image_grid_B = tf.stack([images_B, render_BB], axis=1)[:1]
                                    image_grid_BA = tf.stack([images_B, deform_BA], axis=1)[:1]
                                    image_grid = tf.concat([image_grid_A, image_grid_B, image_grid_BA], axis=0)
                                    image_grid = tf.reshape(image_grid, [-1] + list(images_A.shape[1:]))
                                    image_grid = tflib.image_grid(image_grid, (3,2))
                                    tf.summary.image('image_grid', image_grid)


                                # Build all losses
                                loss_list_G  = []
                                loss_list_D  = []
                               
                                # Advesarial loss for deform_BA
                                if config.losses['coef_adv'] > 0:

                                    loss_D, loss_G = self.cls_adv_loss(logits_A, logits_B, logits_BA,
                                        labels_A, labels_B, labels_B, num_classes)

                                    coef_adv = config.losses['coef_adv']
                                    loss_D, loss_G = coef_adv*loss_D, coef_adv*loss_G

                                    insert_dict('Dim', loss_D)
                                    insert_dict('Gim', loss_G)
                                    loss_list_D.append(loss_D)
                                    loss_list_G.append(loss_G)

                                # Patch Advesarial loss for deform_BA
                                if config.losses['coef_patch_adv'] > 0:
            
                                    loss_D, loss_G = self.patch_adv_loss(patch_logits_A, patch_logits_B, patch_logits_BA)

                                    coef_patch_adv = config.losses['coef_patch_adv']
                                    loss_D, loss_G = coef_patch_adv*loss_D, coef_patch_adv*loss_G

                                    insert_dict('Dpa', loss_D)
                                    insert_dict('Gpa', loss_G)
                                    loss_list_D.append(loss_D)
                                    loss_list_G.append(loss_G)

                                # Identity loss
                                if config.losses['coef_idt'] > 0:
                                    loss_idt_A = tf.reduce_mean(tf.abs(render_AA - images_A), name='idt_loss_A')
                                    loss_idt_A =  config.losses['coef_idt'] * loss_idt_A

                                    loss_idt_B = tf.reduce_mean(tf.abs(render_BB - images_B), name='idt_loss_B')
                                    loss_idt_B =  config.losses['coef_idt'] * loss_idt_B

                                    insert_dict('idtA', loss_idt_A)
                                    insert_dict('idtB', loss_idt_B)
                                    loss_list_G.append(loss_idt_A+loss_idt_B)


                                # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                insert_dict('reg_loss', reg_loss)
                                loss_list_G.append(reg_loss)
                                loss_list_D.append(reg_loss)


                                loss_G = tf.add_n(loss_list_G, name='loss_G')
                                grads_G_split = tf.gradients(loss_G, self.G_vars)
                                grads_G_splits.append(grads_G_split)

                                loss_D = tf.add_n(loss_list_D, name='loss_D')
                                grads_D_split = tf.gradients(loss_D, self.D_vars)
                                grads_D_splits.append(grads_D_split)

                                # Keep some useful tensor in GPU_0
                                if device == 0:
                                    self.inputs = images
                                    self.switch = switch
                                    self.images_A = images_A
                                    self.images_B = images_B
                                    self.scales_A = scales_A
                                    self.scales_B = scales_B
                                    self.deform_BA = deform_BA
                                    self.ldmark_pred = ldmark_pred
                                    self.ldmark_diff = ldmark_diff


                # Merge the splits
                self.watch_list = {}
                grads_G = tflib.average_grads(grads_G_splits)
                grads_D = tflib.average_grads(grads_D_splits)
                for k,v in average_dict.items():
                    v = tflib.average_tensors(v)
                    average_dict[k] = v
                    self.watch_list[k] = v
                    if 'loss' in k:
                        tf.summary.scalar('losses/' + k, v)
                    else:
                        tf.summary.scalar(k, v)

            # Training Operaters
            train_ops = []

            opt_G = tf.train.AdamOptimizer(self.learning_rate_placeholder, beta1=0.5, beta2=0.9)
            opt_D = tf.train.AdamOptimizer(self.learning_rate_placeholder, beta1=0.5, beta2=0.9)
            apply_G_gradient_op = opt_G.apply_gradients(list(zip(grads_G, self.G_vars)))
            apply_D_gradient_op = opt_D.apply_gradients(list(zip(grads_D, self.D_vars)))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_ops.extend([apply_G_gradient_op, apply_D_gradient_op] + update_ops)

            '''
            apply_G_gradient_op = tflib.apply_gradient(self.G_vars, grads_G, config.optimizer,
                                    self.learning_rate_placeholder, config.learning_rate_multipliers)
            update_G_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if any([sc in op.name for sc in self.scopes_G])]
            train_ops.extend([apply_G_gradient_op] + update_G_ops)

            apply_D_gradient_op = tflib.apply_gradient(self.D_vars, grads_D, config.optimizer,
                                    self.learning_rate_placeholder, config.learning_rate_multipliers)
            update_D_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if any([sc in op.name for sc in self.scopes_D])]
            train_ops.extend([apply_D_gradient_op] + update_D_ops)
            '''

            train_ops.append(tf.assign_add(self.global_step, 1))
            self.train_op = tf.group(*train_ops)

            tf.summary.scalar('learning_rate', self.learning_rate_placeholder)
            self.summary_op = tf.summary.merge_all()

            # Initialize variables
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=99)
 
               

    def setup_network_model(self, config, num_classes):
        network_models = imp.load_source('network_models', config.network)
        self.encoder = partial(network_models.encoder, 
            style_size = config.style_size,
            phase_train = self.phase_train_placeholder,
            weight_decay = config.weight_decay, 
            reuse=tf.AUTO_REUSE, scope = 'Encoder')
        self.decoder = partial(network_models.decoder, 
            style_size = config.style_size,
            image_size = config.image_size,
            phase_train = self.phase_train_placeholder,
            weight_decay = config.weight_decay, 
            reuse=tf.AUTO_REUSE, scope = 'Decoder')
        self.discriminator = partial(network_models.discriminator,
            num_classes = 3*num_classes,
            bottleneck_size = config.bottleneck_size,
            phase_train = self.phase_train_placeholder,
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
            labels_G_BA= tf.zeros(tf.shape(logits_BA)[0:1], dtype=tf.int32)
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
        
                

    def train(self, images_batch, labels_batch, switch_batch, learning_rate, keep_prob):
        batch_size = images_batch.shape[0]
        scales_batch = np.ones((batch_size,))
        feed_dict = {self.images_placeholder: images_batch,
                    self.labels_placeholder: labels_batch,
                    self.scales_placeholder: scales_batch,
                    self.switch_placeholder: switch_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,}
        _, wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict = feed_dict)

        step = self.sess.run(self.global_step)

        return wl, sm, step

    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tflib.restore_model(self.sess, trainable_variables, *args, **kwargs)


    def save_model(self, model_dir, global_step):
        tflib.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, model_path, *args, **kwargs):
        tflib.load_model(self.sess, model_path, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('device_0/images:0')
        self.switch = self.graph.get_tensor_by_name('device_0/switch:0')
        self.images_A = self.graph.get_tensor_by_name('device_0/images_A:0')
        self.images_B = self.graph.get_tensor_by_name('device_0/images_B:0')
        self.scales_A = self.graph.get_tensor_by_name('device_0/scales_A:0')
        self.scales_B = self.graph.get_tensor_by_name('device_0/scales_B:0')
        self.styles_A = self.graph.get_tensor_by_name('device_0/styles_A:0')
        self.styles_B = self.graph.get_tensor_by_name('device_0/styles_B:0')
        self.deform_BA = self.graph.get_tensor_by_name('device_0/deform_BA:0')
        self.input_style = self.graph.get_tensor_by_name('device_0/Decoder/StyleController/input_style:0')
        self.warp_input = self.graph.get_tensor_by_name('device_0/Decoder/WarpController/warp_input:0')
        self.ldmark_pred = self.graph.get_tensor_by_name('device_0/Decoder/WarpController/ldmark_pred:0')
        self.ldmark_diff = self.graph.get_tensor_by_name('device_0/Decoder/WarpController/ldmark_diff:0')


    def generate_BA(self, images, scales, batch_size, styles=None, with_texture=True, visualization=False):
        num_images = images.shape[0]
        h, w, c = tuple(self.deform_BA.shape[1:])
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        ldmark_pred = np.ndarray((num_images, self.ldmark_pred.shape[1].value), dtype=np.float32)
        ldmark_diff = np.ndarray((num_images, self.ldmark_pred.shape[1].value), dtype=np.float32)
        output_image = self.deform_BA if with_texture else self.raw_images_transformed
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            indices = slice(start_idx, end_idx)
            inputs = images[indices]
            scales_batch = scales[indices]
            feed_dict = {self.images_B: inputs,          
                        self.scales_B: scales_batch,
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            if styles is not None:
                feed_dict[self.input_style] = styles[indices]
            if visualization:
                result[indices], ldmark_pred[indices], ldmark_diff[indices] = \
                     self.sess.run([output_image, self.ldmark_pred, self.ldmark_diff], feed_dict=feed_dict)
            else:
                result[indices] = self.sess.run(output_image, feed_dict=feed_dict)

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
            inputs = images[indices]
            feed_dict = {self.images_B: inputs,           
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            styles[indices] = self.sess.run(self.styles_B, feed_dict=feed_dict)

        return styles

    def extract_feature(self, images, switch, batch_size, proc_func=None, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.latent_A.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            switch_batch = switch[start_idx:end_idx]
            result_temp = np.ndarray((end_idx-start_idx, num_features), dtype=np.float32)
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            if proc_func is not None:
                inputs = proc_func(inputs)
            feed_dict = {
                    self.inputs: inputs,
                    self.switch: switch_batch,
                    self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            if not np.all(switch_batch):
                result_A = self.sess.run(self.latent_AB, feed_dict=feed_dict)
                result_temp[~switch_batch,:] = result_A
            if np.any(switch_batch):
                result_B = self.sess.run(self.latent_B, feed_dict=feed_dict)
                result_temp[switch_batch,:] = result_B
            result[start_idx:end_idx] = result_temp
        return result

