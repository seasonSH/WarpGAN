""" Utility functions for Tensorflow operations
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim



def image_grid(images, size):
    m, n = size
    h, w, c = images.shape[1:4]
    h, w, c = h.value, w.value, c.value
    images = tf.reshape(images, [m, n, h, w, c])
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    image_grid = tf.reshape(images, [1, m*h, n*w, c])
    return image_grid

def average_tensors(tensors, name=None):
    if len(tensors) == 1:
        return tf.identity(tensors[0], name=name)
    else:
        # Each tensor in the list should be of the same size
        expanded_tensors = []

        for t in tensors:
            expanded_t = tf.expand_dims(t, 0)
            expanded_tensors.append(expanded_t)

        average_tensor = tf.concat(axis=0, values=expanded_tensors)
        average_tensor = tf.reduce_mean(average_tensor, 0, name=name)

        return average_tensor


def average_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of gradients. The outer list is over different 
        towers. The inner list is over the gradient calculation in each tower.
    Returns:
        List of gradients where the gradient has been averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        average_grads = []
        for grad_ in zip(*tower_grads):
            # Note that each grad looks like the following:
            #   (grad0_gpu0, ... , grad0_gpuN)
            average_grad = None if grad_[0]==None else average_tensors(grad_)
            average_grads.append(average_grad)

        return average_grads


def save_model(sess, saver, model_dir, global_step):
    with sess.graph.as_default():
        checkpoint_path = os.path.join(model_dir, 'ckpt')
        metagraph_path = os.path.join(model_dir, 'graph.meta')

        print('Saving variables...')
        saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
        if not os.path.exists(metagraph_path):
            print('Saving metagraph...')
            saver.export_meta_graph(metagraph_path)

def restore_model(sess, var_list, model_dir, restore_scopes=None, replace=None):
    ''' Load the variable values from a checkpoint file into pre-defined graph.
    Filter the variables so that they contain at least one of the given keywords.'''
    with sess.graph.as_default():
        if restore_scopes is not None:
            var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
        if replace is not None:
            var_dict = {}
            for var in var_list:
                name_new = var.name
                for k,v in replace.items(): name_new=name_new.replace(k,v)
                name_new = name_new[:-2] # When using dict, numbers should be removed
                var_dict[name_new] = var
            var_list = var_dict
        model_dir = os.path.expanduser(model_dir)
        ckpt_file = tf.train.latest_checkpoint(model_dir)

        print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)

def load_model(sess, model_path, scope=None):
    ''' Load the the graph and variables values from a model path.
    Model path is either a a frozen graph or a directory with both
    a .meta file and checkpoint files.'''
    with sess.graph.as_default():
        model_path = os.path.expanduser(model_path)
        if (os.path.isfile(model_path)):
            # Frozen grpah
            print('Model filename: %s' % model_path)
            with gfile.FastGFile(model_path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(sess, ckpt_file)


