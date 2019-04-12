"""Main training file for WarpGAN
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
import time
import imp
import argparse
import tensorflow as tf
import numpy as np

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from warpgan import WarpGAN


def test(network, config, log_dir, step):

    # Initialize testing
    if not hasattr(test, 'images'):
        testset = Dataset(config.test_dataset_path, prefix=config.data_prefix)
        random_indices = np.random.permutation(np.where(testset.is_photo)[0])[:64]
        test.images = testset.images[random_indices].astype(np.object)
        test.images = preprocess(test.images, config, is_training=False)

    output_dir = os.path.join(log_dir, 'samples')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # scales = np.indices((8,8), dtype=np.float32)[1] * 5
    scales = np.ones((8,8))
    scales = scales.flatten()
    test_results = network.generate_BA(test.images, scales, config.batch_size)
    utils.save_manifold(test_results, os.path.join(output_dir, '{}.jpg'.format(step)))
    
    
def main(args):

    # I/O
    config_file = args.config_file
    config = imp.load_source('config', config_file)
    if args.name:
        config.name = args.name

    trainset = Dataset(config.train_dataset_path, prefix=config.data_prefix)

    network = WarpGAN()
    network.initialize(config, trainset.num_classes)

    # Initalization for running
    if config.save_model:
        log_dir = utils.create_log_dir(config, config_file)
        summary_writer = tf.summary.FileWriter(log_dir, network.graph)
    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)

    proc_func = lambda images: preprocess(images, config, True)
    trainset.start_batch_queue(config.batch_size, proc_func=proc_func)


    # Main Loop
    print('\nStart Training\nname: {}\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n'.format(
            config.name, config.num_epochs, config.epoch_size, config.batch_size))
    global_step = 0
    start_time = time.time()
    for epoch in range(config.num_epochs):

        if epoch == 0: test(network, config, log_dir, global_step)

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()

            wl, sm, global_step = network.train(batch['images'], batch['labels'], batch['is_photo'], learning_rate, config.keep_prob)

            wl['lr'] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                if config.save_model:
                    summary_writer.add_summary(sm, global_step=global_step)

        # Testing
        test(network, config, log_dir, global_step)

        # Save the model
        if config.save_model:
            network.save_model(log_dir, global_step)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    parser.add_argument("--name", help="Rename the log dir",
                        type=str, default=None)
    args = parser.parse_args()
    main(args)
