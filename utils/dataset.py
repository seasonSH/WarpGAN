"""Data fetching
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

import sys
import os
import time
import math
import random
import shutil
from multiprocessing import Process, Queue

import h5py
import numpy as np

is_photo = lambda x: os.path.basename(x).startswith('P')

class DataClass(object):
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = np.array(indices)
        self.label = label
        return

    def random_pc_pair(self):
        photo_idx = np.random.permutation(self.photo_indices)[0]
        caric_idx = np.random.permutation(self.caric_indices)[0]
        return np.array([photo_idx, caric_idx])


class Dataset():

    def __init__(self, path=None, prefix=None):
        self.DataClass = DataClass
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.is_photo = None
        self.idx2cls = None
        self.batch_queue = None
        self.batch_workers = None

        if path is not None:
            self.init_from_list(path, prefix)

    def init_from_list(self, filename, prefix=None):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines)>0, \
            'List file must be in format: "fullpath(str) label(int)"'

        images = [line[0] for line in lines]
        if prefix is not None:
            print('Adding prefix: {}'.format(prefix))
            images = [os.path.join(prefix, img) for img in images]

        if len(lines[0]) > 1:
            labels = [int(line[1]) for line in lines]
        else:
            labels = [os.path.dirname(img) for img in images]
            _, labels = np.unique(labels, return_inverse=True)

        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))
        self.separate_photo_caricature()

    def separate_photo_caricature(self):
        self.is_photo = [is_photo(im) for im in self.images]
        self.is_photo = np.array(self.is_photo, dtype=np.bool)
        for c in self.classes:
            c.photo_indices = c.indices[self.is_photo[c.indices]]
            c.caric_indices = c.indices[~self.is_photo[c.indices]]
        print('{} photos {} caricatures'.format(self.is_photo.sum(), (~self.is_photo).sum()))
        return
              
    def init_classes(self):
        dict_classes = {}
        classes = []
        self.idx2cls = np.ndarray((len(self.labels),)).astype(np.object)    
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(self.DataClass(str(label), indices, label))
            self.idx2cls[indices] = classes[-1]
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)

    def build_subset_from_indices(self, indices, new_labels=True):
        subset = type(self)()
        subset.images = self.images[indices]
        subset.labels = self.labels[indices]
        if new_labels:
            _, subset.labels = np.unique(subset.labels, return_inverse=True)
        subset.init_classes()

        print('built subset: %d images of %d classes' % (len(subset.images), subset.num_classes))
        return subset


    # Data Loading

    def get_batch(self, batch_size):
        ''' Get random pairs of photos and caricatures. '''
        indices_batch = []
        
        # Random photo-caricature pair
        assert batch_size%2 == 0
        classes = np.random.permutation(self.classes)[:batch_size//2]
        indices_batch = np.concatenate([c.random_pc_pair() for c in classes], axis=0)

        batch = {}
        if len(indices_batch) > 0:
            batch['images'] = self.images[indices_batch]
            batch['labels'] = self.labels[indices_batch]
            if self.is_photo is not None:
                batch['is_photo'] = self.is_photo[indices_batch]

        return batch

    # Multithreading preprocessing images
    def start_batch_queue(self, batch_size, proc_func=None, maxsize=1, num_threads=3):

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                batch = self.get_batch(batch_size)
                if proc_func is not None:
                    batch['image_paths'] = batch['images']
                    batch['images'] = proc_func(batch['image_paths'])
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=60):
        return self.batch_queue.get(block=True, timeout=timeout)
      
    def release_queue(self):
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None




