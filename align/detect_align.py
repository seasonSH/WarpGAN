"""Align face images given landmarks."""

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import os
import warnings
import argparse
import random
import cv2

from align.mtcnntf.detector import Detector
from align.matlab_cp2tform import get_similarity_transform_for_cv2


def align(src_img, src_pts, ref_pts, image_size, scale=1.0, transpose_input=False):
    w, h = image_size = tuple(image_size)

    # Actual offset = new center - old center (scaled)
    scale_ = max(w,h) * scale
    cx_ref = cy_ref = 0.
    offset_x = 0.5 * w - cx_ref * scale_
    offset_y = 0.5 * h - cy_ref * scale_

    s = np.array(src_pts).astype(np.float32).reshape([-1,2])
    r = np.array(ref_pts).astype(np.float32) * scale_ + np.array([[offset_x, offset_y]])
    if transpose_input: 
        s = s.reshape([2,-1]).T

    tfm = get_similarity_transform_for_cv2(s, r)
    dst_img = cv2.warpAffine(src_img, tfm, image_size)

    s_new = np.concatenate([s.reshape([2,-1]), np.ones((1, s.shape[0]))])
    s_new = np.matmul(tfm, s_new)
    s_new = s_new.reshape([-1]) if transpose_input else s_new.T.reshape([-1]) 
    tfm = tfm.reshape([-1])
    return dst_img, s_new, tfm


def detect_align(image, image_size=(256,256), scale=0.7, transpose_input=False):

    detector = Detector()
    
    bboxes, landmarks = detector.detect(image)
    if len(bboxes) == 0 : return None
    elif len(bboxes) > 1:
        img_size = image.shape[:2]
        bbox_size = bboxes[:,2] * bboxes[:,3]
        img_center = img_size / 2
        offsets = np.vstack([ bboxes[:,0]+0.5*bboxes[:,2]-img_center[1], bboxes[:,1]+0.5*bboxes[:,3]-img_center[0] ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(offset_dist_squared*2.0) # some extra weight on the centering
        bboxes = bboxes[index][None]
        landmarks = landmarks[index][None]
   
    src_pts = landmarks[0]
    ref_pts = np.array( [[ -1.58083929e-01, -3.84258929e-02],
                         [  1.56533929e-01, -4.01660714e-02],
                         [  2.25000000e-04,  1.40505357e-01],
                         [ -1.29024107e-01,  3.24691964e-01],
                         [  1.31516964e-01,  3.23250893e-01]])

    img_new, new_pts, tfm = align(image, src_pts, ref_pts, image_size, scale, transpose_input)

    return img_new
    
