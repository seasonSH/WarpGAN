import tensorflow as tf
from . import detect_face

class Detector(object):
    def __init__(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(allow_growth=True)
            tf_config = tf.ConfigProto(gpu_options=gpu_options,
                    allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=tf_config)
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
    
        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
        self.factor = 0.85 # scale factor

    
    def detect(self, img):

        bboxes, landmarks = detect_face.detect_face(img,
            self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        bboxes = bboxes[:,:4]
        bboxes[:,2:4] = bboxes[:,2:4] - bboxes[:,:2]
        landmarks = landmarks.reshape([2,5,-1]).transpose([2,1,0]).reshape([-1,10])

        return bboxes, landmarks
