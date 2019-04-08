''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'warpgan_lr'

# The folder to save log and model
log_base_dir = './log/'

# Whether to save the model checkpoints and result logs
save_model = True

# The interval between writing summary
summary_interval = 100

# Prefix to the image files
data_prefix = os.environ["DATABASES2"] + "/caricature/WebCaricature/webcaric_5ptaligned_sc0.7_256/"

# Training dataset path
train_dataset_path = "./data/train.txt"

# Test dataset path
test_dataset_path = "./data/test.txt"

# Target image size (h,w) for the input of network
image_size = (256, 256)

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    ['random_flip'],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1


####### NETWORK #######

# The network architecture
network = 'nets/sphere_conv4.py'

# Model version, only for some networks
model_version = "4"

# Number of dimensions in the embedding space
embedding_size = 512

# Use hyper-spherical latent distribution instead of Gaussian
spherical_latent = False

####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("ADAM", {'beta1': 0.5, 'beta2': 0.9})
# optimizer = ("MOM", {'momentum': 0.9})

# Number of samples per batch
batch_size = 2

# The structure of the batch
batch_format = 'random_pc_pair'

# Number of batches per epoch
epoch_size = 5000

# Number of epochs
num_epochs = 20

# learning rate strategy
learning_rate_strategy = 'linear'

# learning rate schedule
lr = 0.0001
learning_rate_schedule = {
    'initial':  1 * lr,
    'start':    50000,
    'end_step': 100000,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
    # 'Discriminator': ("ADAM", {'beta1': 0.5, 'beta2': 0.9}, 1.0),
    # 'Generator': ("ADAM", {'beta1': 0.5, 'beta2': 0.9}, 1.0),
    # 'ShapeNet': 1e-1,
}

# Restore model
restore_model = "../../project/caricacture/pretrained/sphere224_casia_sc0.7_256_conv4"

# Keywords to filter restore variables, set None for all
restore_scopes =  ['Discriminator/conv', 'Discriminator/Bot', 
                    'DiscriminatorB/conv', 'DiscriminatorB/Bot',
                    'FeatureNet/conv', 'FeatureNet/Bot']

# Replace rules for restoration
replace_rules = {
    # 'Discriminator': 'SphereNet',
    'FeatureNet': 'Discriminator',
}

# Weight decay for model variables
weight_decay = 1e-4

# Keep probability for dropouts
keep_prob = 1.0


####### LOSS FUNCTION #######

losses = {
    'coef_adv': 1.0,
    'coef_patch_adv': 2.0,
    'coef_idt': 10.0,
    'coef_feat': 0.0,
    'coef_cycle': 0.0,
    'coef_style_cycle': 0.0,
}

