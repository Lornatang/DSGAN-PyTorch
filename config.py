# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "dsgan"
# D model arch config
d_in_channels = 3
d_recursions = 1
d_kernel_size = 5
d_stride = 1
d_filter_high_frequencies = True
d_use_gaussian = True
# G model arch config
g_in_channels = 3
g_out_channels = 3
g_channels = 64
g_num_blocks = 8
# Image up magnification
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "DSGAN_Gaussian"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/DF2K/DSGAN/train"
    test_lr_images_dir = f"./data/Set5/original"

    gt_image_size = 512
    batch_size = 16
    num_workers = 4

    # Load the address of the pretrained model
    pretrained_d_model_weights_path = ""
    pretrained_g_model_weights_path = ""

    # Incremental training and migration training
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 300
    decay_epochs = 150

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    pixel_weight = 1.0
    content_weight = 0.01
    adversarial_weight = 0.005

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.5, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    gt_dir = f"./data/Set5/GTmod12"
    lr_dir = f"./results/{exp_name}"

    g_model_weights_path = "./results/pretrained_models/DSGAN_x4-DF2K_Gaussian.pth.tar"
