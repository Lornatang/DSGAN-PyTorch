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
import os
import time

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    d_model, g_model = build_model()
    print(f"Build `{config.g_arch_name}` model successfully.")

    pixel_criterion, pixel_filter_criterion, content_criterion = define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer, config.epochs, config.decay_epochs)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained d model weights...")
    if config.pretrained_d_model_weights_path:
        d_model = load_state_dict(d_model, config.pretrained_d_model_weights_path)
        print(f"Loaded `{config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")

    print("Check whether to load pretrained g model weights...")
    if config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, config.pretrained_g_model_weights_path)
        print(f"Loaded `{config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    print("Check whether the pretrained d model is restored...")
    if config.resume_d:
        d_model, _, start_epoch, est_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            d_model,
            config.pretrained_d_model_weights_path,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    print("Check whether the pretrained g model is restored...")
    if config.resume_g:
        g_model, _, start_epoch, est_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            config.pretrained_g_model_weights_path,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Create an IQA evaluation model
    psnr_model = PSNR(0, config.only_test_y_channel)
    ssim_model = SSIM(0, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train(d_model,
              g_model,
              train_prefetcher,
              pixel_criterion,
              pixel_filter_criterion,
              content_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              writer)
        psnr, ssim = validate(g_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict(),
                         "scheduler": d_scheduler.state_dict()},
                        f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": g_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_gt_images_dir,
                                            config.gt_image_size,
                                            config.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(config.test_lr_images_dir, config.upscale_factor, None)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    d_model = model.__dict__[config.d_arch_name](in_channels=config.d_in_channels,
                                                 recursions=config.d_recursions,
                                                 kernel_size=config.d_kernel_size,
                                                 stride=config.d_stride,
                                                 filter_high_frequencies=config.d_filter_high_frequencies,
                                                 use_gaussian=config.d_use_gaussian)
    g_model = model.__dict__[config.g_arch_name](in_channels=config.g_in_channels,
                                                 out_channels=config.g_out_channels,
                                                 channels=config.g_channels,
                                                 num_blocks=config.g_num_blocks)
    d_model = d_model.to(device=config.device)
    g_model = g_model.to(device=config.device)

    return d_model, g_model


def define_loss() -> [nn.L1Loss, model.FilterLowFrequencies, model.content_loss]:
    pixel_criterion = nn.L1Loss()
    pixel_filter_criterion = model.FilterLowFrequencies(config.d_recursions,
                                                        config.d_kernel_size,
                                                        config.d_stride,
                                                        False,
                                                        True,
                                                        config.d_use_gaussian)
    content_criterion = model.content_loss(config.feature_model_extractor_node,
                                           config.feature_model_normalize_mean,
                                           config.feature_model_normalize_std)

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=config.device)
    pixel_filter_criterion = pixel_filter_criterion.to(device=config.device)
    content_criterion = content_criterion.to(device=config.device)

    return pixel_criterion, pixel_filter_criterion, content_criterion


def define_optimizer(d_model, g_model) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(d_model.parameters(),
                             config.model_lr,
                             config.model_betas,
                             config.model_eps,
                             config.model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             config.model_lr,
                             config.model_betas,
                             config.model_eps,
                             config.model_weight_decay)

    return d_optimizer, g_optimizer


def define_scheduler(
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        epochs: int,
        decay_epochs: int,
) -> [lr_scheduler.LambdaLR, lr_scheduler.LambdaLR]:
    start_decay = epochs - decay_epochs
    lr_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / config.decay_epochs)
    d_scheduler = lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lr_rule)
    g_scheduler = lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lr_rule)

    return d_scheduler, g_scheduler


def train(
        d_model: nn.Module,
        g_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        pixel_filter_criterion: model.FilterLowFrequencies,
        content_criterion: model.content_loss,
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        epoch: int,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_gt_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    d_model.train()
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model
        # Use the generator model to generate fake samples
        sr = g_model(lr)
        gt_outputs = d_model(gt)
        sr_outputs = d_model(sr.detach().clone())
        d_loss_weights = [1.0 / len(gt_outputs)] * len(gt_outputs)
        d_loss = 0.0
        for sr_outputs, gt_output, d_weight in zip(sr_outputs, gt_outputs, d_loss_weights):
            combine_loss = (torch.mean(-torch.log(gt_output + 1e-8)) - torch.mean(torch.log(1 - sr_outputs + 1e-8)))
            d_loss += torch.mul(d_weight, combine_loss)

        d_loss.backward()
        d_optimizer.step()
        # Finish training the discriminator model

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        sr_outputs = d_model(sr)
        lr_pixel_filter_outputs = pixel_filter_criterion(lr)
        sr_pixel_filter_outputs = pixel_filter_criterion(sr)
        pixel_loss = torch.mul(config.pixel_weight, pixel_criterion(lr_pixel_filter_outputs, sr_pixel_filter_outputs))
        content_loss = torch.mul(config.content_weight, content_criterion(lr, sr))
        adversarial_weights = [1.0 / len(sr_outputs)] * len(sr_outputs)
        adversarial_loss = 0.0
        for output, weight in zip(sr_outputs, adversarial_weights):
            adversarial_loss += torch.mul(weight, torch.mean(-torch.log(sr_outputs + 1e-8)))
        adversarial_loss = torch.mul(config.adversarial_weight, adversarial_loss)
        # Calculate the generator total loss value
        g_loss = pixel_loss + content_loss + adversarial_loss

        g_loss.backward()
        g_optimizer.step()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.mean(gt_outputs.detach())
        d_sr_probability = torch.mean(sr_outputs.detach())

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
        batch_index += 1


def validate(
        g_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    g_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            sr = g_model(lr)
            sr = torch.clamp_(sr, 0.0, 1.0)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
