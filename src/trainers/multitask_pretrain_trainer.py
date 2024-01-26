import copy
import logging
import os
import random
import time
import traceback
from datetime import datetime
from datetime import timedelta

import coloredlogs
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import transforms
from wandb import AlertLevel
from src.common.countdown import countdown
from src.common.helpers.helpers import save_dict_as_json
from src.common.resource_metrics import get_resource_metrics
from src.dataloader.torch_mc_rl_data import S3MinioCustomDataset
from src.dataloader.transform_functions import get_2D_image_of_last_3D_img_in_batch, get_concat_h
from src.swin_unetr.losses.loss import Contrast
from src.swin_unetr.models.ssl_head import SSLHead
from src.swin_unetr.utils.ops import aug_rand
from src.swin_unetr.utils.ops2D import aug_rand2D
from src.trainers.a3c_functions import save_gradients
from src.trainers.base_trainer import BaseTrainer
from torchvision import models

coloredlogs.install(level='INFO')
load_dotenv()
CHECKPOINT_WAIT_TIME = int(os.getenv("CHECKPOINT_WAIT_TIME"))
if CHECKPOINT_WAIT_TIME is None:
    CHECKPOINT_WAIT_TIME = 180
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")


device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def get_conv_for_inpainting(dim, output_kernel_size, train_config: dict):
    # vae uplsampling from SSLHead class

    if train_config["pretrain_architecture"].lower() == "swin_vit":

        return nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, 3, kernel_size=output_kernel_size, stride=1),  # changes: kernel_size
        )

    else:
        return nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 16, 3, kernel_size=output_kernel_size, stride=1),  # changes: kernel_size
        )


def get_simclr_head(dim, dropout):
    return nn.Sequential(
        nn.LeakyReLU(), # prevent vanishing gradients
        nn.Dropout(dropout),
        nn.Linear(dim, 512)
    )  # contrastive_head


def get_shift_pred_seq(dim, dropout):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(2*dim, 256),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 32),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 1),
    )


class MultiTaskModel(nn.Module):

    def __init__(self, vision_encoder: nn.Module, num_encoded_features: int, dropout: float,
                 output_kernel_size, train_config: dict):

        super(MultiTaskModel, self).__init__()

        self.train_config = train_config
        self.vision_encoder = vision_encoder
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.num_encoded_features = num_encoded_features

        self.dropout = dropout
        self.output_kernel_size = output_kernel_size

        # head
        self.conv = get_conv_for_inpainting(dim=self.num_encoded_features, output_kernel_size=self.output_kernel_size,
                                            train_config=self.train_config)
        self.contrastive_head = get_simclr_head(dim=self.num_encoded_features, dropout= self.dropout)
        self.shift_seq = get_shift_pred_seq(dim=self.num_encoded_features,
                                            dropout= self.dropout)

    def forward(self, x1, x2):
        """
        x1: augmented image (b, c, h, w)
        x2: augmented image shifted (b, c, h, w)
        """
        # get feature vector
        if self.train_config["pretrain_architecture"].lower() == "swin_vit":
            feat1 = self.vision_encoder(x1)[-1]
            feat2 = self.vision_encoder(x2)[-1]
        else:
            feat1 = self.vision_encoder(x1)
            feat2 = self.vision_encoder(x2)

        # image reconstruction (inpainting)
        x1_rec = self.conv(feat1)
        x2_rec = self.conv(feat2)

        # feature pooling

        feat1 = self.avg_pool(feat1)
        feat1 = self.flatten(feat1)

        feat2 = self.avg_pool(feat2)
        feat2 = self.flatten(feat2)  # bs,  num_encoded_features

        # predict contrastive features
        x1_contrast = self.contrastive_head(feat1)
        x2_contrast = self.contrastive_head(feat2)

        # predict shift
        feat_concat = torch.concat([feat1,feat2], dim=1)  # bs,  num_encoded_features * 2
        pred_shift = self.shift_seq(feat_concat)

        return x1_rec, x2_rec, x1_contrast, x2_contrast, pred_shift


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.shift_loss = torch.nn.L1Loss().cuda() # L1 because L2 (e.g. MSE) does not perform well on outliers
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_shift, target_shift, output_contrastive, target_contrastive, output_recons, target_recons):
        shift_loss = self.alpha1 * self.shift_loss(output_shift, target_shift)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = shift_loss + contrast_loss + recon_loss

        return total_loss, (shift_loss, contrast_loss, recon_loss)

class MultiTaskPretrainTrainer(BaseTrainer):

    def __init__(self, train_config: dict):
        super().__init__(train_config)

        self.output_dir = "./tmp/multitask-pretrain"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.metrics: dict = {}
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.model = None
        self.best_metrics_file_path = None
        self.best_weights_file_path = None
        self.latest_metrics_file_path = None
        self.latest_weights_file_path = None
        self.vision_encoder_arch = self.train_config["pretrain_architecture"]
        self.session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.checkpoint_dir = "tmp/multitask-pretrain/" + self.vision_encoder_arch + "/checkpoint"
        self.checkpoint_filename = "multitask-pretrain_checkpoint.pt"
        self.checkpoint_path = self.checkpoint_dir + "/" + self.checkpoint_filename
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.training_run_counter: int = 0
        self.training_running: bool = False

        self.train_dataset: S3MinioCustomDataset = None
        self.val_dataset: S3MinioCustomDataset = None


        self.vision_encoder_out_dim = self.train_config["vision_encoder_out_dim"]
        model_name: str = self.train_config["pretrain_architecture"].lower()
        vision_encoder: nn.Module = None
        output_kernel_size = [1, 1, 1]  # default output layer kernel size
        self.args = self.ssl_head_args
        self.args.lr = train_config["pretrain_lr"]
        self.args.batch_size = train_config["pretrain_batch_size"]
        self.args.sw_batch_size = train_config["sw_batch_size"]
        num_encoded_features = 768
        if model_name.startswith("res"):
            vision_encoder = models.get_model(model_name, num_classes=self.vision_encoder_out_dim)
            #self.cnn.conv1 = nn.Conv2d(self.train_config["in_channels"], 64,
            #                           kernel_size=7, stride=2, padding=3, bias=False)  # channels
            num_encoded_features = self.vision_encoder_out_dim

        elif model_name == "swin_vit":
            if self.train_config["input_depth"] < 32:
                output_kernel_size[2] = int(32 - self.train_config["input_depth"] + 1)

            # using adapted output_kernel_siz to reconstruct 3d image with same depth as input depth
            output_kernel_size = [1, 1, 1]  # default output layer kernel size
            if self.train_config["input_depth"] < 32:
                output_kernel_size[2] = int(32 - self.train_config["input_depth"] + 1)
            vision_encoder: SSLHead = SSLHead(
                self.args, output_kernel_size=tuple(output_kernel_size)).swinViT  # only ViT

        self.model = MultiTaskModel(vision_encoder=vision_encoder, num_encoded_features=num_encoded_features,
                                    dropout=train_config["dropout"],
                                    output_kernel_size=output_kernel_size, train_config=self.train_config)

        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.train_config["pretrain_lr"], weight_decay=1e-4)
        self.scheduler = None


        # setup wandb run
        session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.log_dir = os.path.join(self.output_dir, session_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config_filename = self.log_dir + "/train_config.json"
        save_dict_as_json(self.train_config, None, self.config_filename)
        self.best_metrics_file_path = self.log_dir + "/best_metrics.json"
        self.best_weights_file_path = self.log_dir + "/best_model_" + self.vision_encoder_arch + "_state_dict.pt"
        self.latest_metrics_file_path = self.log_dir + "/latest_metrics.json"
        self.latest_weights_file_path = self.log_dir + "/latest_model_" + self.vision_encoder_arch + "_state_dict.pt"

        self.best_val_loss = 9999

        self.run_id = session_id

        self.train_dataset = None  # is set on training begin
        self.val_dataset = None  # is set on training begin

        self.img_input = None
        self.img_input_shifted = None
        self.img_aug_0 = None
        self.img_aug_1 = None
        self.img_rec_0 = None
        self.img_rec_1 = None
        self.val_img_rec_0 = None
        self.val_img_rec_1 = None

        self.epochs = self.train_config["ssl_pretrain_epochs"]

        self.batch_size = self.args.batch_size
        self.fp16_precision = True  # amp (automatic mixed precision)
        self.toPilTransform = transforms.ToPILImage()
        self.scaler = GradScaler(enabled=self.fp16_precision)
        self.loss_function = Loss(self.batch_size * self.args.sw_batch_size, self.args)

        self.img_log_n = 1000

    def prepare_3d_data(self, batch):
        batch_dict_x = batch
        x = batch_dict_x["tensor_image"]
        x = x.to(device)
        max_shift = int(self.train_config["img_size"][0] * 0.1)
        target_shift = random.randint(0, max_shift)
        if random.randint(0, 1) == 0:
            target_shift = -target_shift  # left shift else right shift
        # Percentage shift to use smaller numbers for mor stable predictions e.g. 1/20 or -1/20
        target_shift_tensor = torch.tensor(
            list(np.full(self.batch_size, target_shift / max_shift))).float().to(device)[:, None]  # size b, 1
        x1 = x
        x2 = torch.roll(x, shifts=target_shift, dims=3)
        x1_augment = aug_rand(self.args, x1)
        x2_augment = aug_rand(self.args, x2)
        return target_shift_tensor, x1, x1_augment, x2, x2_augment

    def prepare_2d_data(self, batch):
        batch_dict_x = batch
        x = batch_dict_x["tensor_image"]
        x = x.to(device)
        max_shift = int(self.train_config["img_size"][0] * 0.1)
        target_shift = random.randint(0, max_shift)
        if random.randint(0, 1) == 0:
            target_shift = -target_shift  # left shift else right shift
        # Percentage shift to use smaller numbers for mor stable predictions e.g. 1/20 or -1/20
        target_shift_tensor = torch.tensor(
            list(np.full(self.batch_size, target_shift / max_shift))).float().to(device)[:, None]  # size b, 1
        x1 = x
        x2 = torch.roll(x, shifts=target_shift, dims=2)
        x1_augment = aug_rand2D(self.args, x1)
        x2_augment = aug_rand2D(self.args, x2)
        return target_shift_tensor, x1, x1_augment, x2, x2_augment


    def train(self, train_loader, val_loader):

        logging.info(f"Start Multitask training for {self.epochs} epochs.")
        logging.info(f"Training with device: {device}.")

        for epoch_counter in range(self.epochs):
            self.step = 0

            loss_train = []
            loss_train_shift = []
            loss_train_contrast = []
            loss_train_recon = []

            for step, batch in enumerate(train_loader):
                # limit number of batches for comparable experiments
                if self.train_config["pretrain_max_train_batches"] == -1:
                    raise Exception("ERROR: train_config['pretrain_max_train_batches'] was not set")
                if self.train_config["pretrain_max_val_batches"] == -1:
                    raise Exception("ERROR: train_config['pretrain_max_val_batches'] was not set")

                if self.train_config["pretrain_max_train_batches"] > train_loader.__len__():
                    raise (Exception("ERROR: Not enough train batches available. Current: "
                                     + str(train_loader.__len__()) + " expected:"
                                     + str(self.train_config["pretrain_max_train_batches"])))

                if self.train_config["pretrain_max_val_batches"] > train_loader.__len__():
                    raise (Exception("ERROR: Not enough val batches available. Current: "
                                     + str(train_loader.__len__()) + " expected:"
                                     + str(self.train_config["pretrain_max_val_batches"])))

                if step == self.train_config["pretrain_max_train_batches"]:
                    break

                if self.global_step >= self.train_config["ssl_pretrain_steps"]:
                    logging.info("Finished last train step: " + str(self.global_step))
                    self.validate(val_loader=val_loader)
                    self.save_checkpoint()
                    return

                self.model.train()

                if self.train_config["pretrain_architecture"].lower() == "swin_vit":
                    target_shift_tensor, x1, x1_augment, x2, x2_augment = self.prepare_3d_data(batch)
                else:
                    target_shift_tensor, x1, x1_augment, x2, x2_augment = self.prepare_2d_data(batch)

                if (self.step < 1000 and self.step % 10 == 0) or self.step % self.img_log_n == 0:
                    try:
                        self.img_input = self.get_3D_image_as_2D_from_last_tensor_in_batch(x1)
                        self.img_input_shifted = self.get_3D_image_as_2D_from_last_tensor_in_batch(x2)
                        self.img_aug_0 = self.get_3D_image_as_2D_from_last_tensor_in_batch(x1_augment)
                        self.img_aug_1 = self.get_3D_image_as_2D_from_last_tensor_in_batch(x2_augment)
                    except Exception as e:
                        logging.warning(e)


                with autocast(enabled=self.fp16_precision):

                    rec_x1, rec_x2, contrastive1_p, contrastive2_p, pred_shift = self.model(x1_augment, x2_augment)

                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)

                    loss, losses_tasks = self.loss_function(pred_shift, target_shift_tensor, contrastive1_p,
                                                            contrastive2_p, imgs_recon, imgs)
                    loss_train.append(loss.item())
                    loss_train_shift.append(losses_tasks[0].item())
                    loss_train_contrast.append(losses_tasks[1].item())
                    loss_train_recon.append(losses_tasks[2].item())

                if self.step < 200 or self.step % self.img_log_n == 0:
                    try:
                        self.img_rec_0 = self.get_3D_image_as_2D_from_last_tensor_in_batch(rec_x1)
                        self.img_rec_1 = self.get_3D_image_as_2D_from_last_tensor_in_batch(rec_x2)
                    except Exception as e:
                        logging.warning(e)


                # optimize model

                if self.fp16_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    #gradients = save_gradients(self.model)  # debug
                    self.optimizer.step()


                if self.epoch >= 10 or self.step > 500:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.metrics["epoch"] = self.epoch
                self.metrics["step"] = self.step
                self.metrics["global_step"] = self.global_step
                self.metrics["len_val_loader"] = val_loader.__len__()
                self.metrics["len_train_loader"] = train_loader.__len__()
                self.metrics["loss"] = float(loss.detach().cpu().numpy())
                self.metrics["loss_shift_loss"] = float(losses_tasks[0].detach().cpu().numpy())
                self.metrics["loss_contrast_loss"] = float(losses_tasks[1].detach().cpu().numpy())
                self.metrics["loss_recon_loss"] = float(losses_tasks[2].detach().cpu().numpy())

                self.metrics['after_lr'] = float(self.scheduler.get_last_lr()[-1])
                self.metrics["current_cache_size_train_loader"] = int(len(train_loader.dataset.cache))
                self.metrics["current_tensor_cache_size_train_loader"] = int(len(train_loader.dataset.tensor_cache))

                # 80% train, 20% validation -> 8 train, 2 val steps
                val_condition = self.step == 0 or self.step % 80 == 0 or self.metrics["len_train_loader"] < 80

                if val_condition:
                    self.validate(val_loader=val_loader)

                self.step += 1
                self.global_step+=1

                try:
                    if self.step < 200 or self.step % self.img_log_n == 0:
                        self.log_images()
                except Exception as e:
                    logging.warning(e)

                # resources metrics
                if step == 0 or (step > 0 and step % 100 == 0):
                    logging.info(get_resource_metrics())
                self.metrics.update(get_resource_metrics())
                wandb_data: dict = {}
                wandb_data.update(self.metrics)
                self.run.log(wandb_data)
                if step > 1 and step % self.img_log_n == 0:
                    self.save_checkpoint()

                logging.info(
                    f"Epoch: {self.epoch}\tStep: {str(self.step)+'/'+str(len(self.train_dataset))}\tLoss: {loss}")

            self.metrics["mean_train_loss"] = float(np.mean(loss_train))
            self.metrics["mean_train_shift_loss"] = float(np.mean(loss_train_shift))
            self.metrics["mean_train_contrast_loss"] = float(np.mean(loss_train_contrast))
            self.metrics["mean_train_recon_loss"] = float(np.mean(loss_train_recon))

            self.epoch += 1
            # warmup for the first epochs

            logging.info(f"Epoch: {self.epoch}\tmean_train_loss: {str(self.metrics['mean_train_loss'])}\tmean_val_loss: {str(self.metrics['mean_val_loss'])}")

            self.best_metrics = copy.deepcopy(self.metrics)
            self.save_checkpoint()


        logging.info("Training has finished.")
        # save model checkpoints

    def log_images(self):
        self.run.log({"img_input": wandb.Image(self.img_input)}, commit=False)
        self.run.log({"img_input_shifted": wandb.Image(self.img_input_shifted)}, commit=False)
        self.run.log({"img_aug_0": wandb.Image(self.img_aug_0)}, commit=False)
        self.run.log({"img_aug_1": wandb.Image(self.img_aug_1)}, commit=False)
        self.run.log({"img_rec_0": wandb.Image(self.img_rec_0)}, commit=False)
        self.run.log({"img_rec_1": wandb.Image(self.img_rec_1)}, commit=False)
        if self.val_img_rec_0 is not None:
            self.run.log({"val_img_rec_0": wandb.Image(self.val_img_rec_0)}, commit=False)
        if self.val_img_rec_1 is not None:
            self.run.log({"val_img_rec_1": wandb.Image(self.val_img_rec_1)}, commit=False)

    def validate(self, val_loader):
        logging.info("\n\nVALIDATION")
        # Only use a part of data to reduce needed train time
        self.model.eval()
        loss_val = []
        loss_val_shift = []
        loss_val_contrast = []
        loss_val_recon = []
        val_dataset: Dataset = val_loader.dataset
        max_index: int = min(val_dataset.__len__() - 1, self.train_config["pretrain_max_val_batches"] -1)
        val_steps = min(int(20), max_index)
        val_subset = torch.utils.data.Subset(val_loader.dataset,
                                             random.sample(range(0, max_index), min(val_steps, max_index)))
        sub_val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for step, batch in enumerate(sub_val_loader):

            self.model.train()
            target_shift_tensor, x1, x1_augment, x2, x2_augment = self.prepare_3d_data(batch)

            with autocast(enabled=self.fp16_precision):

                rec_x1, rec_x2, contrastive1_p, contrastive2_p, pred_shift = self.model(x1_augment, x2_augment)

                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)

                loss, losses_tasks = self.loss_function(pred_shift, target_shift_tensor, contrastive1_p,
                                                        contrastive2_p, imgs_recon, imgs)
                loss_val.append(loss.item())
                loss_val_shift.append(losses_tasks[0].item())
                loss_val_contrast.append(losses_tasks[1].item())
                loss_val_recon.append(losses_tasks[2].item())

            if self.step < 200 or self.step % self.img_log_n == 0:
                try:
                    self.val_img_rec_0 = self.get_3D_image_as_2D_from_last_tensor_in_batch(rec_x1)
                    self.val_img_rec_1 = self.get_3D_image_as_2D_from_last_tensor_in_batch(rec_x2)
                except Exception as e:
                    logging.warning(e)

            self.metrics["val_loss"] = float(loss.detach().cpu().numpy())
            self.metrics["val_loss_shift_loss"] = float(losses_tasks[0].detach().cpu().numpy())
            self.metrics["val_loss_contrast_loss"] = float(losses_tasks[1].detach().cpu().numpy())
            self.metrics["val_loss_recon_loss"] = float(losses_tasks[2].detach().cpu().numpy())
            self.metrics["current_cache_size_val_loader"] = int(len(val_loader.dataset.cache))
            self.metrics["current_tensor_cache_size_val_loader"] = int(len(val_loader.dataset.tensor_cache))

            # resources metrics
            self.metrics.update(get_resource_metrics())
            if step > 1 and step % self.img_log_n == 0:
                self.save_model(self.model, self.metrics, self.latest_metrics_file_path,
                                self.latest_weights_file_path)
            logging.info(
                f"Validation Epoch: {self.epoch}\tStep: {str(step)+'/'+str(val_steps)}\tLoss: {loss}")


        self.metrics["mean_val_loss"] = float(np.mean(loss_val))
        self.metrics["mean_val_shift_loss"] = float(np.mean(loss_val_shift))
        self.metrics["mean_val_contrast_loss"] = float(np.mean(loss_val_contrast))
        self.metrics["mean_val_recon_loss"] = float(np.mean(loss_val_recon))
        if self.metrics["mean_val_loss"] < self.best_val_loss:
            self.best_val_loss = self.metrics["mean_val_loss"]

            self.best_metrics = copy.deepcopy(self.metrics)
            self.save_checkpoint()
            try:
                wandb.alert(
                    title='Multitask Trainer ' + self.run_id,
                    text=str(
                        "New best val_loss: " + str(self.best_val_loss) + " epochs: " + str(self.epochs)),
                    level=wandb.AlertLevel.INFO,
                    wait_duration=timedelta(minutes=15)
                )
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
            if not (self.step < 200 or self.step % self.img_log_n == 0):
                self.log_images()

    def get_3D_image_as_2D_from_last_tensor_in_batch(self, tensor):
        if len(tensor.shape) == 4:
            return transforms.ToPILImage()(tensor[0])
        else:
            concat_img = None
            for i in range(self.train_config["input_depth"]):
                img = get_2D_image_of_last_3D_img_in_batch(tensor, image_index=i, squeeze=False)
                if concat_img is None:
                    concat_img = img
                else:
                    concat_img = get_concat_h(concat_img, img)
            return concat_img

    def save_model(self, model: MultiTaskModel, metrics: dict, metrics_filepath, weights_filepath):
        save_dict_as_json(metrics, None, metrics_filepath)
        torch.save(model.vision_encoder.state_dict(), weights_filepath)
        print("Model saved to dir:", weights_filepath)

    def run_training(self, train_dataset, val_dataset):
        if self.training_running:
            return

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if self.train_config["scheduler"] == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=0,
                                                                    last_epoch=-1)
        elif self.train_config["scheduler"] == "LinearLR":
            self.scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=40000)

        self.wandb_sweep_config = {
            'name': 'sweep',
            'method': 'grid',  # grid, random, bayes
            'metric': {'goal': 'minimize', 'name': 'loss'},
            'parameters': {
                "seq_len": {"value": self.train_config["input_depth"]},
                "minio_bucket_name": {"value": self.train_config["minio_bucket_name"]},
                "len_val_dataset": {"value": len(self.val_dataset)},
                "len_train_dataset": {"value": len(self.train_dataset)},
            }
        }
        for key in self.train_config:
            self.wandb_sweep_config["parameters"][key] = {"value": self.train_config[key]}

        self.checkpoint_loaded = False
        if CHECKPOINT_WAIT_TIME > 0:
            countdown(sec=CHECKPOINT_WAIT_TIME,
                      optional_text="Giving time to upload torch model checkpoint to "
                                    + self.checkpoint_path + " e.g. using post request",
                      cancel_condition_function=self.checkpoint_exists)
            self.checkpoint_loaded: bool = self.load_checkpoint()


        self.run = wandb.init(project="multitask-pretrain", config=self.wandb_sweep_config, resume=self.checkpoint_loaded,
                              id=self.run_id)
        wandb.watch(self.model)

        model_artifact = wandb.Artifact('model', type='model')
        self.training_run_counter += 1
        self.training_running = True

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        logging.info("train_dataset.length " + str(train_dataset.length))
        logging.info("val_dataset.length " + str(val_dataset.length))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)


        try:
            self.train(train_loader, val_loader)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            self.save_checkpoint()
            try:
                wandb.alert(
                    title='Multitask Pretraining Exception',
                    text=str(e),
                    level=AlertLevel.ERROR,
                    wait_duration=timedelta(minutes=5)
                )
            except Exception as e:
                logging.error(e)
                traceback.print_exc()

        self.save_model(self.model, self.metrics, self.latest_metrics_file_path, self.latest_weights_file_path)

        if os.path.isfile(self.best_metrics_file_path):
            model_artifact.add_file(self.best_metrics_file_path)

        if os.path.isfile(self.best_weights_file_path):
            model_artifact.add_file(self.best_weights_file_path)

        if os.path.isfile(self.latest_metrics_file_path):
            model_artifact.add_file(self.latest_metrics_file_path)

        if os.path.isfile(self.latest_metrics_file_path):
            model_artifact.add_file(self.latest_metrics_file_path)
        self.run.log_artifact(model_artifact, aliases=['latest', str(self.epoch)])
        self.training_running = False

    def make_backup_wandb_artifact(self):
        try:
            logging.info("save checkpoint artifact")
            self.start_time_tmp_artifact = time.time()
            logging.info("save checkpoint artifact")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            tmp_checkpoint_artifact = wandb.Artifact(
                "checkpoint_" + self.run.project + '_' + self.run_id + '_' + str(self.epoch),
                type='model')
            if os.path.isfile(self.checkpoint_path):
                tmp_checkpoint_artifact.add_file(self.checkpoint_path)
            if os.path.isfile(self.best_metrics_file_path):
                tmp_checkpoint_artifact.add_file(self.best_metrics_file_path)
            if os.path.isfile(self.best_weights_file_path):
                tmp_checkpoint_artifact.add_file(self.best_weights_file_path)
            if os.path.isfile(self.latest_metrics_file_path):
                tmp_checkpoint_artifact.add_file(self.latest_metrics_file_path)
            if os.path.isfile(self.latest_weights_file_path):
                tmp_checkpoint_artifact.add_file(self.latest_weights_file_path)
            if os.path.isfile(self.config_filename):
                tmp_checkpoint_artifact.add_file(self.config_filename)
            self.run.log_artifact(tmp_checkpoint_artifact)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    def save_checkpoint(self) -> []:
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            save_obj: dict = {
                'epoch': self.epoch,
                'step': self.step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.metrics,
                'train_config': self.train_config,
                'run_id': self.run_id,
                'global_step': self.global_step
            }

            if self.scheduler is not None:
                save_obj['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(save_obj, self.checkpoint_path)
            self.save_model(self.model, self.metrics, self.latest_metrics_file_path,
                            self.latest_weights_file_path)
            self.save_model(self.model, self.best_metrics, self.best_metrics_file_path, self.best_weights_file_path)
            self.make_backup_wandb_artifact()
            logging.info("Checkpoint saved")
            return [self.checkpoint_path, self.latest_metrics_file_path, self.best_metrics_file_path,
                    self.latest_weights_file_path, self.best_weights_file_path, self.config_filename]

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            return "error " + str(e)

    def load_checkpoint(self) -> bool:
        try:
            if os.path.isfile(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path)
                self.epoch = checkpoint['epoch']
                self.step = checkpoint['step']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.metrics = checkpoint['metrics']
                self.train_config = checkpoint['train_config']
                if "run_id" in checkpoint:
                    self.run_id = checkpoint["run_id"]
                if "global_step" in checkpoint:
                    self.global_step = checkpoint["global_step"]

                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                if str(MINIO_BUCKET_NAME) != self.train_config["minio_bucket_name"]:
                    text = "MINIO_BUCKET_NAME is not the same. checkpoint: " + str(
                        self.train_config["minio_bucket_name"]) + "env variable: " + str(MINIO_BUCKET_NAME)
                    logging.error("text")
                    self.stop()
                    raise Exception(text)

                self.model.train()
                logging.info("Checkpoint loaded")
                return True
            else:
                logging.info("OK. Checkpoint could not be loaded. No file existing. Stop training manually if this was not expected and try uploading checkpoint again")
                return False
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        return False
