import copy
import logging
import os
import random
import socket
import time
import traceback
from datetime import datetime
from datetime import timedelta

import coloredlogs
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import transforms
from wandb import AlertLevel

from src.common.countdown import countdown
from src.common.helpers.helpers import save_dict_as_json
from src.common.resource_metrics import get_resource_metrics
from src.dataloader.torch_mc_rl_data import S3MinioCustomDataset
from src.trainers.base_trainer import BaseTrainer
from src.trainers.resnet_simclr import ResNetSimCLR

coloredlogs.install(level='INFO')
load_dotenv()
CHECKPOINT_WAIT_TIME = int(os.getenv("CHECKPOINT_WAIT_TIME"))
if CHECKPOINT_WAIT_TIME is None:
    CHECKPOINT_WAIT_TIME = 180
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SimClrPretrainTrainer(BaseTrainer):

    def __init__(self, train_config: dict):
        super().__init__(train_config)

        self.output_dir = "./tmp/simclr-pretrain"
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
        self.session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.checkpoint_dir = "tmp/simclr-pretrain/checkpoint"
        self.checkpoint_filename = "simclr-pretrain_checkpoint.pt"
        self.checkpoint_path = self.checkpoint_dir + "/" + self.checkpoint_filename
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.training_run_counter: int = 0
        self.training_running: bool = False

        self.train_dataset: S3MinioCustomDataset = None
        self.val_dataset: S3MinioCustomDataset = None

        self.vision_encoder_arch = self.train_config["pretrain_architecture"]

        self.model = ResNetSimCLR(base_model=self.vision_encoder_arch,
                                  out_dim=self.train_config["vision_encoder_out_dim"],
                                  train_config=self.train_config)

        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.train_config["pretrain_lr"], weight_decay=1e-4)
        self.scheduler = None

        self.criterion = torch.nn.CrossEntropyLoss().to(device)

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
        self.img_aug_0 = None
        self.img_aug_1 = None

        self.epochs = self.train_config["ssl_pretrain_epochs"]
        self.batch_size = self.train_config["pretrain_batch_size"]
        self.n_views = 2
        self.fp16_precision = False
        self.temperature = 0.7

        self.data_transforms = transforms.Compose([transforms.ColorJitter(brightness=(0.5, 1.0))])
        self.toPilTransform = transforms.ToPILImage()
        self.scaler = GradScaler(enabled=self.fp16_precision)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.n_views * self.batch_size, self.n_views * self.batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # sim
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, val_loader):
        logging.info(f"Start SimCLR training for {self.epochs} epochs.")
        logging.info(f"Training with device: {device}.")

        for epoch_counter in range(self.epochs):
            self.step = 0

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
                batch_dict_x = batch
                images = batch_dict_x["tensor_image"]

                self.img_input = self.toPilTransform(images[-1])

                augmented_image_batches = []
                for n in range(self.n_views):
                    # Augmentation pipeline
                    augmented_batch = images.detach().clone()
                    if "ColorJitter" in self.train_config["pretrain_augmentation"]:
                        augmented_batch = self.data_transforms(images)

                    if "horizontal_shift" in self.train_config["pretrain_augmentation"]:
                        max_shift_horizontal = int(self.train_config["img_size"][1] * 0.1)
                        horizontal_shift = random.randint(0, max_shift_horizontal)
                        if random.randint(0, 1) == 0:
                            horizontal_shift = -horizontal_shift  # left shift else right shift

                        flip_point_h = horizontal_shift
                        if flip_point_h > 0:
                            augmented_batch = torch.cat(
                                [augmented_batch[:, :, :, flip_point_h:],
                                 augmented_batch[:, :, :, -flip_point_h:].flip(3)], dim=3)
                        elif flip_point_h < 0:
                            augmented_batch = torch.cat(
                                [augmented_batch[:, :, :, :-flip_point_h].flip(3),
                                 augmented_batch[:, :, :, :flip_point_h]], dim=3)

                    if "vertical_shift" in self.train_config["pretrain_augmentation"]:
                        max_shift_vertical = int(self.train_config["img_size"][0] * 0.1)
                        vertical_shift = random.randint(0, max_shift_vertical)
                        if random.randint(0, 1) == 0:
                            vertical_shift = -vertical_shift  # down shift else up shift
                        flip_point_v = vertical_shift
                        if flip_point_v > 0:
                            augmented_batch = torch.cat(
                                [augmented_batch[:, :, flip_point_v:, :],
                                 augmented_batch[:, :, -flip_point_v:, :].flip(2)], dim=2)
                        elif flip_point_v < 0:
                            augmented_batch = torch.cat(
                                [augmented_batch[:, :, :-flip_point_v, :].flip(2),
                                 augmented_batch[:, :, :flip_point_v, :]], dim=2)

                    #augmented_batch = torch.roll(augmented_batch, shifts=(horizontal_shift, vertical_shift),
                    #                             dims=(3, 2))
                    augmented_image_batches.append(augmented_batch)

                    try:
                        if n == 0:
                            self.img_aug_0 = self.toPilTransform(augmented_batch[-1])
                        elif n == 1:
                            self.img_aug_1 = self.toPilTransform(augmented_batch[-1])
                    except Exception as e:
                        logging.warning(e)
                try:
                    images = torch.cat(augmented_image_batches, dim=0)

                    images = images.to(device)

                    with autocast(enabled=self.fp16_precision):
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    self.scaler.scale(loss).backward()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception as e:
                    logging.warning(e)
                    try:
                        wandb.alert(
                            title='simCLR Pretraining Exception',
                            text=str(e),
                            level=AlertLevel.ERROR,
                            wait_duration=timedelta(minutes=5)
                        )
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                self.metrics["epoch"] = self.epoch
                self.metrics["step"] = self.step
                self.metrics["global_step"] = self.global_step
                self.metrics["len_val_loader"] = val_loader.__len__()
                self.metrics["len_train_loader"] = train_loader.__len__()
                self.metrics["loss"] = float(loss.detach().cpu().numpy())
                self.metrics['acc_top1'] = float(top1[0])
                if self.batch_size * self.n_views > 5:
                    self.metrics['acc_top5'] = float(top5[0])
                self.metrics['after_lr'] = float(self.scheduler.get_lr()[-1])
                self.metrics["current_cache_size_train_loader"] = int(len(train_loader.dataset.cache))
                self.metrics["current_tensor_cache_size_train_loader"] = int(len(train_loader.dataset.tensor_cache))

                self.step += 1
                self.global_step += 1
                if step < 2 and self.img_input is not None and self.epoch < 5:
                    wandb.log({"img_input": wandb.Image(self.img_input)}, commit=False)
                    wandb.log({"img_aug_0": wandb.Image(self.img_aug_0)}, commit=False)
                    wandb.log({"img_aug_1": wandb.Image(self.img_aug_1)}, commit=False)

                val_condition = step == 0 or step % 80 == 0  # 80% train, 20% validation -> 80 train, 20 val steps

                if val_condition:
                    self.validate(val_loader=val_loader)

                # resources metrics
                if step == 0 or (step > 0 and step % 100 == 0):
                    logging.info(get_resource_metrics())
                self.metrics.update(get_resource_metrics())
                wandb_data: dict = {}
                wandb_data.update(self.metrics)
                wandb.log(wandb_data)
                if step > 1 and step % 100 == 0:
                    self.save_model(self.model, self.metrics, self.latest_metrics_file_path,
                                    self.latest_weights_file_path)
                logging.info(
                    f"Epoch: {self.epoch}\tStep: {str(self.step) + '/' + str(len(self.train_dataset))}\tLoss: {loss}\tTop1 accuracy: {self.metrics['acc_top1']}")

            self.epoch += 1
            # warmup for the first epochs
            if self.epoch >= 10 or self.step > 10000:
                self.scheduler.step()
            logging.info(f"Epoch: {self.epoch}\tLoss: {loss}\tTop1 accuracy: {self.metrics['acc_top1']}")

            if self.best_metrics is None:
                self.best_metrics = copy.deepcopy(self.metrics)
            self.save_checkpoint()

        logging.info("Training has finished.")
        # save model checkpoints

    def validate(self, val_loader):
        self.model.eval()
        val_dataset: Dataset = val_loader.dataset
        max_index: int = min(val_dataset.__len__() - 1, self.train_config["pretrain_max_val_batches"] - 1)
        val_steps = min(int(20), max_index)
        val_subset = torch.utils.data.Subset(val_loader.dataset,
                                             random.sample(range(0, max_index), min(val_steps, max_index)))
        sub_val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for step, batch in enumerate(sub_val_loader):

            batch_dict_x = batch
            images = batch_dict_x["tensor_image"]

            augmented_image_batches = []
            for n in range(self.n_views):
                augmented_batch = self.data_transforms(images)
                augmented_image_batches.append(augmented_batch)

            images = torch.cat(augmented_image_batches, dim=0)
            images = images.to(device)

            with autocast(enabled=self.fp16_precision):
                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.metrics["val_loss"] = float(loss.detach().cpu().numpy())
            self.metrics['val_acc_top1'] = float(top1[0])
            if self.batch_size * self.n_views > 5:
                self.metrics['val_acc_top5'] = float(top5[0])
            self.metrics['val_after_lr'] = float(self.scheduler.get_lr()[0])
            self.metrics["current_cache_size_val_loader"] = int(len(val_loader.dataset.cache))
            self.metrics["current_tensor_cache_size_val_loader"] = int(len(val_loader.dataset.tensor_cache))

            if self.metrics["val_loss"] < self.best_val_loss and self.epoch > 0:
                self.best_val_loss = self.metrics["val_loss"]

                self.best_metrics = copy.deepcopy(self.metrics)
                self.save_model(self.model, self.best_metrics, self.best_metrics_file_path, self.best_weights_file_path)
                self.save_checkpoint()
                try:
                    wandb.alert(
                        title='simCLR Trainer ' + self.run_id,
                        text=str(
                            "New best val_loss: " + str(self.best_val_loss)),
                        level=wandb.AlertLevel.INFO,
                        wait_duration=timedelta(minutes=15)
                    )
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()

            # resources metrics
            self.metrics.update(get_resource_metrics())
            if step > 1 and step % 100 == 0:
                self.save_model(self.model, self.metrics, self.latest_metrics_file_path,
                                self.latest_weights_file_path)
            logging.info(
                f"validation Epoch: {self.epoch}\tStep: {str(step) + '/' + str(len(self.val_dataset))}\tLoss: {loss}\tTop1 val accuracy: {self.metrics['val_acc_top1']}")

    def save_model(self, model: ResNetSimCLR, metrics: dict, metrics_filepath, weights_filepath):
        save_dict_as_json(metrics, None, metrics_filepath)
        torch.save(model.backbone.state_dict(), weights_filepath)
        print("Model saved to dir:", weights_filepath)

    def run_training(self, train_dataset, val_dataset):
        if self.training_running:
            return

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=0,
                                                                    last_epoch=-1)
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
            countdown(sec=10,
                      optional_text="Checking for file")  # additional wait time while writing uploaded file
            self.checkpoint_loaded: bool = self.load_checkpoint()

            retry: int = 0
            while not self.checkpoint_loaded:
                countdown(sec=10,
                          optional_text="Retry " + str(retry) + " Checking for file")
                self.checkpoint_loaded: bool = self.load_checkpoint()
                retry += 1

        self.run = wandb.init(project="simclr-pretrain", config=self.wandb_sweep_config, resume=self.checkpoint_loaded,
                              id=self.run_id)
        wandb.watch(self.model)

        self.training_run_counter += 1
        self.training_running = True

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        logging.info("train_dataset.length " + str(train_dataset.length))
        logging.info("val_dataset.length " + str(val_dataset.length))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        try:
            # dataset = ContrastiveLearningDataset('./tmp/datasets')
            # train_dataset = dataset.get_dataset('stl10', self.n_views)
            # train_loader = torch.utils.data.DataLoader(
            #    train_dataset, batch_size=self.batch_size, shuffle=True,
            #    num_workers=1, pin_memory=True, drop_last=True)
            self.train(train_loader, val_loader)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            self.save_checkpoint()
            try:
                wandb.alert(
                    title='simCLR Pretraining Exception',
                    text=str(e),
                    level=AlertLevel.ERROR,
                    wait_duration=timedelta(minutes=5)
                )
            except Exception as e:
                logging.error(e)
                traceback.print_exc()

        self.save_model(self.model, self.metrics, self.latest_metrics_file_path, self.latest_weights_file_path)

        self.training_running = False
        try:
            wandb.alert(
                title='finished ' + str(socket.gethostname()[-50:]),
                text=str(socket.gethostname()) + " Config: " + str(self.train_config),
                level=wandb.AlertLevel.INFO,
                wait_duration=timedelta(minutes=1)
            )
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        self.run.finish()
        logging.info("finished. Waiting")
        while True:
            time.sleep(120)

    def make_backup_wandb_artifact(self):
        try:
            logging.info("save checkpoint artifact")
            self.start_time_tmp_artifact = time.time()
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
            wandb.log_artifact(tmp_checkpoint_artifact)
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
                    try:
                        wandb.alert(
                            title='simCLR Trainer ' + self.run_id,
                            text=str("Exception:" + text),
                            level=wandb.AlertLevel.ERROR,
                            wait_duration=timedelta(minutes=1)
                        )
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()
                    raise Exception(text)

                self.model.train()
                logging.info("Checkpoint loaded")
                return True
            else:
                logging.info(
                    "OK. Checkpoint could not be loaded. No file existing. Stop training manually if this was not expected and try uploading checkpoint again")
                return False
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        return False
