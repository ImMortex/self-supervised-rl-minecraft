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
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from wandb import AlertLevel

from src.common.countdown import countdown
from src.common.helpers.helpers import save_dict_as_json
from src.common.resource_metrics import get_resource_metrics
from src.dataloader.torch_mc_rl_data import S3MinioCustomDataset
from src.dataloader.transform_functions import get_2D_image_of_last_3D_img_in_batch, get_concat_h
from src.swin_unetr.losses.loss import Loss
from src.swin_unetr.models.ssl_head import SSLHead
from src.swin_unetr.optimizers.lr_scheduler import WarmupCosineSchedule
from src.swin_unetr.utils.ops import aug_rand, rot_rand
from src.trainers.base_trainer import BaseTrainer

coloredlogs.install(level='INFO')
load_dotenv()
CHECKPOINT_WAIT_TIME = int(os.getenv("CHECKPOINT_WAIT_TIME"))
if CHECKPOINT_WAIT_TIME is None:
    CHECKPOINT_WAIT_TIME = 180
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))



class PretrainTrainer(BaseTrainer):

    def __init__(self, train_config: dict):
        super().__init__(train_config)

        self.output_dir = "./tmp/swin-t_ssl"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.metrics: dict = {}
        self.metrics["train_loss"] = 99999.0
        self.metrics["train_loss_recon"] = 99999.0
        self.metrics["val_loss"] = 99999.0
        self.metrics["val_loss_recon"] = 99999.0
        self.metrics["best_val_loss_recon"] = 99999.0
        self.epoch = 0
        self.step = 0
        self.model = None
        self.best_metrics_file_path = None
        self.best_weights_file_path = None
        self.latest_metrics_file_path = None
        self.latest_weights_file_path = None
        self.session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.checkpoint_dir = "tmp/swin-t_ssl/checkpoint"
        self.checkpoint_filename = "swin-t_ssl_checkpoint.pt"
        self.checkpoint_path = self.checkpoint_dir + "/" + self.checkpoint_filename
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.training_run_counter: int = 0
        self.training_running: bool = False

        self.args = self.ssl_head_args
        self.args.lr = train_config["pretrain_lr"]
        self.args.batch_size = train_config["pretrain_batch_size"]
        self.args.sw_batch_size = train_config["sw_batch_size"]
        self.args.eval_num = 80
        self.train_dataset: S3MinioCustomDataset = None
        self.val_dataset: S3MinioCustomDataset = None
        logging.info("using args for training")
        args_dict: dict = vars(self.args)
        for key in args_dict:
            print(str(key) + " : " + str(args_dict[key]))

        if self.args.rank == 0:
            os.makedirs(self.args.logdir, exist_ok=True)
            writer = SummaryWriter(self.args.logdir)
        else:
            writer = None

        output_kernel_size = [1, 1, 1]  # default output layer kernel size
        if self.train_config["input_depth"] < 32:
            output_kernel_size[2] = int(32 - self.train_config["input_depth"] + 1)
        self.model = SSLHead(self.args, output_kernel_size=tuple(output_kernel_size))  # only encoder of SwinUNETR

        self.model.to(device)

        self.optimizer = self.get_optimizer(self.args, self.model, self.args.lr)
        # setup wandb run
        session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.log_dir = os.path.join(self.output_dir, session_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config_filename = os.path.join(self.output_dir, "train_config.json")
        save_dict_as_json(self.train_config, None, self.config_filename)
        self.best_metrics_file_path = self.log_dir + "/best_metrics.json"
        self.best_weights_file_path = self.log_dir + "/best_model_swinViT_state_dict.pt"
        self.latest_metrics_file_path = self.log_dir + "/latest_metrics.json"
        self.latest_weights_file_path = self.log_dir + "/latest_model_swinViT_state_dict.pt"

        self.run_id = session_id

        self.train_dataset = None  # is set on training begin
        self.val_dataset = None  # is set on training begin

        self.img_input = None
        self.img_recon = None
        self.img_recon2 = None
        self.img_x1 = None
        self.img_x2 = None

    def train(self, model, args, epoch, train_loader, val_loader, val_best, scaler, optimizer, loss_function,
              scheduler, log_dir):
        model.train()
        loss_train = []
        loss_train_recon = []
        batch_count = len(train_loader)
        # init
        val_loss = self.metrics["val_loss"]
        val_loss_recon = self.metrics["val_loss_recon"]
        mean_train_loss = 9999
        mean_train_loss_recon = 9999
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

            self.step = step
            t1 = time.time()
            batch_dict_x = batch
            x = batch_dict_x["tensor_image"]
            if len(x.shape) == 4 and self.train_config["input_depth"] == 1:
                x = x[:, :, :, :, None]  # add missing dimension for depth of the 3D image
            print(x.shape)

            try:
                concat_img = None
                for i in range(self.train_config["input_depth"]):
                    img = get_2D_image_of_last_3D_img_in_batch(x, image_index=i, squeeze=False)
                    if concat_img is None:
                        concat_img = img
                    else:
                        concat_img = get_concat_h(concat_img, img)
                self.img_input = concat_img
            except Exception as e:
                logging.warning(e)
            x = x.to(device)

            # such as mentioned in the paper: labels are not used for pre-training stage
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            with autocast(enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)

                print("imgs_recon")
                print(imgs_recon.shape)
                try:
                    concat_img = None
                    for i in range(self.train_config["input_depth"]):
                        img = get_2D_image_of_last_3D_img_in_batch(rec_x1, image_index=i, squeeze=False)
                        if concat_img is None:
                            concat_img = img
                        else:
                            concat_img = get_concat_h(concat_img, img)
                    self.img_recon = concat_img
                except Exception as e:
                    logging.warning(e)
                try:
                    concat_img = None
                    for i in range(self.train_config["input_depth"]):
                        img = get_2D_image_of_last_3D_img_in_batch(rec_x2, image_index=i, squeeze=False)
                        if concat_img is None:
                            concat_img = img
                        else:
                            concat_img = get_concat_h(concat_img, img)
                    self.img_recon2 = concat_img
                except Exception as e:
                    logging.warning(e)
                try:
                    concat_img = None
                    for i in range(self.train_config["input_depth"]):
                        img = get_2D_image_of_last_3D_img_in_batch(x1, image_index=i, squeeze=False)
                        if concat_img is None:
                            concat_img = img
                        else:
                            concat_img = get_concat_h(concat_img, img)
                    self.img_x1 = concat_img
                except Exception as e:
                    logging.warning(e)
                try:
                    concat_img = None
                    for i in range(self.train_config["input_depth"]):
                        img = get_2D_image_of_last_3D_img_in_batch(x2, image_index=i, squeeze=False)
                        if concat_img is None:
                            concat_img = img
                        else:
                            concat_img = get_concat_h(concat_img, img)
                    self.img_x2 = concat_img
                except Exception as e:
                    logging.warning(e)

                loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
                self.metrics["train_step_total_loss"] = float(loss.cpu().detach().numpy())
                self.metrics["train_step_rot_loss"] = float(losses_tasks[0].cpu().detach().numpy())
                self.metrics["train_step_contrast_loss"] = float(losses_tasks[1].cpu().detach().numpy())
                self.metrics["train_step_recon_loss"] = float(losses_tasks[2].cpu().detach().numpy())
                self.metrics["step"] = int(step)
                self.metrics["epoch"] = int(self.epoch)
                try:
                    self.metrics["current_cache_size_train_loader"] = int(len(train_loader.dataset.cache))
                    self.metrics["current_tensor_cache_size_train_loader"] = int(len(train_loader.dataset.tensor_cache))
                except Exception as e:
                    logging.error(e)

            loss_train.append(loss.item())
            loss_train_recon.append(losses_tasks[2].item())

            self.metrics.update(get_resource_metrics())

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            step_time = time.time() - t1
            print(
                "Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(epoch, args.epochs, step, batch_count, loss,
                                                                           step_time))
            self.metrics["step_time"] = float(step_time)

            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (epoch % args.eval_num == 0)
            else:
                val_cond = epoch % args.eval_num == 0

            mean_train_loss = np.mean(loss_train)
            mean_train_loss_recon = np.mean(loss_train_recon)
            if val_cond:
                val_loss, val_loss_recon = self.validation(model, args, val_loader, loss_function)

                if val_loss_recon < val_best:
                    try:
                        wandb.alert(
                            title='Pretraining val_loss_recon < val_best',
                            text='Pretraining val_loss_recon < val_best. Saved new model. val_loss_recon: ' + str(
                                val_loss_recon),
                            level=AlertLevel.WARN,
                            wait_duration=timedelta(minutes=15)
                        )
                    except Exception as e:
                        logging.error(e)
                    val_best = val_loss_recon
                    self.metrics["len_val_loader"] = val_loader.__len__()
                    self.metrics["len_train_loader"] = train_loader.__len__()
                    self.metrics["train_loss"] = mean_train_loss
                    self.metrics["train_loss_recon"] = mean_train_loss_recon
                    self.metrics["val_loss"] = val_loss
                    self.metrics["val_loss_recon"] = val_loss_recon
                    self.metrics["best_val_loss_recon"] = val_best
                    try:
                        self.metrics["current_cache_size_val_loader"] = int(len(val_loader.dataset.cache))
                        self.metrics["current_tensor_cache_size_val_loader"] = int(len(val_loader.dataset.tensor_cache))
                    except Exception as e:
                        logging.error(e)

                    self.best_metrics = copy.deepcopy(self.metrics)
                    self.save_model(model, self.best_metrics, self.best_metrics_file_path, self.best_weights_file_path)
                    self.save_checkpoint()
                    print(
                        "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                            val_best, val_loss_recon
                        )
                    )
                else:
                    print(
                        "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                            val_best, val_loss_recon
                        )
                    )
            if step < 50 and self.img_input is not None:
                wandb.log({"img_input_samples": wandb.Image(self.img_input)}, commit=False)

                if self.img_recon is not None:
                    wandb.log({"img_recon": wandb.Image(self.img_recon)}, commit=False)
                if self.img_recon2 is not None:
                    wandb.log({"img_recon2": wandb.Image(self.img_recon2)}, commit=False)
                if self.img_x1 is not None:
                    wandb.log({"img_x1": wandb.Image(self.img_x1)}, commit=False)
                if self.img_x2 is not None:
                    wandb.log({"img_x2": wandb.Image(self.img_x2)}, commit=False)

            wandb_data: dict = {}
            wandb_data.update(self.metrics)
            wandb.log(wandb_data)
            # step +1
        epoch += 1
        return epoch, loss, val_loss, val_loss_recon, val_best, mean_train_loss, mean_train_loss_recon

    def validation(self, model, args, val_loader: DataLoader, loss_function):
        model.eval()
        loss_val = []
        loss_val_recon = []
        val_dataset: Dataset = val_loader.dataset
        max_index: int = min(val_dataset.__len__() - 1, self.train_config["pretrain_max_val_batches"] -1)
        val_steps = min(20, max_index)
        val_subset = torch.utils.data.Subset(val_loader.dataset, random.sample(range(0, max_index), min(val_steps, max_index)))
        sub_val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        with torch.no_grad():
            for step, batch in enumerate(sub_val_loader):
                batch_dict_x = batch
                x = batch_dict_x["tensor_image"]
                x = x.to(device)

                # such as mentioned in the paper: labels are not used for pre-training stage
                x1, rot1 = rot_rand(args, x)
                x2, rot2 = rot_rand(args, x)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)

                    loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

                try:
                    self.metrics["val_step_total_loss"] = float(loss.cpu().detach().numpy())
                    self.metrics["val_step_rot_loss"] = float(losses_tasks[0].cpu().detach().numpy())
                    self.metrics["val_step_contrast_loss"] = float(losses_tasks[1].cpu().detach().numpy())
                    self.metrics["val_step_recon_loss"] = float(losses_tasks[2].cpu().detach().numpy())
                except Exception as e:
                    logging.error(e)

                loss_recon = losses_tasks[2]
                loss_val.append(loss.item())
                loss_val_recon.append(loss_recon.item())

                print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_recon))

        return np.mean(loss_val), np.mean(loss_val_recon)

    def save_model(self, model: SSLHead, metrics: dict, metrics_filepath, weights_filepath):
        save_dict_as_json(metrics, None, metrics_filepath)
        torch.save(model.swinViT.state_dict(), weights_filepath)  # only encoder weights (model.swinViT)
        print("Model saved to dir:", weights_filepath)

    def run_training(self, train_dataset, val_dataset):
        if self.training_running:
            return

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
        wandb_parameters: dict = vars(self.args)
        for key in wandb_parameters:
            self.wandb_sweep_config["parameters"][key] = {"value": wandb_parameters[key]}
        for key in self.train_config:
            self.wandb_sweep_config["parameters"][key] = {"value": self.train_config[key]}

        self.checkpoint_loaded = False
        if CHECKPOINT_WAIT_TIME > 0:
            countdown(sec=CHECKPOINT_WAIT_TIME,
                      optional_text="Giving time to upload torch model checkpoint to "
                                    + self.checkpoint_path + " e.g. using post request",
                      cancel_condition_function=self.checkpoint_exists)
            self.checkpoint_loaded: bool = self.load_checkpoint()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.run = wandb.init(project="swin-t_ssl", config=self.wandb_sweep_config, resume=self.checkpoint_loaded,
                              id=self.run_id)
        wandb.watch(self.model)

        model_artifact = wandb.Artifact('model', type='model')
        self.training_run_counter += 1
        self.training_running = True
        args = self.args

        scheduler = None
        if args.lrdecay:
            if args.lr_schedule == "warmup_cosine":
                scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

            elif args.lr_schedule == "poly":

                def lambdas(epoch):
                    return (1 - float(epoch) / float(args.epochs)) ** 0.9

                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambdas)

        loss_function = Loss(args.batch_size * args.sw_batch_size, args)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank])

        # def worker_init_fn(worker_id):
        #    random.seed(args.seed + worker_id)

        num_workers = 1
        # num_workers = 8
        # if os.name == 'nt':
        #    num_workers = 0

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        logging.info("train_dataset.length " + str(train_dataset.length))
        logging.info("val_dataset.length " + str(val_dataset.length))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        epoch = 0
        best_val = 1e8
        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None

        while epoch < args.epochs:
            try:
                epoch_start_time = time.time()
                self.epoch, loss, val_loss, val_loss_recon, best_val, mean_train_loss, mean_train_loss_recon = self.train(
                    self.model,
                    args,
                    self.epoch,
                    train_loader,
                    val_loader,
                    best_val,
                    scaler,
                    self.optimizer,
                    loss_function,
                    scheduler,
                    self.log_dir)

                self.metrics["epoch_time"] = float(time.time() - epoch_start_time)
                self.metrics["train_loss"] = mean_train_loss
                self.metrics["train_loss_recon"] = mean_train_loss_recon
                self.metrics["val_loss"] = val_loss
                self.metrics["val_loss_recon"] = val_loss_recon
                self.metrics["best_val_loss_recon"] = best_val
                self.latest_metrics = copy.deepcopy(self.metrics)
                self.save_checkpoint()
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
                self.save_checkpoint()
                try:
                    wandb.alert(
                        title='Pretraining Exception',
                        text=str(e),
                        level=AlertLevel.ERROR,
                        wait_duration=timedelta(minutes=15)
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
        self.run.log_artifact(model_artifact, aliases=['latest', str(epoch)])
        self.training_running = False

    def make_backup_wandb_artifact(self):
        try:
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
                self.metrics.update(checkpoint['metrics'])
                self.train_config.update(checkpoint['train_config'])
                if "run_id" in checkpoint:
                    self.run_id = checkpoint["run_id"]
                if "global_step" in checkpoint:
                    self.global_step = checkpoint["global_step"]

                if str(MINIO_BUCKET_NAME) != self.train_config["minio_bucket_name"]:
                    raise Exception("MINIO_BUCKET_NAME is not the same. checkpoint: "
                                    + str(self.train_config["minio_bucket_name"]) +
                                    "env variable: " + str(MINIO_BUCKET_NAME))

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
