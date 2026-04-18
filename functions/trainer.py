import os
import torch
from tqdm import tqdm
from torch.optim import Adam
from models.backbone import backbone_unet
from datetime import datetime
from functions.data import FM_dataloader, fno_dataloader
from models.FM import FlowMatching
from torch.utils.tensorboard import SummaryWriter
from models.FNO import FNO2d
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F


# Flow Matching Trainer --> Inverse
class FMTrainer:

    def __init__(self,args):

        # functions params
        self.device = [args.device]
        
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.op_weight = args.op_weight
        self.cfg = args.cfg

        # backbone params
        self.num_res_blocks = args.num_res_blocks
        self.dropout = args.dropout
        self.num_heads = args.num_heads
        self.attention_resolutions = args.attention_resolutions
        self.channel_mult = args.channel_mult
        self.num_sensor_points = args.num_sensor_points

        # data prep
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle
        self.trainset_dir = f"data/{args.data_type}/trainset.h5"
        self.testset_dir = f"data/{args.data_type}/testset.h5"
        self.forward_operator = f"output/{args.data_type}-FNOForward/checkpoints/best.pt"
        
        # === prepare output folder
        running_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = f"output/{args.data_type}-FMInverse-op{self.op_weight}-{self.num_sensor_points}/logs"
        self.checkpoint_dir = f"output/{args.data_type}-FMInverse-op{self.op_weight}-{self.num_sensor_points}/checkpoints"

        # === Load FNO Operator
        fno_checkpoint = torch.load(self.forward_operator)
        fno_size_params = fno_checkpoint["size_params"]
        self.fno_model = FNO2d(
                        field_channels=fno_size_params["field_channels"],
                        obs_channels=fno_size_params["obs_channels"],
                        modes1=fno_checkpoint["fno_modes"],
                        modes2=fno_checkpoint["fno_modes"],
                        width=fno_checkpoint["fno_width"],)
        self.fno_model = self.fno_model.cuda(self.device[0])
        self.fno_model.load_state_dict(fno_checkpoint["model_state_dict"])
        print("-> Operator loaded successfully.")

        # === Dataloader
        self.train_loader, self.size_params = FM_dataloader(
            data_dir = self.trainset_dir,
            shuffle = self.shuffle,
            num_sensor_points = self.num_sensor_points,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
        print("-> Train dataloader loaded successfully.")

        self.test_loader,_ = FM_dataloader(
            data_dir = self.testset_dir,
            shuffle = self.shuffle,
            num_sensor_points = self.num_sensor_points,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
        print("-> Test dataloader loaded successfully.")


        # get in_channels, out_channels, model_channels
        self.field_channels = self.size_params["field_channels"]
        self.field_size = self.size_params["field_size"]
        self.obs_channels = self.size_params["obs_channel"]
        self.global_feat_size = self.size_params["global_feat_size"]
            
        print(" ***** Input Data Dimension *****\n"
              f"# Data (Task) type: {args.data_type}\n"
              f"# channel size: {self.field_channels}\n"
              f"# field size: {self.field_size}\n"
              f"# global feat size: {self.global_feat_size}\n"
              f"# obs channel: {self.obs_channels}\n")


        # Backbone Model Import
        self.backbone = backbone_unet(
            input_channels = self.field_channels,
            field_size = self.field_size,
            spatial_feat_channels = self.obs_channels,
            global_feat_size = self.global_feat_size,
            num_res_blocks = self.num_res_blocks,  # num of res block for each step
            attention_resolutions = self.attention_resolutions,  # Attn insertion at: 32x32
            dropout = self.dropout,
            channel_mult = self.channel_mult,  # channel multipliers
            num_heads = self.num_heads,  # num of heads,
            obs_num = self.num_sensor_points,
        )
        self.backbone = self.backbone.cuda(self.device[0])
        print("-> Backbone loaded successfully.")

        # Conditional Flow Matching
        self.model = FlowMatching(model = self.backbone)
        self.model = self.model.cuda(self.device[0])
        print("-> Conditional Flow Matching loaded successfully.")

        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # logging and save
        self.writer = SummaryWriter(self.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # training status
        self.current_epoch = 0
        self.best_test_loss = float('inf')

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for inputs in progress_bar:
            target_field = inputs["target"].to(self.device[0])
            mask = inputs["mask"].to(self.device[0])
            spatial_feat = inputs["spatial_feat"].to(self.device[0])
            global_feat = inputs["global_feat"].to(self.device[0])

            self.optimizer.zero_grad()

            v_pred, ut, x_hat, adaptive_physics_weight = self.model.forward(target_field, 
                                                                            spatial_feat, 
                                                                            global_feat, 
                                                                            mask, 
                                                                            self.op_weight, 
                                                                            self.cfg)

            loss_fm = F.mse_loss(v_pred, ut)

            dense_obs_pred = self.fno_model(x_hat)

            diff = (dense_obs_pred - spatial_feat) * mask.unsqueeze(1)
            loss_physics_raw = torch.sum(diff ** 2) / mask.sum()

            loss_physics_weighted = torch.mean(adaptive_physics_weight * (diff ** 2))

            if self.op_weight != 0.0:
                loss = loss_fm + loss_physics_weighted
            else:
                loss = loss_fm


            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, self.current_epoch)
        return avg_loss

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs in tqdm(self.test_loader, desc="Validating"):
                target_field = inputs["target"].to(self.device[0])
                mask = inputs["mask"].to(self.device[0])
                spatial_feat = inputs["spatial_feat"].to(self.device[0])
                global_feat = inputs["global_feat"].to(self.device[0])

                v_pred, ut, x_hat, adaptive_physics_weight = self.model.forward(target_field, 
                                                                                spatial_feat, 
                                                                                global_feat, 
                                                                                mask, 
                                                                                self.op_weight, 
                                                                                False,)


                loss_fm = F.mse_loss(v_pred, ut)
                dense_obs_pred = self.fno_model(x_hat)

                diff = (dense_obs_pred - spatial_feat) * mask.unsqueeze(1)
                loss_physics_raw = torch.sum(diff ** 2) / mask.sum()

                loss_physics_weighted = torch.mean(adaptive_physics_weight * (diff ** 2))
                
                loss = loss_fm + loss_physics_weighted
                
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar("Loss/val", avg_loss, self.current_epoch)
        return avg_loss

    def save_checkpoint(self, train_loss, test_loss, is_best):
        if is_best:
            torch.save({
                        'epoch': self.num_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'size_params': self.size_params,
                        'num_res_blocks': self.num_res_blocks,
                        'channel_mult': self.channel_mult,
                        'attention_resolutions': self.attention_resolutions,
                        'num_heads': self.num_heads,
                        'dropout':self.dropout,
                        'current_epoch':self.current_epoch,
                        }, self.checkpoint_dir+"/best.pt")
            print("Checkpoint saved - best.pt")

        else:
            torch.save({
                        'epoch': self.num_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'size_params': self.size_params,
                        'num_res_blocks': self.num_res_blocks,
                        'channel_mult': self.channel_mult,
                        'attention_resolutions': self.attention_resolutions,
                        'num_heads': self.num_heads,
                        'dropout':self.dropout,
                        'current_epoch':self.current_epoch,
                        }, self.checkpoint_dir+"/latest.pt")
            print("Checkpoint saved - latest.pt")

    def train(self):
        print("-> Training started.")
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_one_epoch()
            test_loss = self.validate_one_epoch()

            log_str = f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.4f}"
            log_str += f" | Test Loss: {test_loss:.4f}"

            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.save_checkpoint(train_loss, test_loss, is_best=True)
            print(log_str)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(train_loss, test_loss, is_best=False)

        self.writer.close()
        print("Training completed.")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.test_loss = checkpoint["test_loss"]
        self.train_loss = checkpoint["train_loss"]
        self.size_params = checkpoint["size_params"]
        self.num_res_blocks = checkpoint["num_res_blocks"]
        self.channel_mult = checkpoint["channel_mult"]
        self.attention_resolutions = checkpoint["attention_resolutions"]
        self.num_heads = checkpoint["num_heads"]
        self.dropout = checkpoint["dropout"]
        

        print(f"Loaded checkpoint from epoch {self.current_epoch}")





class FNOTrainer:

    def __init__(self, args):
        # functions params
        self.device = [i for i in range(args.device)]
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        
        # FNO modes
        self.fno_modes = args.fno_modes
        self.fno_width = args.fno_width
        
        # data prep
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle
        self.trainset_dir = f"data/{args.data_type}/trainset.h5"
        self.testset_dir = f"data/{args.data_type}/testset.h5"
        
        # === prepare output folder
        running_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = f"output/{args.data_type}-FNOForward/logs"
        self.checkpoint_dir = f"output/{args.data_type}-FNOForward/checkpoints"
        
        # === Dataloader ===
        self.train_loader, self.size_params = fno_dataloader(
            data_dir = self.trainset_dir,
            shuffle = self.shuffle,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
        print("-> Train dataloader loaded successfully.")

        self.test_loader, _ = fno_dataloader(
            data_dir = self.testset_dir,
            shuffle = self.shuffle,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
        print("-> Test dataloader loaded successfully.")

        self.field_channels = self.size_params["field_channels"]
        self.obs_channels = self.size_params["obs_channels"]

        self.model = FNO2d(
            field_channels=self.field_channels,
            obs_channels=self.obs_channels,
            modes1=self.fno_modes,
            modes2=self.fno_modes,
            width=self.fno_width
        )
        self.model = self.model.cuda(self.device[0])
        print("-> FNO Baseline loaded successfully.")

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        self.criterion = nn.MSELoss()

        # logging and save
        self.writer = SummaryWriter(self.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.current_epoch = 0
        self.best_test_loss = float('inf')

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for inputs in progress_bar:
            input_field = inputs["input_field"].to(self.device[0])
            label_obs = inputs["label_obs"].to(self.device[0])
    
            self.optimizer.zero_grad()

            # FNO
            obs_pred = self.model(input_field)
            # MSE Loss
            loss = self.criterion(obs_pred, label_obs)
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, self.current_epoch)
        return avg_loss

    def validate_one_epoch(self):
        if not self.test_loader:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs in tqdm(self.test_loader, desc="Validating"):
                input_field = inputs["input_field"].to(self.device[0])
                label_obs = inputs["label_obs"].to(self.device[0])

                obs_pred = self.model(input_field)
                loss = self.criterion(obs_pred, label_obs)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar("Loss/val", avg_loss, self.current_epoch)
        return avg_loss

    def save_checkpoint(self, train_loss, test_loss, is_best):
        save_dict = {
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'fno_modes': self.fno_modes,
            'fno_width': self.fno_width,
            'current_epoch': self.current_epoch,
            'size_params': self.size_params,
        }
        if is_best:
            torch.save(save_dict, self.checkpoint_dir+"/best.pt")
            print("Checkpoint saved - best.pt")
        else:
            torch.save(save_dict, self.checkpoint_dir+"/latest.pt")
            print("Checkpoint saved - latest.pt")

    def train(self):
        print("-> FNO Training started.")
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_one_epoch()
            test_loss = self.validate_one_epoch()

            log_str = f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.save_checkpoint(train_loss, test_loss, is_best=True)
            print(log_str)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(train_loss, test_loss, is_best=False)

        self.writer.close()
        print("Training completed.")