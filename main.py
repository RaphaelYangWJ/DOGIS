import torch
import argparse
from functions.trainer import FMTrainer, FNOTrainer

'''
Secnario: Darcyflow, SHM
'''


# Training Params Parser
def parse_args_inverse():
    parser = argparse.ArgumentParser(description="Flow Matching Inversion Training Pipeline")
    # === functions params
    parser.add_argument("--device", type=int, default=0, help="GPU Selections")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--op_weight", type=float, default=0.0, help="Operator Loss Weight")
    parser.add_argument("--cfg", type=float, default=0.1, help="CFG Weight")
    # === Flow Matching params
    parser.add_argument("--num_res_blocks", type=int, default=4, help="Residual blocks volume per each resolution layer")
    parser.add_argument("--dropout", type=int, default=0.1, help="Dropout Ratio")
    parser.add_argument("--num_heads", type=int, default=8, help="Volume of attention heads")
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[64, 32, 16, 8], help='Volume of attention modules')
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 4, 8], help='Multiplications of Channels')
    parser.add_argument('--num_sensor_points', type=int, default=16, help='Num of random observation points')
    # === Dataset params
    parser.add_argument("--data_type", type=str, default="SHM", help="Dataset type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--shuffle", type=bool, default=True, help="Data shuffling")
    # === define args
    args = parser.parse_args()
    return args


def parse_args_forward():
    parser = argparse.ArgumentParser(description="FNO Forward Training Pipeline")
    # === functions params
    parser.add_argument("--device", type=int, default=torch.cuda.device_count(), help="GPU Selections")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=400, help="Epochs")
    # === FNO params
    parser.add_argument("--fno_modes", type=int, default=16, help="FNO Modes")
    parser.add_argument("--fno_width", type=int, default=64, help="FNO Width")
    # === Dataset params
    parser.add_argument("--data_type", type=str, default="Darcyflow", help="Dataset type") # [Darcyflow, SHM]
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--shuffle", type=bool, default=True, help="Data shuffling")
    # === define args
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    training_mode = "inverse"

    print(f"Training Pipeline Launch for [{training_mode}]")
    if training_mode == "inverse":
        # Inversion Training
        trainer_inverse = FMTrainer(args=parse_args_inverse())
        trainer_inverse.train()
    else:
        # Forward Training
        trainer_forward = FNOTrainer(args=parse_args_forward())
        trainer_forward.train()