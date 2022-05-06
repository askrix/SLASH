import os
import argparse
from datetime import datetime
import time

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

import scipy.optimize
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../../')

from dataGen import SHAPEWORLD4, get_loader
from slot_attention_module import SlotAttention_model
import model

import set_utils as set_utils
from rtpt import RTPT


import utils as misc_utils


#torch.set_num_threads(6)

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--ap-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--warm-up-epochs", type=int, default=10, help="Number of steps fpr learning rate warmup"
    )
    parser.add_argument(
        "--decay-epochs", type=int, default=10, help="Number of steps fpr learning rate decay"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["shapeworld4"],
        help="Use shapeworld4 dataset",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--export-dir", type=str, help="Directory to output samples to")
    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Use full evaluation set (default: 1/10 of evaluation data)",
        # don't need full evaluation when training to save some time
    )

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    parser.add_argument('--warmup-epochs', default=10, type=int,
	help='number of warmup epochs')

    args = parser.parse_args()

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda:0'

    misc_utils.set_manual_seed(args.seed)

    args.name += f'-{args.seed}'

    return args


def run(net, loader, optimizer, criterion, writer, args, train=False, epoch=0):
	if train:
		net.train()
		prefix = "train"
		torch.set_grad_enabled(True)

		# manual lr warmup
		if epoch < args.warm_up_epochs:
			learning_rate = args.lr * (epoch + 1) / args.warm_up_epochs
		else:
			learning_rate = args.lr
		learning_rate = learning_rate * 0.5 ** ((epoch + 1) / args.decay_epochs)
		optimizer.param_groups[0]["lr"] = learning_rate
		print("LR", learning_rate)


	else:
		net.eval()
		prefix = "test"
		torch.set_grad_enabled(False)

		preds_all = torch.zeros(0, args.n_slots, args.n_attr)
		target_all = torch.zeros(0, args.n_slots, args.n_attr)

	iters_per_epoch = len(loader)



	for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):

		# input is either a set or an image
		if 'cuda' in args.device:
			imgs, target_set = map(lambda x: x.cuda(), sample)
		else:
			imgs, target_set = sample

		output = net.forward(imgs)

		loss = set_utils.hungarian_loss(output, target_set)

		if train:
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()

			writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
			writer.add_scalar("lr/", optimizer.param_groups[0]["lr"], global_step=i)
		else:
			if i % iters_per_epoch == 0:

				preds_all = torch.cat((preds_all, output.detach().cpu()), 0)
				target_all = torch.cat((target_all, target_set.detach().cpu()), 0)

				ap = [
					set_utils.average_precision_clevr(preds_all.detach().cpu().numpy(),
					                                  target_all.detach().cpu().numpy(), d)
					for d in [-1]  # since we are not using 3D coords #[-1., 1., 0.5, 0.25, 0.125]
				]

				print(f"\nCurrent AP: ", ap[0], " %\n")
				writer.add_scalar("metric/ap_inf", ap[0], global_step=i)
				return ap
			writer.add_scalar("metric/val_loss", loss.item(), global_step=i)
	if train:
		print(f"Epoch {epoch} Train Loss: {loss.item()}")


def main():
	args = get_args()

	writer = SummaryWriter(f"runs/{args.name}",
	                       purge_step=0)
	# writer = None
	set_utils.save_args(args, writer)

	dataset_train = SHAPEWORLD4(
		args.data_dir, "train",
	)
	dataset_test = SHAPEWORLD4(
		args.data_dir, "val",
	)
	print('data loaded')

	if not args.eval_only:
		train_loader = get_loader(
			dataset_train, batch_size=args.batch_size
		)
	if not args.train_only:
		test_loader = get_loader(
			dataset_test,
			batch_size=5000,
			shuffle=False,
		)

	# print(torch.cuda.is_available())
	net = SlotAttention_model(n_slots=4, n_iters=3, n_attr=15,
	                                encoder_hidden_channels=32,
	                                attention_hidden_channels=64,
									mlp_prediction = True,
	                                device=args.device)

	args.n_attr = net.n_attr

	start_epoch = 0
	if args.resume:
		print("Loading ckpt ...")
		log = torch.load(args.resume)
		weights = log["weights"]
		net.load_state_dict(weights, strict=True)
		start_epoch = log["args"]["epochs"]

	if not args.no_cuda:
		net = net.cuda()

	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

	criterion = torch.nn.SmoothL1Loss()
	# Create RTPT object
	rtpt = RTPT(name_initials='DO', experiment_name=f"Shapeworld4 SetPrediction",
	            max_iterations=args.epochs)

	# store args as txt file
	set_utils.save_args(args, writer)

	# Start the RTPT tracking
	rtpt.start()

	start_time = time.time()
	ap_list = []

	for epoch in np.arange(start_epoch, args.epochs + start_epoch):
		if not args.eval_only:
			run(net, train_loader, optimizer, criterion, writer, args, train=True, epoch=epoch)
			cur_lr = optimizer.param_groups[0]["lr"]
			writer.add_scalar("lr", cur_lr, global_step=epoch * len(train_loader))
			# if args.resume is not None:
			rtpt.step()
		if not args.train_only:
			ap = run(net, test_loader, None, criterion, writer, args, train=False, epoch=epoch)
			ap_list.append(ap)
			if args.eval_only:
				exit()

		results = {
			"name": args.name,
			"weights": net.state_dict(),
			"args": vars(args),
			"ap_list": ap_list
		}
		# print(os.path.join("logs", args.name))
		torch.save(results, os.path.join("runs/", args.name, args.name))
		if args.eval_only:
			break

		print("total time", misc_utils.time_delta_now(start_time))


if __name__ == "__main__":
	main()
