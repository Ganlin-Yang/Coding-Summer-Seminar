from ast import parse
import os, argparse
from data_loader import VimeoDataset, KodakDataset
import model
from trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import RateDistortionLoss
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default='/data1/liubj/vimeo_1')
    parser.add_argument("--lmbda", type=float, default=1e-2, help="weights for distoration.")
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_dataset", default='/data1/liubj/Kodak24')
    parser.add_argument("--test_batch_size", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--check_time", type=float, default=10, help='frequency for recording state (min).')
    parser.add_argument("--prefix", type=str, default='FactorizedPrior',
                        help="prefix of checkpoints/logger, etc. e.g. FactorizedPrior, HyperPrior")
    parser.add_argument("--model_name", default='Factorized',
                        help="other implemntation: 'Hyperprior', 'JointAutoregressiveHierarchicalPriors', 'CheckerboardAutogressive'")
    parser.add_argument("--exp_version", default=0, type=int,
                        help='target an experimental version to avoid overwritting')
    # if you want to use 1 gpu, just --gpu_id '0', if 3 gpus are needed, pass --gpu_id '0 1 2' for example
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited")
    parser.add_argument("--code_rate", type=str, default="low", help="choice of code_rate: 'low' or 'high'")
    args = parser.parse_args()
    return args


class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, lmbda, lr, check_time, device):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.lmbda = lmbda
        self.lr = lr
        self.check_time = check_time
        self.device = device


def set_optimizer(model, lr):
    params_lr_list = []
    for module_name in model._modules.keys():
        params_lr_list.append({"params": model._modules[module_name].parameters(), 'lr': lr})
    optimizer = torch.optim.Adam(params_lr_list)
    return optimizer

if __name__ == '__main__':
    # log
    args = parse_args()
    args.gpu_id = list(map(int, args.gpu_id.split()))
    # we select the first gpu as the main device
    device = torch.device("cuda:%d" % args.gpu_id[0]) if torch.cuda.is_available() else torch.device("cpu")
    training_config = TrainingConfig(
        logdir=os.path.join('./logs', args.prefix, str(args.exp_version)),
        ckptdir=os.path.join('./ckpts', args.prefix, str(args.exp_version)),
        init_ckpt=args.init_ckpt,
        lmbda=args.lmbda,
        lr=args.lr,
        check_time=args.check_time,
        device=device)
    # model
    assert args.model_name in ['Factorized', 'Hyperprior', 'JointAutoregressiveHierarchicalPriors',
                               'CheckerboardAutogressive']
    if args.code_rate == 'low':
        model_ = getattr(model, args.model_name)
    elif args.code_rate == 'high':
        model_ = getattr(model, args.model_name)(N=120, M=320)

    if len(args.gpu_id) > 1:
        model_ = torch.nn.DataParallel(model_, device_ids=args.gpu_id, dim=0)

    criterion = RateDistortionLoss(lmbda=args.lmbda)
    # trainer
    trainer = Trainer(config=training_config, model=model_, criterion=criterion)

    # dataset
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(256), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )

    train_dataset = VimeoDataset(args.dataset, 'train', train_transforms)
    test_dataset = KodakDataset(args.test_dataset, test_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,  # 丢弃最后一个batch
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # optimizer setting
    optimizer = set_optimizer(model_.to(training_config.device), args.lr)
    # lr update: Reduce learning rate when loss has stopped reducing.
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, verbose=True)
    # training
    for epoch in range(0, args.epoch):
        loss = trainer.train(train_dataloader, optimizer)
        lr_scheduler.step(loss)