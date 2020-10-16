import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import torchvision.transforms.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils.utils import Timer

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger()


def get_data_loaders(world_size, rank, batch_size, num_workers):
    dataset_directory = "experiments/datasets" 

    NORMALIZE_CIFAR10 = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.ColorJitter(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          NORMALIZE_CIFAR10])

    train_set = torchvision.datasets.CIFAR10(dataset_directory, download=True, transform=train_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=world_size,
                                                                    rank=rank)

    train_set_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=False,
                                                   sampler=train_sampler,
                                                   pin_memory=True)



    test_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR10])
    test_set = torchvision.datasets.CIFAR10(dataset_directory, train=False, transform=test_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=False,
                                                  pin_memory=True)

    return train_set_loader, test_set_loader

def train_model(device, model, train_set_loader, optimizer):
    timer = Timer().start()
    model.train() # For special layers
    total = 0
    correct = 0
    total_loss = 0
    for images, targets in train_set_loader:
        total += images.shape[0]
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        loss = F.cross_entropy(output, targets, reduction='mean')
        total_loss += torch.sum(loss)
        loss.backward()
        optimizer.step()
        # logger.info(f"Batch Loss: {loss}")

        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()

    average_train_loss = total_loss / total
    accuracy = 100. * correct.item()/total
    logger.info(f"Training Took {timer.stop():0.2f}s. Images in epoch: {total} ")

    return average_train_loss, accuracy

def test_model(device, model, test_set_loader, optimizer):
    timer = Timer().start()
    model.eval() # For special layers
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in test_set_loader:
            total += images.shape[0]

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

    accuracy = 100. * correct.item()/total
    logger.info(f"Testing Took {timer.stop():0.2f}s. Images in epoch: {total}")

    return accuracy

def main(device, mp_args, experiment_name, optimizer, output_directory_root="experiments/resnet18_distributed",
         lr_scheduler=None, epochs=150, batch_size=512, num_workers=1):    

    global_rank = mp_args.nr * mp_args.gpus + device
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=mp_args.world_size, rank=global_rank) 

    output_directory = os.path.join(output_directory_root, experiment_name, f"rank_{global_rank}")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Setup regular log file + tensorboard
    logfile_path = os.path.join(output_directory, "logfile.txt")
    setup_logger_tqdm(logfile_path)

    tensorboard_log_directory = os.path.join("runs", "resnet18_distributed", experiment_name, f"rank_{global_rank}")
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

    # Datasets and Loaders
    train_set_loader, test_set_loader = get_data_loaders(mp_args.world_size, global_rank, batch_size, num_workers)

    # Create Model & Optimizer (uses Partial Functions)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.to(device)
    optimizer = optimizer(model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer)

    logger.info("=========== Commencing Training ===========")
    logger.info(f"Epoch Count: {epochs}")
    logger.info(f"Batch Size: {batch_size}")

    # Load Checkpoint
    checkpoint_file_path = os.path.join(output_directory, "checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_file_path):
        logger.info("Checkpoint Found - Loading!")
        checkpoint = torch.load(checkpoint_file_path)
        logger.info(f"Last completed epoch: {checkpoint['epoch']}")
        logger.info(f"Average Train Loss: {checkpoint['train_loss']}")
        logger.info(f"Top-1 Train Accuracy: {checkpoint['train_accuracy']}")
        logger.info(f"Top-1 Test Accuracy: {checkpoint['test_accuracy']}")
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming at epoch {start_epoch}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    else:
        logger.info("No checkpoint found, starting from scratch.")

    # Training Loop
    t = Timer()
    for epoch in range(start_epoch, epochs):
        t.start()
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch}")
        logger.info("-" * 10)

        train_loss, train_accuracy = train_model(device, model, train_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        
        test_accuracy = test_model(device, model, test_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

        scheduler_dict = None
        if lr_scheduler:
            lr_scheduler.step()
            scheduler_dict = lr_scheduler.state_dict()

        # Save Checkpoint
        logger.info("Saving checkpoint.")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': scheduler_dict,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
            }, checkpoint_file_path)

        elapsed_time = t.stop()
        logger.info(f"End of epoch {epoch}, took {elapsed_time:0.4f} seconds.")
        logger.info(f"Average Train Loss: {train_loss}")
        logger.info(f"Top-1 Train Accuracy: {train_accuracy}")
        logger.info(f"Top-1 Test Accuracy: {test_accuracy}")
        logger.info("")

# Lambdas can't pickle...
def main_adamw(device, mp_args):
    experiment_name = "adamw_default"
    optimizer = lambda x : torch.optim.AdamW(x)
    main(device, mp_args, experiment_name, optimizer)

parser_description = 'Simple example of pytorch distributed using resnet18 and cifar10.'
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')

if __name__ == '__main__':
    assert(torch.cuda.is_available())
    assert(torch.distributed.is_available())

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    for experiment in adamw_experiments:
        mp.spawn(main_adamw, nprocs=args.gpus, args=(args,))