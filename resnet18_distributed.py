import os
import argparse

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import torchvision.transforms.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import Timer

import logging
from logger import setup_logger
logger = logging.getLogger()


# 980ti optimisation, added sequentially on batch size 32:
# 70 second epochs without any async or pinned memory
# 68 seconds with pinned memory
# 67.7 seconds with pinned memory and non_blocking=True on inputs and targets
# num_workers=8: 71-72 seconds
# num_workers=6: 66 seconds
# num_workers=4: 62 seconds
# num_workers=3: 58 seconds
# num_workers=2: 57.3 seconds
# num_workers=1: 53.4 seconds
# Conclusion: Use Asynchronous memory copying and pinned memory. 1 worker per gpu? Untested on multi gpu
# Guides:
# https://pytorch.org/docs/stable/notes/cuda.html
# https://www.reddit.com/r/pytorch/comments/ijvts0/pytorch_performance_tuning_guide/

# Optimizers Tested
# Adam
# AdamW
# Many variations of sgd, sgdm
# Learning rate finder and 1 cycle not tested yet
# https://www.fast.ai/2018/07/02/adam-weight-decay/
# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
# https://sgugger.github.io/the-1cycle-policy.html

# Batch Sizing
# https://medium.com/datadriveninvestor/batch-vs-mini-batch-vs-stochastic-gradient-descent-with-code-examples-cd8232174e14
# Ran a linear search on batchsize - code below. Over 544 blew up.

def get_data_loaders(world_size, rank, batch_size, num_workers):
    dataset_directory = "experiments/datasets" 

    normalize_cifar = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.ColorJitter(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize_cifar])

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



    test_transform = transforms.Compose([transforms.ToTensor(), normalize_cifar])
    test_set = torchvision.datasets.CIFAR10(dataset_directory, train=False, transform=test_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=False,
                                                  pin_memory=True)

    return train_set_loader, test_set_loader

def train_model(device, model, train_set_loader, optimizer):
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

    return average_train_loss, accuracy

def test_model(device, model, test_set_loader, optimizer):
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
    return accuracy

# A simple example of a resnet18 training on CIFAR10 to demonstrate ML training optimization
def main(device, mp_args, experiment_name, optimizer, output_directory_root="experiments/resnet18_on_cifar10",
         lr_scheduler=None, epochs=60, batch_size=64, num_workers=1):    

    global_rank = mp_args.nr * mp_args.gpus + device
    dist.init_process_group(backend='gloo', init_method='env://', 
                            world_size=mp_args.world_size, rank=global_rank) 

    output_directory = os.path.join(output_directory_root, experiment_name, f"rank_{global_rank}")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Setup regular log file + tensorboard
    logfile_path = os.path.join(output_directory, "logfile.txt")
    setup_logger(logfile_path)

    tensorboard_log_directory = os.path.join("runs", experiment_name, f"rank_{global_rank}")
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
        
# Starts off slow at batch size 8, maxes out on 980 ti around 18 seconds per batch
# Exploded after 552 batch size - setting to 512
def batch_size_linear_search():
    min = 8
    max = 600
    step_size = 8

    optimizer = lambda x : torch.optim.SGD(x, lr=0.1)
    experiment_name = "batch_size_linear_search"
    t = Timer()

    batch_size_times = {}
    for i, batch_size in enumerate(range(min, max, step_size)):
        t.start()
        main(experiment_name, optimizer, epochs=i+2, batch_size=batch_size)
        elapsed_time = t.stop()
        batch_size_times[batch_size] = elapsed_time

    pickle.dump(batch_size_times, open("batch_size_times.pickle","wb"))

    # Plot
    batch_sizes = []
    times = []
    for k in sorted(batch_size_times):
        batch_sizes.append(k)
        times.append(batch_size_times[k])

    plt.plot(np.array(batch_sizes), np.array(times))
    plt.xlabel("Batch Size")
    plt.ylabel("Epoch Time")
    plt.title("Batch Size vs Epoch Time")
    plt.show()    


# Vanilla SGD
def experiment1():
    experiment_name = "sgd1"
    optimizer = lambda x : torch.optim.SGD(x, lr=1)
    main(experiment_name, optimizer)

def experiment2():
    experiment_name = "sgd0.1"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.1)
    main(experiment_name, optimizer)

def experiment3():
    experiment_name = "sgd0.01"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.01)
    main(experiment_name, optimizer)

def experiment4():
    experiment_name = "sgd0.005"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.005)
    main(experiment_name, optimizer)

def experiment5():
    experiment_name = "sgd0.001"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.001)
    main(experiment_name, optimizer)

# SGD With Momentum
def experiment6():
    experiment_name = "sgd1mom0.9"
    optimizer = lambda x : torch.optim.SGD(x, lr=1, momentum=0.9)
    main(experiment_name, optimizer)

def experiment7():
    experiment_name = "sgd0.1mom0.9"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9)
    main(experiment_name, optimizer)

def experiment8():
    experiment_name = "sgd0.01mom0.9"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.01, momentum=0.9)
    main(experiment_name, optimizer)

def experiment9():
    experiment_name = "sgd0.005mom0.9"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.005, momentum=0.9)
    main(experiment_name, optimizer)

def experiment10():
    experiment_name = "sgd0.001mom0.9"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9)
    main(experiment_name, optimizer)

# Adam
def experiment11():
    experiment_name = "adam_default"
    optimizer = lambda x : torch.optim.Adam(x)
    main(experiment_name, optimizer)

def experiment12(device, mp_args):
    experiment_name = "adamw_default"
    optimizer = lambda x : torch.optim.AdamW(x)
    main(device, mp_args, experiment_name, optimizer)

# SGD + momentum with 1 Cycle Schedule, 150 epoch
# https://sgugger.github.io/the-1cycle-policy.html
# https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
# Only stepping after each epoch, not batch
def experiment13():
    experiment_name = "sgd0.01mom0.9_1cycle_150epoch"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.01, momentum=0.9)
    epochs = 150
    total_steps = epochs
    start_lr = 0.001
    max_lr = 0.01
    div_factor = max_lr / start_lr
    pct_start = 0.3 # Default
    final_div_factor = 0.0001 # Default
    anneal_strategy = "linear"
    base_momentum = 0.85 # Default
    minimum_momentum = 0.95 # Default
    cycle_momentum = True # Default

    #lr_scheduler = lambda optim, last_epoch : torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, total_steps=total_steps, 
    #                                                                              anneal_strategy=anneal_strategy, div_factor=div_factor,
    #                                                                              last_epoch=last_epoch)

    lr_scheduler = lambda x : torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, total_steps=total_steps, 
                                                                  anneal_strategy=anneal_strategy, div_factor=div_factor)

    main(experiment_name, optimizer, lr_scheduler=lr_scheduler, epochs=epochs)

def experiment14():
    experiment_name = "sgd0.01mom0.9_1cycle_75epoch"
    optimizer = lambda x : torch.optim.SGD(x, lr=0.01, momentum=0.9)
    epochs = 75
    total_steps = epochs
    start_lr = 0.001
    max_lr = 0.01
    div_factor = max_lr / start_lr
    pct_start = 0.3 # Default
    final_div_factor = 0.0001 # Default
    anneal_strategy = "linear"
    base_momentum = 0.85 # Default
    minimum_momentum = 0.95 # Default
    cycle_momentum = True # Default

    lr_scheduler = lambda x : torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, total_steps=total_steps, 
                                                                  anneal_strategy=anneal_strategy, div_factor=div_factor)

    main(experiment_name, optimizer, lr_scheduler=lr_scheduler, epochs=epochs)

# Conclusions:
# Vanilla SGD only works well with optimal learning rate. Around 0.1 in this case.
# Adding momentum 0.9 mostly lets you ignore learning rate, but if you combine mom=0.9 with lr=0.1 you get
# slightly better test accuracy result.
# lr=0.001 didn't fit training data at 150 epochs so it might be useful to do another 150-300 on this rate.
# Strangely, lr=1.0 works surprisingly well, converging slower, but beaten only by optimal mom and lr above
# Adam and AdamW are basically identical and beat all tested SGD and SGDM on accuracy and convergence speed
# Can't test 1 cycle (without writing it myself) until I upgrade pytorch version to recent
def experiment():
    assert(torch.cuda.is_available())
    assert(torch.distributed.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    #batch_size_linear_search()
    vanilla_sgd_experiments = [experiment1, experiment2, experiment3, experiment4, experiment5]
    sgd_momentum_experiments = [experiment6, experiment7, experiment8, experiment9, experiment10]
    adam_experiments = [experiment11]
    adamw_experiments = [experiment12]
    one_cycle_experiments = [experiment13, experiment14]

    #for experiment in vanilla_sgd_experiments:
    #    experiment()
    
    #for experiment in sgd_momentum_experiments:
    #    experiment()

    #for experiment in adam_experiments:
    #    experiment()

    for experiment in adamw_experiments:
        mp.spawn(experiment, nprocs=args.gpus, args=(args,))

    # for experiment in one_cycle_experiments:
    #     experiment()

if __name__ == '__main__':
    experiment()