import os
import numpy as np

import torchvision
import torchvision.transforms.transforms as transforms
import torch
from torch.nn import functional as F

from utils import Timer
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import logging
from logger import setup_logger_tqdm
logger = logging.getLogger()

import tqdm

normalize_imagenette = transforms.Normalize(mean=[0.4618, 0.4571, 0.4288], std=[0.2531, 0.2472, 0.2564])

def get_mean_and_std(train_images_path, test_images_path):

    train_set = torchvision.datasets.ImageFolder(train_images_path, transform=transforms.ToTensor())
    train_set_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=True)

    test_set = torchvision.datasets.ImageFolder(test_images_path, transform=transforms.ToTensor())
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=1, 
                                                  shuffle=False,
                                                  pin_memory=True)

    image_count = 0
    mean = 0.
    var = 0.
    for batch, _ in tqdm.tqdm(train_set_loader):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)

        # Update total number of images
        image_count += batch.size(0)

        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0) 

    for batch, _ in tqdm.tqdm(test_set_loader):
        batch = batch.view(batch.size(0), batch.size(1), -1)
        image_count += batch.size(0)
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0) 

    # Final step
    mean /= image_count
    var /= image_count
    std = torch.sqrt(var)

    print(mean)
    print(std)

def get_data_loaders(batch_size, num_workers, train_images_path, test_images_path):
    # Train
    train_transform = transforms.Compose([transforms.RandomCrop(256, pad_if_needed=True),
                                          transforms.ColorJitter(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize_imagenette])

    train_set = torchvision.datasets.ImageFolder(train_images_path, transform=train_transform)

    train_set_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)


    # Test
    test_transform = transforms.Compose([transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         normalize_imagenette])
    test_set = torchvision.datasets.ImageFolder(test_images_path, transform=test_transform)
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
    for images, targets in tqdm.tqdm(train_set_loader, desc="Training", position=0):
        total += images.shape[0]
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        loss = F.cross_entropy(output, targets, reduction='mean')
        total_loss += torch.sum(loss)
        loss.backward()
        optimizer.step()

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
        for images, targets in tqdm.tqdm(test_set_loader, desc="Testing", position=0):
            total += images.shape[0]

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

    accuracy = 100. * correct.item()/total
    return accuracy

def main(optimizer, train_images_path, test_images_path, 
         output_directory, tensorboard_log_directory,
         lr_scheduler=None, epochs=60, batch_size=16, num_workers=1, test=True): 

    os.makedirs(output_directory, exist_ok=True)

    # Setup log file + tensorboard
    logfile_path = os.path.join(output_directory, "logfile.txt")
    setup_logger_tqdm(logfile_path)
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

    # Choose Training Device
    use_cuda = torch.cuda.is_available()
    logger.info(f"CUDA Available? {use_cuda}")
    device = "cuda" if use_cuda else "cpu"   

    # Datasets and Loaders
    train_set_loader, test_set_loader = get_data_loaders(batch_size, num_workers,
                                                         train_images_path, test_images_path)

    # Create Model & Optimizer (uses Partial Functions)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.to(device)
    optimizer = optimizer(model.parameters())

    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer)

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
    progress = tqdm.tqdm(total=epochs, initial=start_epoch, desc="Epochs", position=1)
    for epoch in range(start_epoch, epochs):
        t.start()
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch}")
        logger.info("-" * 10)

        train_loss, train_accuracy = train_model(device, model, train_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        
        if test:
            test_accuracy = test_model(device, model, test_set_loader, optimizer)
            tensorboard_summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)
        else:
            test_accuracy = "N/A"

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
        progres.update()
    progress.close()

# Starts off slow at batch size 8, maxes out on 980 ti around 18 seconds per batch
# Exploded after 552 batch size - setting to 512
def batch_size_linear_search(train_images_path, test_images_path):
    min = 8
    max = 64
    step_size = 2

    optimizer = lambda x : torch.optim.SGD(x, lr=0.1)
    experiment_name = "batch_size_linear_search"
    t = Timer()

    batch_size_times = {}
    for i, batch_size in enumerate(range(min, max, step_size)):
        t.start()
        main(experiment_name, optimizer, train_images_path, test_images_path, epochs=i+2, batch_size=batch_size, test=False)
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

if __name__ == '__main__':
    train_images_path = "E:/imagenette2/train"
    test_images_path = "E:/imagenette2/val"

    #batch_size_linear_search(train_images_path, test_images_path)

    experiment_name = "adamw_default"
    optimizer = lambda x : torch.optim.AdamW(x)
    epochs = 150
    output_directory = os.path.join("experiments", "resnet18_imagenette", experiment_name)
    tensorboard_log_directory = os.path.join("runs", "resnet18_imagenette", experiment_name)
    main(optimizer, train_images_path, test_images_path, output_directory, tensorboard_log_directory,
         epochs=epochs)

    #get_mean_and_std(train_images_path, test_images_path)