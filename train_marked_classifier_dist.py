import torchvision
import torchvision.transforms.transforms as transforms
import torch
import logging
from torch.nn import functional as F
import os
import re
import shutil
from utils import Timer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from dataset_wrappers import MergedDataset
from utils import NORMALIZE_CIFAR, NORMALIZE_IMAGENET, NORMALIZE_IMAGENETTE

from logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

# Datasets use pickle, so we can't just pass in a lambda
def numpy_loader(x):
    return transforms.ToPILImage()(np.load(x))

#def get_data_loaders_cifar10(marked_images_directory, augment, batch_size=512, num_workers=1):

#    cifar10_dataset_root = "experiments/datasets" # Will download here

#    # Base Training Set
#    base_train_set = torchvision.datasets.CIFAR10(cifar10_dataset_root, download=True)

#    # Load marked data from Numpy img format - no transforms
#    extensions = ("npy")
#    marked_images = torchvision.datasets.DatasetFolder(marked_images_directory, numpy_loader, extensions=extensions)

#    # Setup Merged Training Set: Vanilla -> Merged <- Marked
#    # MergedDataset allows you to replace certain examples with marked alternatives
#    merge_to_vanilla = [None] * len(marked_images)
#    for i, (path, target) in enumerate(marked_images.samples):
#        img_id = re.search('[0-9]+', os.path.basename(path))
#        merge_to_vanilla[i] = int(img_id[0])

#    merged_train_set = MergedDataset(base_train_set, marked_images, merge_to_vanilla)

#    # Add Transform and Get Training set dataloader
#    transforms_list = []
#    if augment:
#        transforms_list += [transforms.RandomCrop(32, padding=4),
#                            transforms.RandomHorizontalFlip()]
    
#    transforms_list += [transforms.ToTensor(), NORMALIZE_CIFAR]
    
#    train_transform = transforms.Compose(transforms_list)
#    merged_train_set.transform = train_transform

#    train_set_loader = torch.utils.data.DataLoader(merged_train_set,
#                                                   batch_size=batch_size,
#                                                   num_workers=num_workers,
#                                                   shuffle=True,
#                                                   pin_memory=True)

#    # Test Set (Simple)
#    test_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR])
#    test_set = torchvision.datasets.CIFAR10(cifar10_dataset_root, train=False, transform=test_transform)
#    test_set_loader = torch.utils.data.DataLoader(test_set, 
#                                                  batch_size=batch_size, 
#                                                  num_workers=num_workers, 
#                                                  shuffle=False,
#                                                  pin_memory=True)

#    return train_set_loader, test_set_loader

def data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, normalizer, 
                                     batch_size, num_workers, world_size, rank):

    # Base Training Set
    base_train_set = torchvision.datasets.ImageFolder(train_images_path)

    # Load marked data from Numpy img format - no transforms
    extensions = ("npy")
    marked_images = torchvision.datasets.DatasetFolder(marked_images_directory, numpy_loader, extensions=extensions)

    # Setup Merged Training Set: Vanilla -> Merged <- Marked
    # MergedDataset allows you to replace certain examples with marked alternatives
    merge_to_vanilla = [None] * len(marked_images)
    for i, (path, target) in enumerate(marked_images.samples):
        img_id = re.search('[0-9]+', os.path.basename(path))
        merge_to_vanilla[i] = int(img_id[0])

    merged_train_set = MergedDataset(base_train_set, marked_images, merge_to_vanilla)

    # Add Transform, sampler and get loader
    train_transform = transforms.Compose([transforms.RandomCrop(256, pad_if_needed=True),
                                          transforms.ColorJitter(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalizer])    

    merged_train_set.transform = train_transform

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train_set,
                                                                    num_replicas=world_size,
                                                                    rank=rank)

    train_set_loader = torch.utils.data.DataLoader(merged_train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=False,
                                                   sampler=train_sampler,
                                                   pin_memory=True)

    # Test
    test_transform = transforms.Compose([transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         normalizer])
    test_set = torchvision.datasets.ImageFolder(test_images_path, transform=test_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=False,
                                                  pin_memory=True)

    return train_set_loader, test_set_loader

def get_data_loaders_imagenette(train_images_path, test_images_path, marked_images_directory, 
                                batch_size, num_workers, world_size, rank):
    normalizer = NORMALIZE_IMAGENETTE
    return data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, 
                                            normalizer, batch_size, num_workers, world_size, rank)

def get_data_loaders_imagenet(train_images_path, test_images_path, marked_images_directory, 
                              batch_size, num_workers, world_size, rank):
    normalizer = NORMALIZE_IMAGENET
    return data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, 
                                            normalizer, batch_size, num_workers, world_size, rank)

def train_model(device, model, train_set_loader, optimizer):
    model.train() # For special layers
    total = 0
    correct = 0
    total_loss = 0
    for images, targets in tqdm(train_set_loader, desc="Training"):
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
    accuracy = 100. * correct.item() / total

    return average_train_loss, accuracy

def test_model(device, model, test_set_loader, optimizer):
    model.eval() # For special layers
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in tqdm(test_set_loader, desc="Testing"):
            total += images.shape[0]

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

    accuracy = 100. * correct.item() / total
    return accuracy

# Can't pickle lambdas...
def adamw_logistic(model):
    return torch.optim.AdamW(model.fc.parameters())

def main(device, mp_args, dataloader_func, model, optimizer_callback, output_directory, tensorboard_log_directory, 
         epochs):

    global_rank = mp_args.nr * mp_args.gpus + device
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=mp_args.world_size, rank=global_rank) 

    output_directory = os.path.join(output_directory, f"rank_{global_rank}")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Setup regular log file
    logfile_path = os.path.join(output_directory, "logfile.txt")
    setup_logger_tqdm(logfile_path)

    # Setup TensorBoard logging
    tensorboard_log_directory = os.path.join(tensorboard_log_directory, f"rank_{global_rank}")
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

    # Dataloaders
    train_set_loader, test_set_loader = dataloader_func(mp_args.world_size, global_rank)

    # Model & Optimizer
    model.to(device)
    optimizer = optimizer_callback(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    logger.info(f"Epoch Count: {epochs}")

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
    else:
        logger.info("No checkpoint found, starting from scratch.")

    # Training Loop
    t = Timer()
    progress = tqdm(total=epochs, initial=start_epoch, desc="Epochs")
    for epoch in range(start_epoch, epochs):
        t.start()
        logger.info(f"Commence EPOCH {epoch}")

        # Train
        train_loss, train_accuracy = train_model(device, model, train_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        
        # Test
        test_accuracy = test_model(device, model, test_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

        # Save Checkpoint
        logger.info("Saving checkpoint.")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
            }, checkpoint_file_path)

        elapsed_time = t.stop()
        logger.info(f"End of epoch {epoch}, took {elapsed_time:0.4f} seconds.")
        logger.info(f"Average Train Loss: {train_loss}")
        logger.info(f"Top-1 Train Accuracy: {train_accuracy}")
        logger.info(f"Top-1 Test Accuracy: {test_accuracy}")
        progress.update()
    progress.close()

from functools import partial

if __name__ == '__main__':
    """
    Basic Example
    You will need to blow away the TensorBoard logs and checkpoint file if you want
    to train from scratch a second time.
    """
    marked_images_directory = "experiments/radioactive/marked_images"
    output_directory="experiments/radioactive/train_marked_classifier"
    tensorboard_log_directory="runs/train_marked_classifier"
    optimizer = lambda model : torch.optim.AdamW(model.parameters())
    epochs = 60
    dataloader_func = partial(get_data_loaders_cifar10, marked_images_directory, False)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    main(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory,
         epochs=epochs)