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

def get_data_loaders_cifar10(marked_images_directory, augment, batch_size=512, num_workers=1):

    cifar10_dataset_root = "experiments/datasets" # Will download here

    # Base Training Set
    base_train_set = torchvision.datasets.CIFAR10(cifar10_dataset_root, download=True)

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

    # Add Transform and Get Training set dataloader
    transforms_list = []
    if augment:
        transforms_list += [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]
    
    transforms_list += [transforms.ToTensor(), NORMALIZE_CIFAR]
    
    train_transform = transforms.Compose(transforms_list)
    merged_train_set.transform = train_transform

    train_set_loader = torch.utils.data.DataLoader(merged_train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)

    # Test Set (Simple)
    test_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR])
    test_set = torchvision.datasets.CIFAR10(cifar10_dataset_root, train=False, transform=test_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=False,
                                                  pin_memory=True)

    return train_set_loader, test_set_loader

def data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, normalizer, batch_size=16, num_workers=1):

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

    # Add Transform and Get Training set dataloader
    train_transform = transforms.Compose([transforms.RandomCrop(256, pad_if_needed=True),
                                          transforms.ColorJitter(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalizer])    

    merged_train_set.transform = train_transform

    train_set_loader = torch.utils.data.DataLoader(merged_train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
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

def get_data_loaders_imagenette(train_images_path, test_images_path, marked_images_directory, batch_size=16, num_workers=1):
    normalizer = NORMALIZE_IMAGENETTE
    return data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, 
                                            normalizer, batch_size=batch_size, num_workers=num_workers)

def get_data_loaders_imagenet(train_images_path, test_images_path, marked_images_directory, batch_size=16, num_workers=1):
    normalizer = NORMALIZE_IMAGENET
    return data_loaders_imagenet_imagenette(train_images_path, test_images_path, marked_images_directory, 
                                            normalizer, batch_size=batch_size, num_workers=num_workers)

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class MarkedClassifier(pl.LightningModule):

    def __init__(self, model, optimizer_func, dataloader_func, batch_size=16, num_workers=1):
        super().__init__()
        self.model = model
        self.optimizer_func = optimizer_func
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = F.nll_loss(logits, targets)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = F.nll_loss(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.model)
        return optimizer

    def setup(self, stage=None):
        train_set_loader, test_set_loader = dataloader_func()
        self.train_set_loader = train_set_loader
        self.test_set_loader = test_set_loader

    def train_dataloader(self):
        return self.train_set_loader

    def val_dataloader(self):
        return self.test_set_loader

    def test_dataloader(self):
        return self.test_set_loader

#def main(dataloader_func, model, optimizer_callback, output_directory, tensorboard_log_directory, 
#         lr_scheduler=None, epochs=150):

#    model = MarkedClassifier("experiments/datasets")
#    trainer = pl.Trainer(gpus=1, max_epochs=60, progress_bar_refresh_rate=20)
#    trainer.fit(model)

#    if not os.path.isdir(output_directory):
#        os.makedirs(output_directory, exist_ok=True)

#    # Setup TensorBoard logging
#    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

#    # Choose Training Device
#    use_cuda = torch.cuda.is_available()
#    logger.info(f"CUDA Available? {use_cuda}")
#    device = "cuda" if use_cuda else "cpu"   

#    # Dataloaders
#    train_set_loader, test_set_loader = dataloader_func()

#    # Model & Optimizer
#    model.to(device)
#    optimizer = optimizer_callback(model)
#    if lr_scheduler:
#        lr_scheduler = lr_scheduler(optimizer)

#    logger.info(f"Epoch Count: {epochs}")

#    # Load Checkpoint
#    checkpoint_file_path = os.path.join(output_directory, "checkpoint.pth")
#    start_epoch = 0
#    if os.path.exists(checkpoint_file_path):
#        logger.info("Checkpoint Found - Loading!")

#        checkpoint = torch.load(checkpoint_file_path)
#        logger.info(f"Last completed epoch: {checkpoint['epoch']}")
#        logger.info(f"Average Train Loss: {checkpoint['train_loss']}")
#        logger.info(f"Top-1 Train Accuracy: {checkpoint['train_accuracy']}")
#        logger.info(f"Top-1 Test Accuracy: {checkpoint['test_accuracy']}")
#        start_epoch = checkpoint["epoch"] + 1
#        logger.info(f"Resuming at epoch {start_epoch}")

#        model.load_state_dict(checkpoint["model_state_dict"])
#        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#        if lr_scheduler:
#            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
#    else:
#        logger.info("No checkpoint found, starting from scratch.")

#    # Training Loop
#    t = Timer()
#    progress = tqdm(total=epochs, initial=start_epoch, desc="Epochs")
#    for epoch in range(start_epoch, epochs):
#        t.start()
#        logger.info(f"Commence EPOCH {epoch}")

#        # Train
#        train_loss, train_accuracy = train_model(device, model, train_set_loader, optimizer)
#        tensorboard_summary_writer.add_scalar("train_loss", train_loss, epoch)
#        tensorboard_summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        
#        # Test
#        test_accuracy = test_model(device, model, test_set_loader, optimizer)
#        tensorboard_summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

#        scheduler_dict = None
#        if lr_scheduler:
#            lr_scheduler.step()
#            scheduler_dict = lr_scheduler.state_dict()

#        # Save Checkpoint
#        logger.info("Saving checkpoint.")
#        torch.save({
#            'epoch': epoch,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'lr_scheduler_state_dict': scheduler_dict,
#            'train_loss': train_loss,
#            'train_accuracy': train_accuracy,
#            'test_accuracy': test_accuracy
#            }, checkpoint_file_path)

#        elapsed_time = t.stop()
#        logger.info(f"End of epoch {epoch}, took {elapsed_time:0.4f} seconds.")
#        logger.info(f"Average Train Loss: {train_loss}")
#        logger.info(f"Top-1 Train Accuracy: {train_accuracy}")
#        logger.info(f"Top-1 Test Accuracy: {test_accuracy}")
#        progress.update()
#    progress.close()

#from functools import partial

#if __name__ == '__main__':
#    """
#    Basic Example
#    You will need to blow away the TensorBoard logs and checkpoint file if you want
#    to train from scratch a second time.
#    """
#    marked_images_directory = "experiments/radioactive/marked_images"
#    output_directory="experiments/radioactive/train_marked_classifier"
#    tensorboard_log_directory="runs/train_marked_classifier"
#    optimizer = lambda model : torch.optim.AdamW(model.parameters())
#    epochs = 60
#    dataloader_func = partial(get_data_loaders_cifar10, marked_images_directory, False)
#    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

#    main(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory,
#         epochs=epochs)