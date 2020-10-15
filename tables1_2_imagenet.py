"""
1. Download the pytorch resnet18 pretrained on Imagenet.
2. Generate 1, 2, 5, 10% Markings using network from step 1 as marking network.
3. Create a logistic regression network on the end of the pretrained resnet from step 1.
   Train it using the radioactive data from step 2.
4. Perform the radioactive detection p-tests on the network trained in step 3. Compare the top-1 accuracy
   of this network with the network trained in step 1.
5. Generate Table 1. "Center Crop" augmentation makes no sense when using CIFAR10 data so this is skipped.
"""

import random
import os
import glob
import shutil
from functools import partial
import argparse

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cutie
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from make_data_radioactive import main as do_marking
from train_marked_classifier_lightning import MarkedClassifier, get_data_loaders_imagenet
from detect_radioactivity import main as detect_radioactivity
import differentiable_augmentations
from utils import NORMALIZE_IMAGENET

import logging
from logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

def get_images_for_marking_multiclass(training_set, tensorboard_log_directory, overall_marking_percentage):

    # Randomly sample images
    total_marked_images = int(len(training_set) * (overall_marking_percentage / 100))
    train_marked_indexes = random.sample(range(0, len(training_set)), total_marked_images)

    # Sort images into classes
    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    image_data = {class_id:[] for class_id in range(0, len(training_set.classes))}    
    for index in train_marked_indexes:
        image, label = training_set[index]
        image_data[label].append((transform(image).unsqueeze(0), index))

    # Save to tensorboard - sorted by class, original_index
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images = []
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    for class_id, image_list in image_data.items():
        if image_list:
            _, original_indexes = map(list, zip(*image_list))
            for index in original_indexes:
                image, label = training_set[index]
                images.append(transform(image))
    img_grid = torchvision.utils.make_grid(images, nrow=3)
    tensorboard_summary_writer.add_image('images_for_marking', img_grid)

    return image_data

def do_marking_run_multiclass(overall_marking_percentage, experiment_directory, tensorboard_log_directory,
                              marking_network, training_set):

    # Setup experiment directory
    if os.path.isdir(experiment_directory):
        error_message = f"Directory {experiment_directory} already exists. By default we assume you don't want to "\
                        "repeat the marking stage."
        logger.info(error_message)
        return

    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger_tqdm(filepath=logfile_path)

    # Carriers
    marking_network_fc_feature_size = 512
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))


    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    image_data = get_images_for_marking_multiclass(training_set,
                                                   tensorboard_log_directory,
                                                   overall_marking_percentage)

    marked_images = []
    for class_id, image_list in image_data.items():
        if image_list:
            images, original_indexes = map(list, zip(*image_list))
            optimizer = lambda x : torch.optim.AdamW(x)
            epochs = 250
            batch_size = 8
            output_directory = os.path.join(experiment_directory, "marked_images")
            augmentation = differentiable_augmentations.CenterCrop(256, 224)
            tensorboard_class_log = os.path.join(tensorboard_log_directory, f"class_{class_id}")
            marked_images_temp = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                                            class_id, NORMALIZE_IMAGENET, optimizer, tensorboard_class_log, epochs=epochs, 
                                            batch_size=batch_size, overwrite=False, augmentation=augmentation)
            
            marked_images =  marked_images + marked_images_temp   

    # Show marked images in Tensorboard - centercrop for grid
    from PIL import Image as im 
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    images_for_tensorboard = [transform(im.fromarray(x)) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=3)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

    # Record marking completion
    with open(os.path.join(experiment_directory, "marking.complete"),"w") as fh:
        fh.write("1")

def calculate_p_values(marking_percentages, marking_checkpoint_path, table_number, align):
    logfile_path = f"experiments/table{table_number}_imagenet/detect_radioactivity.log"
    setup_logger_tqdm(logfile_path)

    p_values = []

    # Load Marking Network and remove fully connected layer
    marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
    marking_checkpoint = torch.load(marking_checkpoint_path)
    marking_network.load_state_dict(marking_checkpoint["model_state_dict"])
    marking_network.fc = nn.Sequential()

    for run in marking_percentages:
        run_name = f"{run}_percent"
        carrier_path = f"experiments/table1_imagenet/{run_name}/carriers.pth"

        model = MarkedClassifier.load_from_checkpoint(PATH)

        target_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
        target_checkpoint_path = f"experiments/table{table_number}_imagenet/{run_name}/marked_classifier/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        target_network.load_state_dict(target_checkpoint["model_state_dict"])
        target_network.fc = nn.Sequential()
  
        (scores, p_vals, combined_pval) = detect_radioactivity(carrier_path, marking_network, 
                                                               target_network, target_checkpoint,
                                                               align=align)
        p_values.append(combined_pval)

    return p_values

def generate_table_1(marking_percentages, p_values, marking_checkpoint_path):
    # Get Vanilla Accuracy
    vanilla_checkpoint = torch.load(marking_checkpoint_path)

    # The Rest
    accuracies = [vanilla_checkpoint["test_accuracy"]]
    for run in marking_percentages:
        run_name = f"{run}_percent"
        target_checkpoint_path = f"experiments/table1_imagenette/{run_name}/marked_classifier/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        accuracies.append(target_checkpoint["test_accuracy"])

    formatted_accuracies = list(map(lambda x: f"{float(x):0.2f}", accuracies))

    # Create the table!
    column_labels = tuple([0] + marking_percentages)
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(column_labels)))
    row_labels = ["log10(p)", "Top-1 %"]
    formatted_pvalues = ["n/a"]
    formatted_pvalues += [f"{p:0.4f}" for p in np.log10(p_values)]

    cell_text = [formatted_pvalues, formatted_accuracies]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colColours=colors,
                     colLabels=column_labels,
                     loc='center')
    plt.savefig("experiments/table1_imagenette/table1.png")
    plt.show()

#def generate_table_2(marking_percentages, p_values, marking_checkpoint_path):
#    # Get Vanilla Accuracy
#    vanilla_checkpoint = torch.load(marking_checkpoint_path)

#    # The Rest
#    accuracies = [vanilla_checkpoint["test_accuracy"]]
#    for run in marking_percentages:
#        run_name = f"{run}_percent"
#        target_checkpoint_path = f"experiments/table2_imagenette/{run_name}/marked_classifier/checkpoint.pth"
#        target_checkpoint = torch.load(target_checkpoint_path)
#        accuracies.append(target_checkpoint["test_accuracy"])

#    formatted_accuracies = list(map(lambda x: f"{float(x):0.2f}", accuracies))

#    # Create the table!
#    column_labels = tuple([0] + marking_percentages)
#    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(column_labels)))
#    row_labels = ["log10(p)", "Top-1 %"]
#    formatted_pvalues = ["n/a"]
#    formatted_pvalues += [f"{p:0.4f}" for p in np.log10(p_values)]

#    cell_text = [formatted_pvalues, formatted_accuracies]

#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.axis('off')
#    table = ax.table(cellText=cell_text,
#                     rowLabels=row_labels,
#                     colColours=colors,
#                     colLabels=column_labels,
#                     loc='center')
#    plt.savefig("experiments/table2_imagenette/table2.png")
#    plt.show()

imagenet_classes = 10

def table_1_work(gpu_count, imagenet_path, step_3_batch_size):
    logger.info("Table 1 Preparation Commencing")
    logger.info("=============================")
    marking_percentages = [1, 2, 5]
    marking_percentages = [1]
    marking_percentages = [0.1]

    train_images_path = os.path.join(imagenet_path, "train")
    test_images_path = os.path.join(imagenet_path, "val")
    p_values_file = "experiments/table1_imagenet/p_values.pth"

    logger.info("")
    logger.info("Step 1 - Download Marking Network")
    logger.info("---------------------------------")
    marking_network = torchvision.models.resnet18(pretrained=True)

    logger.info("")
    logger.info("Step 2 - Image Marking")
    logger.info("----------------------")
    #training_set = torchvision.datasets.ImageFolder(train_images_path)
    #for marking_percentage in marking_percentages:
    #    experiment_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent")
    #    if os.path.exists(os.path.join(experiment_directory, "marking.complete")):
    #        message = f"Marking step already completed for {marking_percentage}%. Do you want to restart this part of " \
    #                  "the experiment?"
    #        if not cutie.prompt_yes_or_no(message, yes_text="Restart", no_text="Skip marking step"):
    #            continue

    #    tensorboard_log_directory = os.path.join("runs", "table1_imagenet", f"{marking_percentage}_percent", "marking")
    #    shutil.rmtree(experiment_directory, ignore_errors=True)
    #    shutil.rmtree(tensorboard_log_directory, ignore_errors=True)
    #    do_marking_run_multiclass(marking_percentage, experiment_directory, tensorboard_log_directory,
    #                              marking_network, training_set)

    logger.info("")
    logger.info("Step 3 - Training Target Networks")
    logger.info("---------------------------------")
    for marking_percentage in marking_percentages:

        marked_images_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent", "marked_images")
        output_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent", "marked_classifier")
        os.makedirs(output_directory, exist_ok=True)
        tensorboard_log_directory = os.path.join("runs", "table1_imagenet", f"{marking_percentage}_percent", "target")

        # Download a fresh pretrained resnet18
        model = torchvision.models.resnet18(pretrained=True)

        # Retrain the fully connected layer only
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, imagenet_classes)
        optimizer = lambda x : torch.optim.AdamW(x.fc.parameters())

        epochs = 2
        dataloader_func = partial(get_data_loaders_imagenet, 
                                  train_images_path, test_images_path, marked_images_directory, batch_size=step_3_batch_size)
   
        model = MarkedClassifier(model, optimizer, dataloader_func)
        #tb_logger = TensorBoardLogger(tensorboard_log_directory)
        trainer = pl.Trainer(default_root_dir=output_directory, #logger=tb_logger,
                             gpus=gpu_count, max_epochs=epochs, progress_bar_refresh_rate=20)
        trainer.fit(model)

    logger.info("")
    logger.info("Step 4 - Calculating p-values")
    logger.info("-----------------------------")
    p_values = calculate_p_values(marking_percentages, checkpoint_path, 1, False)  
    torch.save(p_values, p_values_file)
    p_values = torch.load(p_values_file)

    logger.info("")
    logger.info("Step 5 - Generating Table 1")
    logger.info("---------------------------")
    generate_table_1(marking_percentages, p_values, checkpoint_path)


# def table_2_work(imagenette_path, step_3_batch_size):
#     logger.info("")
#     logger.info("Table 2 Preparation Commencing")
#     logger.info("=============================")

#     marking_percentages = [1, 2, 5, 10, 20]

#     train_images_path = os.path.join(imagenette_path, "train")
#     test_images_path = os.path.join(imagenette_path, "val")
#     p_values_file = "experiments/table2_imagenette/p_values.pth"

#     logger.info("")
#     logger.info("Step 1 - Train Marking Network")
#     logger.info("------------------------------")
#     # Reuses marking network from Table 1 if available
#     optimizer = lambda x : torch.optim.AdamW(x)
#     epochs = 60
#     output_directory = os.path.join("experiments", "table1_imagenette", "marking_network")
#     checkpoint_path = os.path.join(output_directory, "checkpoint.pth") # Used later
#     tensorboard_log_directory = os.path.join("runs", "table1_imagenette", "marking_network")
#     resnet18_imagenette.main(optimizer, train_images_path, test_images_path, output_directory, tensorboard_log_directory, epochs=epochs)

#     marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
#     marking_network_checkpoint = torch.load(checkpoint_path)
#     marking_network.load_state_dict(marking_network_checkpoint["model_state_dict"])

#     logger.info("")
#     logger.info("Step 2 - Image Marking")
#     logger.info("----------------------")
#     # Reuses marked images from Table 1 if available
#     training_set = torchvision.datasets.ImageFolder(train_images_path)
#     for marking_percentage in marking_percentages:
#         experiment_directory = os.path.join("experiments", "table1_imagenette", f"{marking_percentage}_percent")
#         if os.path.exists(os.path.join(experiment_directory, "marking.complete")):
#             message = f"Marking step already completed for {marking_percentage}%. Do you want to restart this part of " \
#                       "the experiment?"
#             if not cutie.prompt_yes_or_no(message, yes_text="Restart", no_text="Skip marking step"):
#                 continue

#         tensorboard_log_directory = os.path.join("runs", "table1_imagenette", f"{marking_percentage}_percent", "marking")
#         shutil.rmtree(experiment_directory, ignore_errors=True)
#         shutil.rmtree(tensorboard_log_directory, ignore_errors=True)
#         do_marking_run_multiclass(marking_percentage, experiment_directory, tensorboard_log_directory,
#                                   marking_network, training_set)

#     logger.info("")
#     logger.info("Step 3 - Training Target Networks")
#     logger.info("---------------------------------")
#     for marking_percentage in marking_percentages:
#         marked_images_directory = os.path.join("experiments", "table1_imagenette", f"{marking_percentage}_percent", "marked_images")
#         output_directory = os.path.join("experiments", "table2_imagenette", f"{marking_percentage}_percent", "marked_classifier")
#         tensorboard_log_directory = os.path.join("runs", "table2_imagenette", f"{marking_percentage}_percent", "target")

#         # Train resnet18 from scratch
#         model = torchvision.models.resnet18(pretrained=False, num_classes=10)
#         optimizer = lambda model : torch.optim.AdamW(model.parameters())

#         epochs = 60
#         dataloader_func = partial(train_marked_classifier.get_data_loaders_imagenette, 
#                                   train_images_path, test_images_path, marked_images_directory, batch_size=step_3_batch_size)
#         train_marked_classifier.main(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory, 
#                                      epochs=epochs)

#     logger.info("")
#     logger.info("Step 4 - Calculating p-values")
#     logger.info("-----------------------------")
#     p_values = calculate_p_values(marking_percentages, checkpoint_path, 2, True)  
#     torch.save(p_values, p_values_file)
#     p_values = torch.load(p_values_file)

#     logger.info("")
#     logger.info("Step 5 - Generating Table 2")
#     logger.info("---------------------------")
#     generate_table_2(marking_percentages, p_values, checkpoint_path)

def main(gpu_count, imagenet_path, step_3_batch_size):
    setup_logger_tqdm() # Commence logging to console
    table_1_work(gpu_count, imagenet_path, step_3_batch_size)
    # table_2_work(imagenet_path, step_3_batch_size)

parser_description = 'Perform experiments and generate Table 1 and 2 for imagenet.'
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument("-dir", "--imagenet_path", default="E:/imagenette2/")
parser.add_argument("-bs", "--batch_size_step_3", type=int, default=16)
parser.add_argument("-gpus", "--gpu_count", type=int, default=1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.gpu_count, args.imagenet_path, args.batch_size_step_3)