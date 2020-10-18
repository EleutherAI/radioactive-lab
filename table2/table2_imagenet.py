"""
Table 2
-------
1. Download resnet18 pretrained on Imagenet ILSVRC 2012.
2. Re-use the marked images generated for Table 1
3. Train a resnet18 from scratching with marked images merged with vanilla images.
4. Perform the radioactive detection p-tests on the network trained in step 3. Compare the top-1 accuracy
   of this network with the network downloaded in step 1.
5. Generate Table 2.

NOTES: Step 2 will reuse the data from Table 1 if available.
       We use differentiable center crop for marking.
       We use random crop for training the target network.
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
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cutie

from table1.table1_imagenet import do_marking_run_multiclass
from radioactive.make_data_radioactive import main as do_marking
import radioactive.train_marked_classifier_dist as train_marked_classifier_dist
from radioactive.detect_radioactivity import main as detect_radioactivity
import radioactive.differentiable_augmentations as differentiable_augmentations
from utils.utils import NORMALIZE_IMAGENET

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

def calculate_p_values(marking_percentages, batch_size, test_set_loader):
    logfile_path = f"experiments/table2_imagenet/detect_radioactivity.log"
    setup_logger_tqdm(logfile_path)

    p_values = []

    # Load Marking Network and remove fully connected layer
    marking_network = torchvision.models.resnet18(pretrained=True)
    marking_network.fc = nn.Sequential()

    for run in marking_percentages:
        run_name = f"{run}_percent"
        carrier_path = f"experiments/table1_imagenet/{run_name}/carriers.pth" # Reuse table 1 carrier

        target_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
        target_checkpoint_path = f"experiments/table2_imagenet/{run_name}/marked_classifier/rank_0/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        target_checkpoint['model_state_dict'] = {k.replace("module.", ""): v for k, v in target_checkpoint['model_state_dict'].items()}

        target_network.load_state_dict(target_checkpoint['model_state_dict'])
        target_network.fc = nn.Sequential()
  
        (scores, p_vals, combined_pval) = detect_radioactivity(carrier_path, marking_network, 
                                                               target_network, target_checkpoint,
                                                               batch_size=batch_size,
                                                               align=True, test_set_loader=test_set_loader)
        p_values.append(combined_pval)

    return p_values

def generate_table_2(marking_percentages, p_values, vanilla_accuracy):
    # The Rest
    accuracies = [vanilla_accuracy]
    for run in marking_percentages:
        run_name = f"{run}_percent"
        target_checkpoint_path = f"experiments/table2_imagenet/{run_name}/marked_classifier/rank_0/checkpoint.pth"
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
    plt.savefig("experiments/table2_imagenet/table2.png")
    plt.show()

def main(imagenet_path, step_3_batch_size, mp_args):
    logger.info("")
    logger.info("Table 2 Preparation Commencing")
    logger.info("=============================")
    marking_percentages = [1, 2, 5, 10]

    train_images_path = os.path.join(imagenet_path, "train")
    test_images_path = os.path.join(imagenet_path, "val")
    p_values_file = "experiments/table2_imagenet/p_values.pth"

    logger.info("")
    logger.info("Step 1 - Download Marking Network")
    logger.info("------------------------------")
    marking_network = torchvision.models.resnet18(pretrained=True)

    logger.info("")
    logger.info("Step 2 - Image Marking")
    logger.info("----------------------")
    # Parallelized separately
    # Reuses marked images from Table 1 if available
    training_set = torchvision.datasets.ImageFolder(train_images_path)
    for marking_percentage in marking_percentages:
        experiment_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent")
        if os.path.exists(os.path.join(experiment_directory, "marking.complete")):
            message = f"Marking step already completed for {marking_percentage}%. Do you want to restart this part of " \
                      "the experiment?"
            #if not cutie.prompt_yes_or_no(message, yes_text="Restart", no_text="Skip marking step"):
            #    continue

            logger.info("SKIPPING MARKING - Fix this up before sending to DGX")
            continue

        tensorboard_log_directory = os.path.join("runs", "table1_imagenet", f"{marking_percentage}_percent", "marking")
        shutil.rmtree(experiment_directory, ignore_errors=True)
        shutil.rmtree(tensorboard_log_directory, ignore_errors=True)
        do_marking_run_multiclass(marking_percentage, experiment_directory, tensorboard_log_directory,
                                  marking_network, training_set)

    logger.info("")
    logger.info("Step 3 - Training Target Networks")
    logger.info("---------------------------------")
    # Parallelized with DDP
    for marking_percentage in marking_percentages:
        marked_images_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent", "marked_images")
        output_directory = os.path.join("experiments", "table2_imagenet", f"{marking_percentage}_percent", "marked_classifier")
        tensorboard_log_directory = os.path.join("runs", "table2_imagenet", f"{marking_percentage}_percent", "target")

        # Train resnet18 from scratch
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        optimizer = lambda model : torch.optim.AdamW(model.parameters())
        optimizer = train_marked_classifier_dist.adamw_full

        epochs = 60
        dataloader_func = partial(train_marked_classifier_dist.get_data_loaders_imagenet, 
                                  train_images_path, test_images_path, marked_images_directory, step_3_batch_size, 1)
        train_args = (mp_args, dataloader_func, model, optimizer, output_directory, tensorboard_log_directory, epochs)
        mp.spawn(train_marked_classifier_dist.main, nprocs=mp_args.gpus, args=train_args)

    logger.info("")
    logger.info("Step 4 - Calculating p-values")
    logger.info("-----------------------------")
    test_set_loader = train_marked_classifier_dist.get_imagenet_test_loader(test_images_path, NORMALIZE_IMAGENET, 
                                                                            batch_size=step_3_batch_size)
    p_values = calculate_p_values(marking_percentages, step_3_batch_size, test_set_loader)  
    torch.save(p_values, p_values_file)
    p_values = torch.load(p_values_file)

    logger.info("")
    logger.info("Step 5 - Generating Table 2")
    logger.info("---------------------------")
    # Get Vanilla Accuracy
    marking_network = torchvision.models.resnet18(pretrained=True)
    marking_network.to("cuda")
    vanilla_accuracy = train_marked_classifier_dist.test_model("cuda", marking_network, test_set_loader)

    # Finish Up
    generate_table_2(marking_percentages, p_values, vanilla_accuracy)

parser_description = 'Perform experiments and generate Table 1 and 2 for imagenet.'
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument("-dir", "--imagenet_path", default="E:/imagenet2/")
parser.add_argument("-bs", "--batch_size_step_3", type=int, default=16)
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')

class AnonObject(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    setup_logger_tqdm() # Commence logging to console

    assert(torch.cuda.is_available())
    assert(torch.distributed.is_available())
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    mp_args = AnonObject()
    mp_args.nr = args.nr
    mp_args.gpus = args.gpus
    mp_args.world_size = args.gpus * args.nodes

    main(args.imagenet_path, args.batch_size_step_3, mp_args)

