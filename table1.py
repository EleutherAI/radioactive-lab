import torch
import torchvision
import os
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.transforms as transforms
import torch.nn as nn
import numpy as np

import resnet18_on_cifar10 as resnet18cifar10
from make_data_radioactive import get_images_for_marking, get_images_for_marking_multiclass
from make_data_radioactive import main as do_marking
from train_marked_classifier import main as train_marked_classifier
from detect_radioactivity import main as detect_radioactivity

import matplotlib.pyplot as plt

import logging
from logger import setup_logger
logger = logging.getLogger(__name__)

"""
1. Train a resnet18 classifier on cifar10
2. Generate 1, 2, 5, 10% Markings using network from step 1 as marking network.
3. Create a logistic regression network on the end of the pretrained resnet from step 1.
   Train it using the radioactive data from step 2.
4. Perform the radioactive detection p-tests on the network trained in step 3. Compare the top-1 accuracy
   of this network with the network trained in step 1.
5. Generate Table 1. "Center Crop" augmentation makes no sense when using CIFAR10 data so this is skipped.
"""

def do_marking_run(class_marking_percentage, run_name, augment=True, overwrite=False):

    # Setup experiment directory
    experiment_directory = os.path.join("experiments/table1", run_name)    
    if os.path.isdir(experiment_directory):
        if not overwrite:
            error_message = "Overwrite set to False. By default we assume you don't want to repeat the marking stage." \
                            " Modify the main to either remove the marking stage or set overwrite=True when calling this function." \
                            " See the note in main for further information."
            raise Exception(error_message)
        shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger(filepath=logfile_path)

    # Prepare for TensorBoard
    tensorboard_log_directory_base = f"runs/table1_{run_name}"
    our_tensorboard_logs = glob.glob(f"{tensorboard_log_directory_base}*")
    for tensorboard_log in our_tensorboard_logs:
        shutil.rmtree(tensorboard_log)

    # Load randomly sampled images from random class along with list of original indexes 
    training_set = torchvision.datasets.CIFAR10(root="experiments/datasets", download=True)
    class_id, images, original_indexes = get_images_for_marking(training_set, 
        class_marking_percentage=class_marking_percentage,
        tensorboard_log_directory=tensorboard_log_directory_base)

    # Marking network is the resnet18 we trained on CIFAR10
    marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)    
    checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    marking_network_checkpoint = torch.load(checkpoint_path)
    marking_network.load_state_dict(marking_network_checkpoint["model_state_dict"])

    # Carriers
    marking_network_fc_feature_size = 512
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))

    # Run!
    optimizer = lambda x : torch.optim.Adam(x, lr=0.1)
    epochs = 100
    batch_size = 32
    output_directory = os.path.join(experiment_directory, "marked_images")
    marked_images = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                               class_id, optimizer, tensorboard_log_directory_base, epochs=epochs, 
                               batch_size=batch_size, overwrite=True, augment=augment)

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory_base)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

def do_marking_run_multiclass(overall_marking_percentage, run_name, augment=True, overwrite=False):
    # Setup experiment directory
    experiment_directory = os.path.join("experiments/table1", run_name)    
    if os.path.isdir(experiment_directory):
        if not overwrite:
            error_message = "Overwrite set to False. By default we assume you don't want to repeat the marking stage." \
                            " Modify the main to either remove the marking stage or set overwrite=True when calling this function." \
                            " See the note in main for further information."
            raise Exception(error_message)
        shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger(filepath=logfile_path)

    # Prepare for TensorBoard
    tensorboard_log_directory_base = f"runs/table1_{run_name}"
    our_tensorboard_logs = glob.glob(f"{tensorboard_log_directory_base}*")
    for tensorboard_log in our_tensorboard_logs:
        shutil.rmtree(tensorboard_log)

    training_set = torchvision.datasets.CIFAR10(root="experiments/datasets", download=True)

    # Marking network is the resnet18 we trained on CIFAR10
    marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)    
    checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    marking_network_checkpoint = torch.load(checkpoint_path)
    marking_network.load_state_dict(marking_network_checkpoint["model_state_dict"])

    # Carriers
    marking_network_fc_feature_size = 512
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))


    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    image_data = get_images_for_marking_multiclass(training_set,
                                                    tensorboard_log_directory_base,
                                                    overall_marking_percentage)

    marked_images = []
    for class_id, image_list in image_data.items():
        images, original_indexes = map(list, zip(*image_list))
        optimizer = lambda x : torch.optim.Adam(x, lr=0.1)
        epochs = 100
        batch_size = 32
        output_directory = os.path.join(experiment_directory, "marked_images")
        marked_images_temp = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                                        class_id, optimizer, tensorboard_log_directory_base, epochs=epochs, 
                                        batch_size=batch_size, overwrite=False, augment=augment)
            
        marked_images =  marked_images + marked_images_temp   

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory_base)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

def do_training_run(run_name, augment=True):
    # Load our trained resnet18 from step1
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Retrain the fully connected layer only
    for param in model.parameters():
        param.requires_grad = False
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = lambda model : torch.optim.AdamW(model.fc.parameters())

    tensorboard_log_directory = f"runs/table1_{run_name}_target"
    epochs = 20
    output_directory = f"experiments/table1/{run_name}/marked_classifier"
    marked_images_directory = f"experiments/table1/{run_name}/marked_images"
    train_marked_classifier(marked_images_directory, optimizer, output_directory, tensorboard_log_directory, 
                            custom_model=model, epochs=epochs, augment=augment)

def step1():
    optimizer = lambda x : torch.optim.AdamW(x)
    output_directory_root = "experiments/table1"
    experiment_name = "step1"
    epochs = 60
    resnet18cifar10.main(experiment_name, optimizer, 
                         output_directory_root=output_directory_root,
                         epochs=epochs)

def step2(marking_percentages): 
    for marking_percentage in marking_percentages:
        do_marking_run_multiclass(marking_percentage, f"{marking_percentage}_percent", augment=False)

def step3(marking_percentages): 
    """
    You will need to blow away the TensorBoard logs and checkpoint files if you want
    to train from scratch a second time.
    """

    for marking_percentage in marking_percentages:
        do_training_run(f"{marking_percentage}_percent", augment=False)

def step4(marking_percentages):
    logfile_path = f"experiments/table1/detect_radioactivity.log"
    setup_logger(logfile_path)

    p_values = []

    # Load Marking Network and remove fully connected layer
    marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
    marking_checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    marking_checkpoint = torch.load(marking_checkpoint_path)
    marking_network.load_state_dict(marking_checkpoint["model_state_dict"])
    marking_network.fc = nn.Sequential()

    ## Perform detection on unmarked network as sanity test
    #random_carrier_path = "experiments/table1/1_percent/carriers.pth"
    #(scores, p_vals, combined_pval) = detect_radioactivity(random_carrier_path, marking_network, 
    #                                                       marking_network, marking_checkpoint, 
    #                                                       align=False)
    #p_values.append(combined_pval)

    # The Rest
    for run in marking_percentages:
        run_name = f"{run}_percent"
        carrier_path = f"experiments/table1/{run_name}/carriers.pth"

        target_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
        target_checkpoint_path = f"experiments/table1/{run_name}/marked_classifier/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        target_network.load_state_dict(target_checkpoint["model_state_dict"])
        target_network.fc = nn.Sequential()

        # No need to align when only retraining the logistic regression
        (scores, p_vals, combined_pval) = detect_radioactivity(carrier_path, marking_network, 
                                                               target_network, target_checkpoint,
                                                               align=False)
        p_values.append(combined_pval)

    return p_values

def step5(marking_percentages, p_values):
    # Get Vanilla Accuracy
    vanilla_checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    vanilla_checkpoint = torch.load(vanilla_checkpoint_path)

    # The Rest
    accuracies = [vanilla_checkpoint["test_accuracy"]]
    for run in marking_percentages:
        run_name = f"{run}_percent"
        marked_checkpoint_path = f"experiments/table1/{run_name}/marked_classifier/checkpoint.pth"
        marked_checkpoint = torch.load(marked_checkpoint_path)
        accuracies.append(marked_checkpoint["test_accuracy"])

    # Create the table!
    column_labels = tuple([0] + marking_percentages)
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(column_labels)))
    row_labels = ["log10(p)", "Top-1 %"]
    formatted_pvalues = ["n/a"]
    formatted_pvalues += [f"{p:0.4f}" for p in np.log10(p_values)]

    cell_text = [formatted_pvalues, accuracies]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colColours=colors,
                     colLabels=column_labels,
                     loc='center')
    plt.savefig("experiments/table1/table1.png")
    plt.show()


if __name__ == '__main__':
    marking_percentages = [1, 2, 5, 10, 20]
    p_values_file = "experiments/table1/p_values.pth"

    step1() # Train Marking Network

    # Marking
    # If you had to stop mid marking check your experiments/table1 directory and delete the largest
    # x_percent before setting the marking_percentage list with the percentages you haven't marked.
    # Make sure you set it back to the original list before resuming the rest of the steps.
    # 
    # If you have already completed the marking stage, comment it out.
    # If you want to generate new marking data then the below steps will need to be repeated, delete
    # everything in the experiments/table1 directory except step1 and start again.
    #step2(marking_percentages)

    step3(marking_percentages) # Training Target Networks
    p_values = step4(marking_percentages)  # Calculate p-values
    torch.save(p_values, p_values_file)
    p_values = torch.load(p_values_file)
    step5(marking_percentages, p_values) # Generate Table 1
