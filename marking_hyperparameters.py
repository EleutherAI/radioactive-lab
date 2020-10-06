import torch
import torchvision
import os
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.transforms as transforms
import random

from make_data_radioactive import get_images_for_marking_multiclass
from make_data_radioactive import main as do_marking

import logging
from logger import setup_logger
logger = logging.getLogger(__name__)

def do_marking_run_multiclass(overall_marking_percentage, experiment_name, 
                              learning_rate=0.1, epochs=100, augment=True, overwrite=False):
    # Setup experiment directory
    experiment_directory = os.path.join("experiments/marking_hyperparams", experiment_name)
    if os.path.isdir(experiment_directory):
        if not overwrite:
            error_message = "Overwrite set to False. By default we assume you don't want to repeat marking."
            raise Exception(error_message)
        shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger(filepath=logfile_path)

    # Prepare for TensorBoard
    tensorboard_log_directory_base = f"runs/{experiment_name}"
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
    random.seed(10)
    image_data = get_images_for_marking_multiclass(training_set,
                                                    tensorboard_log_directory_base,
                                                    overall_marking_percentage)

    marked_images = []
    for class_id, image_list in image_data.items():
        if image_list:
            images, original_indexes = map(list, zip(*image_list))
            optimizer = lambda x : torch.optim.Adam(x, lr=learning_rate)        
            batch_size = 32
            output_directory = os.path.join(experiment_directory, "marked_images")
            tensorboard_log_directory = f"{tensorboard_log_directory_base}_class{class_id}"
            marked_images_temp = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                                            class_id, optimizer, tensorboard_log_directory, epochs=epochs, 
                                            batch_size=batch_size, overwrite=False, augment=augment)
            
            marked_images =  marked_images + marked_images_temp   

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory_base)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    marking_percentage = 0.1
    experiment_name = "lr1"
    do_marking_run_multiclass(marking_percentage, experiment_name, learning_rate=1, epochs=200, augment=False, overwrite=True)

    torch.manual_seed(0)
    experiment_name = "lr0.1"
    do_marking_run_multiclass(marking_percentage, experiment_name, learning_rate=0.1, epochs=200, augment=False, overwrite=True)

    torch.manual_seed(0)
    experiment_name = "lr0.01"
    do_marking_run_multiclass(marking_percentage, experiment_name, learning_rate=0.01, epochs=300, augment=False, overwrite=True)

    torch.manual_seed(0)
    experiment_name = "lr0.001"
    do_marking_run_multiclass(marking_percentage, experiment_name, learning_rate=0.001, epochs=400, augment=False, overwrite=True)

    torch.manual_seed(0)
    experiment_name = "lr0.0001"
    do_marking_run_multiclass(marking_percentage, experiment_name, learning_rate=0.0001, epochs=200, augment=False, overwrite=True)