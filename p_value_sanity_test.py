import torch
from torch import nn as nn
import torchvision
import os
import matplotlib.pyplot as plt
import pickle
from scipy.stats import combine_pvalues
from scipy.special import betainc
import tqdm
import numpy as np

from detect_radioactivity import main as detect_radioactivity
from detect_radioactivity import cosine_pvalue

import logging
from logger import setup_logger
logger = logging.getLogger(__name__)

data_file = "experiments/sanity_test/data.pth"

def sample_and_detect():
    experiment_directory = "experiments/sanity_test"
    os.makedirs(experiment_directory, exist_ok=True) 

    logfile_path = "experiments/sanity_test/detect_radioactivity.log"
    setup_logger(logfile_path)

    p_values = []

    # Load Marking Network and remove fully connected layer
    marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
    marking_checkpoint_path = "experiments/table1/step1/checkpoint.pth"
    marking_checkpoint = torch.load(marking_checkpoint_path)
    marking_network.load_state_dict(marking_checkpoint["model_state_dict"])
    marking_network.fc = nn.Sequential()

    # Perform detection on unmarked network as sanity test
    datas = []
    for i in tqdm.tqdm(range(0, 100)):
        carriers = torch.randn(10, 512)
        carriers /= torch.norm(carriers, dim=1, keepdim=True)
        random_carrier_path = "experiments/sanity_test/carriers.pth"
        torch.save(carriers, random_carrier_path)
        
        data = detect_radioactivity(random_carrier_path, marking_network, marking_network, 
                                    marking_checkpoint, align=False)
        datas.append(data)

    try:
        torch.save(datas, data_file)
    except:
        logger.warning("Torch save failed")
        try:
            pickle.dump(datas, open(data_file, "wb"))
        except:
            logger.warning("Pickle save failed")

    return datas

def verify_carriers():
    carriers = []
    for i in range(0, 1000):
        carrier = torch.randn(10, 512)
        carrier /= torch.norm(carrier, dim=1, keepdim=True)
        carriers.append(carrier)


    for carrierss in carriers:
        for carrier in carrierss:
            print(torch.norm(carrier))    

    # Plot Carriers
    #plt.subplot(111)
    #plt.hist(torch.flatten(carriers), bins=50)
    #plt.gca().set(title='Carriers Histogram', ylabel='Frequency');
    #plt.show()




def plot_histogram(scores, p_vals, combined_pvals, class_id = None):

    # Flatten
    scores_flat = []
    p_vals_flat = []
    if not class_id:
        for sub in scores:
            for score in sub:
                scores_flat.append(score)

        for sub in p_vals:
            for p_val in sub:
                p_vals_flat.append(p_val)
    else:
        scores_flat = [score[class_id] for score in scores]
        p_vals_flat = [p_val[class_id] for p_val in p_vals]

    #plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.figure(1)

    # Plot Scores
    plt.subplot(511)
    plt.hist(scores_flat, bins=100)
    plt.gca().set(title='Scores Histogram', ylabel='Frequency');

    # Plot P Values
    plt.subplot(512)
    plt.hist(p_vals_flat, bins=100)
    plt.gca().set(title='P Values Histogram', ylabel='Frequency');

    plt.subplot(513)
    plt.hist(p_vals_flat, bins=100, cumulative=True, density=True)
    plt.gca().set(title='P Values CDF', ylabel='Cumulative P');

    if not class_id:
        # Plot Combined Values
        plt.subplot(514)
        plt.hist(combined_pvals, bins=100)
        plt.gca().set(title='Combined P Values Histogram', ylabel='Frequency');

        plt.subplot(515)
        plt.hist(combined_pvals, bins=100, cumulative=True, density=True)
        plt.gca().set(title='Combined P Values CDF', ylabel='Cumulative P');

    plt.tight_layout()
    plt.show()

def cosine_pvalue_original(c, d):
    """
    Given a dimension d, returns the probability that the dot product between
    random unitary vectors is higher than c
    """
    assert type(c) in [float, np.float64, np.float32]

    a = (d - 1) / 2.
    b = 1 / 2.

    if c >= 0:
        return 0.5 * betainc(a, b, 1-c**2)
    else:
        return 1 - cosine_pvalue(-c, d=d)

def cosine_pvalue_swap_a_b(c, d):
    """
    Given a dimension d, returns the probability that the dot product between
    random unitary vectors is higher than c
    """
    assert type(c) in [float, np.float64, np.float32]

    b = (d - 1) / 2.
    a = 1 / 2.

    if c >= 0:
        return 0.5 * betainc(a, b, 1-c**2)
    else:
        return 1 - cosine_pvalue(-c, d=d)

def cosine_pvalue_full_switch(c, d):
    """
    Given a dimension d, returns the probability that the dot product between
    random unitary vectors is higher than c
    """
    assert type(c) in [float, np.float64, np.float32]

    #something = 0.5 * betainc(1/2, 4, 0.3)
    #print(something)

    #a = (d - 1) / 2.
    #b = 1 / 2.

    b = (d - 1) / 2.
    a = 1 / 2.

    if c >= 0:
        #return 0.5 * betainc(a, b, 1-c**2)
        return 1 - betainc(a, b, c)
    else:
        return 1 - cosine_pvalue(-c, d=d)

def rerun_pvalues(scores, cosine_pval_function):
    p_vals = []
    combined_pvals = []
    for score in scores:
        single_test_pval = [cosine_pval_function(c, 512) for c in list(score)]
        p_vals.append(single_test_pval)

        single_test_combined_pvals = combine_pvalues(single_test_pval)[1]
        combined_pvals.append(single_test_combined_pvals)

    return (p_vals, combined_pvals)


if __name__ == "__main__":
    datas = sample_and_detect()
    datas = torch.load(data_file)
    scores, p_vals, combined_pvals = zip(*datas)
    plot_histogram(scores, p_vals, combined_pvals)

    ## Swap a and b
    #p_vals, combined_pvals = rerun_pvalues(scores, cosine_pvalue_full_switch)    
    #plot_histogram(scores, p_vals, combined_pvals)

    ## Full changes
    #p_vals, combined_pvals = rerun_pvalues(scores, cosine_pvalue_swap_a_b)        
    #plot_histogram(scores, p_vals, combined_pvals)

    #verify_carriers()

