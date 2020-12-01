import logging
import torch.utils.data as data

# Get an unconfigured logger, will propagate to root we configure in main program
logger = logging.getLogger(__name__)

# Wrapper that allows you to merge an augmented subset into a dataset
# Ensure base datasets output same format so we don't need two different transforms
class MergedDataset(data.Dataset):
    def __init__(self, vanilla_dataset, merge_dataset, merge_to_vanilla, transform=None):

        self.vanilla_dataset = vanilla_dataset
        self.merge_dataset = merge_dataset
        self.transform = transform

        # Create reverse index 
        self.vanilla_to_merge = {}
        for i, vanilla_index in enumerate(merge_to_vanilla):
            self.vanilla_to_merge[vanilla_index] = i

        num_marked = len(merge_dataset)
        per_marked = num_marked / len(vanilla_dataset)
        logger.info(f"There are {num_marked} merged examples in this dataset ({100 * per_marked:.2f}%)")

    def __getitem__(self, index):
        if index in self.vanilla_to_merge:
            sample, _ = self.merge_dataset[self.vanilla_to_merge[index]]
            _, target = self.vanilla_dataset[index]
        else:
            sample, target = self.vanilla_dataset[index]

        return self.transform(sample), target


    def __len__(self):
        return len(self.vanilla_dataset)