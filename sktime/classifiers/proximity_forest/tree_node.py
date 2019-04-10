import numpy as np

class TreeNode:
    'Node of a proximity tree'

    def __init__(self, instances, class_labels, num_evaluations):
        self.instances = instances
        self.class_labels = class_labels
        self.num_evaluations = num_evaluations  # AKA R (the number of splits to evaluate before choosing best)

    def is_pure(self):
        return np.unique(self.class_labels)


