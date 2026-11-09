def create_dataset(n):
    # Create an empty list for the training data
    dataset = []

    # Generate input-output pairs
    for x in range(n):
        y = 2 * x

        # Store each pair as floating point numbers
        dataset.append((float(x), float(y)))

    return dataset

import numpy as np


def and_dataset():
    # Create all possible combinations of two binary inputs
    return [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0]), 0.0),
        (np.array([1.0, 1.0]), 1.0),
    ]

def or_dataset():
        #create all possible combinations of two binary inputs
        return [
               (np.array([0.0, 0.0]), 0.0),
               (np.array([0.0, 1.0]), 1.0),
               (np.array([1.0, 0.0]), 1.0),
               (np.array([1.0, 1.0]), 1.0),
        ]
        
    
        