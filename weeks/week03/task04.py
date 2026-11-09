import numpy as np

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    initialize_weights,
    forward,
    numerical_gradient,
    analytical_gradient
)
from dl_lib.losses.functional import mse

def main():
    
    #creating the dataset
    dataset = create_dataset(6)
    
    #prepare inputs and correct outputs
    x = np.array([pair[0] for pair in dataset]).reshape(-1, 1)
    y_true = np.array([pair[1] for pair in dataset])
    
    #initialize the same weight as in Task 03
    
    weights = initialize_weights(0, 10, 1, seed=42)
    
    #calculate the loss
    
    def loss_fn(params):
        predictions = forward(x, params)
        return mse(y_true, predictions)
    
    loss = loss_fn(weights)
    
    #calculate both gradients
    
    numerical = numerical_gradient(loss_fn, weights)
    analytical, bias_gradient = analytical_gradient(dataset, weights)
    
    #compare the results
    
    difference = np.abs(numerical - analytical)
    
    print(f"Loss: {loss:.6f}")
    print(f"Numerical gradient: {numerical[0]:.6f}")
    print(f"Analytical gradient: {analytical[0, 0]:.6f}")
    print(f"Difference: {difference[0, 0]:.8f}")
    
if __name__ == "__main__":
    main()
    