from numpy.random import default_rng
import numpy as np

def initialize_weights(low, high, n, seed=None):
    # Create a random number generator
    rng = default_rng(seed)

    # Generate random starting weights
    weights = rng.uniform(low, high, size=n)

    return weights 

def forward(x, weights, bias=0.0, activation=None):
    # Calculate the weighted sum of the inputs
    output = np.dot(x, weights) + bias

    # Apply activation if one is provided
    if activation is not None:
        output = activation(output)

    # Return the model prediction
    return output

def numerical_gradient(loss_fn, params, h=1e-5):
    #converting parameters to a numpy array
    params = np.asarray(params, dtype=float)
    #creating an array to store the gradient
    
    gradient = np.zeros_like(params)
    
    #calculating the gradient for each parameter
    
    for index in np.ndindex(params.shape):
        original_value = params[index]
        
        #calculating the loss with a slightly larger parameter
        
        params_plus = params.copy()
        params_plus[index] = original_value + h
        loss_plus = loss_fn(params_plus)
        
        #calculating the loss with a slightly smaller parameter
        params_minus = params.copy()
        params_minus[index] = original_value - h
        loss_minus = loss_fn(params_minus)
        
        #approximate the derivate using central difference
        
        gradient[index] = (loss_plus - loss_minus) / (2 * h)
        
    return gradient

def gradient_descent_step(params, gradient, learning_rate):
    
    #update the parameters in the opposite direction of the gradient
    updated_params = params - learning_rate * gradient
    
    return updated_params

def analytical_gradient(dataset, weights, bias=0.0, activation=None):
    #separate inputs and correct outputs
    
    x = np.array([pair[0] for pair in dataset], dtype=float)
    y_true = np.array([[pair[1]] for pair in dataset], dtype=float)
    
    #calculate the model predictions
    
    predictions = forward(
        x.reshape(-1, 1), weights, bias, activation
    ) 

    # Calculate the prediction errors
    errors = predictions - y_true

    # Calculate the analytical weight gradient
    weight_gradient = (2 / len(dataset)) * np.dot(x, errors)

    # Calculate the analytical bias gradient
    bias_gradient = (2 / len(dataset)) * np.sum(errors)

    return np.array([weight_gradient]), bias_gradient


def dataset_loss(dataset, weights, bias=0.0, activation=None, loss_fn=None):
    # Use mean squared error by default
    if loss_fn is None:
        from dl_lib.losses.functional import mse
        loss_fn = mse

    # Separate the inputs and correct outputs
    x = np.asarray([pair[0] for pair in dataset], dtype=float)
    y_true = np.asarray([pair[1] for pair in dataset], dtype=float)

    # Calculate predictions for all inputs
    predictions = forward(x, weights, bias=bias, activation=activation)

    # Calculate the loss for the whole dataset
    return loss_fn(y_true, predictions)


def dataset_loss_given_bias(dataset, weights, activation=None, loss_fn=None, bias=None):
    # Convert the bias array to a single number
    if bias is None:
        bias = np.array([0.0])

    bias_value = float(np.asarray(bias).reshape(-1)[0])

    # Calculate the loss using this bias
    return dataset_loss(
        dataset,
        weights,
        bias=bias_value,
        activation=activation,
        loss_fn=loss_fn,
    )
