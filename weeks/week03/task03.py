import numpy as np
from dl_lib.losses.functional import mse

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    initialize_weights,
    forward,
    numerical_gradient,
    gradient_descent_step,
)

def main():
    # Create the dataset
    dataset = create_dataset(6)

    # Separate the inputs and correct outputs
    x = np.array([pair[0] for pair in dataset]).reshape(-1, 1)
    y_true = np.array([pair[1] for pair in dataset])

    # Initialize the starting weight
    weights = initialize_weights(0, 10, 1, seed=42)

    # Calculate the loss for a given weight
    def loss_fn(params):
        predictions = forward(x, params)
        return mse(y_true, predictions)

    # Calculate the initial loss and gradient
    initial_loss = loss_fn(weights)
    gradient = numerical_gradient(loss_fn, weights)

    print(f"Loss before update: {initial_loss:.6f}")
    print(f"Numerical gradient: {gradient[0]:.6f}")

    # Try updating the weight without a learning rate
    raw_weights = weights - gradient
    print(
        f"w after subtracting the raw gradient: "
        f"w={raw_weights[0]:.6f}, loss={loss_fn(raw_weights):.6f}"
    )

    # Apply one step with a learning rate
    learning_rate = 0.001
    updated_weights = gradient_descent_step(
        weights, gradient, learning_rate
    )

    print(
        f"w after one step with learning_rate=0.001: "
        f"w={updated_weights[0]:.6f}, "
        f"loss={loss_fn(updated_weights):.6f}"
    )

    # Train the model for 10 epochs
    print(f"\nLoss before update: {initial_loss:.6f}")

    for epoch in range(1, 11):
        gradient = numerical_gradient(loss_fn, weights)
        weights = gradient_descent_step(
            weights, gradient, learning_rate
        )

        print(
            f"epoch {epoch}: "
            f"w={weights[0]:.6f}, "
            f"loss={loss_fn(weights):.6f}"
        )

if __name__ == "__main__": 
    main()