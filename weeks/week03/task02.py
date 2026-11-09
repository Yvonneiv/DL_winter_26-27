import numpy as np

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, forward
from dl_lib.losses.functional import mse

def main():
    # Create a dataset with 6 observations
    dataset = create_dataset(6)

    # Separate inputs and expected outputs
    x = np.array([pair[0] for pair in dataset])
    y_true = np.array([pair[1] for pair in dataset])

    # Initialize the weight
    weights = initialize_weights(0, 10, 1, seed=42)

    # Calculate the initial predictions and loss
    y_pred = forward(x.reshape(-1, 1), weights)
    initial_loss = mse(y_true, y_pred)

    print("Dataset:", dataset)
    print("w:", weights[0])
    print("MSE:", initial_loss)

    # Test small changes to the weight
    print("\nPerturbing w:")

    for change in [0.002, 0.001, -0.001, -0.002]:
        new_weights = weights + change
        new_predictions = forward(x.reshape(-1, 1), new_weights)
        new_loss = mse(y_true, new_predictions)

        print(f"w + ({change:+.3f}) -> w={new_weights[0]:.6f}, MSE={new_loss:.6f}")


if __name__ == "__main__":
    main()