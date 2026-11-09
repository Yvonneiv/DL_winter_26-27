
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main():
    # Create the first four observations
    dataset = create_dataset(4)

    # Generate two random weights
    weight1 = initialize_weights(0, 100, 1, seed=42)
    weight2 = initialize_weights(0, 10, 1, seed=42)

    # Print the results
    print("Dataset:", dataset)
    print("initialize_weights(0, 100, 1):", weight1)
    print("initialize_weights(0, 10, 1):", weight2)


if __name__ == "__main__":
    main()
