
from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def run_experiment(name, dataset):
    # Create a neuron with two inputs and reproducible weights
    model = Perceptron(
        n_inputs=2,
        weight_range=(-1.0, 1.0),
        seed=42,
        use_bias=False,
        activation=None,
    )

    print(f"\n{name} gate - initial weights: {model.weights}")

    # Train the neuron and record the loss every 10000 epochs
    history = train(
        model,
        dataset,
        learning_rate=0.001,
        epochs=100000,
        record_every=10000,
    )

    # Print the training progress
    for epoch, loss in history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")

    print(f"{name} gate - final weights: {model.weights}")
    print(f"{name} gate - predictions:")

    # Compare predictions with the correct outputs
    for x, y in dataset:
        prediction = model.forward(x)

        a = int(x[0])
        b = int(x[1])

        print(
            f"  {name}({a}, {b}) = {int(y)} "
            f"-> predicted {prediction:.6f}"
        )


def main():
    # Train a neuron for the AND operation
    run_experiment("AND", and_dataset())

    # Train another neuron for the OR operation
    run_experiment("OR", or_dataset())


if __name__ == "__main__":
    main()
