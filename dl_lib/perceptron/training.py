
import numpy as np

from dl_lib.perceptron.functional import (
    numerical_gradient,
    gradient_descent_step,
    dataset_loss,
    dataset_loss_given_bias,
)


def train(model, dataset, learning_rate, epochs, record_every=10000):
    # Store the training progress
    history = []

    for epoch in range(1, epochs + 1):

        # Calculate the loss using the current weights
        def loss_fn(weights):
            return dataset_loss(
                dataset,
                weights,
                bias=model.bias if model.use_bias else 0.0,
                activation=model.activation,
            )

        # Calculate and update the weights
        gradient = numerical_gradient(loss_fn, model.weights)
        model.weights = gradient_descent_step(
            model.weights, gradient, learning_rate
        )

        # Update the bias only when it is enabled
        if model.use_bias:
            def bias_loss_fn(bias):
                return dataset_loss_given_bias(
                    dataset,
                    model.weights,
                    activation=model.activation,
                    bias=bias,
                )

            bias_gradient = numerical_gradient(
                bias_loss_fn, np.array([model.bias])
            )

            updated_bias = gradient_descent_step(
                np.array([model.bias]),
                bias_gradient,
                learning_rate,
            )

            model.bias = float(updated_bias[0])

        # Record the loss at regular intervals and at the final epoch
        if epoch % record_every == 0 or epoch == epochs:
            current_loss = dataset_loss(
                dataset,
                model.weights,
                bias=model.bias if model.use_bias else 0.0,
                activation=model.activation,
            )

            history.append((epoch, current_loss))

    return history
