
import numpy as np

from dl_lib.perceptron.functional import initialize_weights, forward


class Perceptron:
    def __init__(
        self,
        n_inputs,
        weight_range=(-1.0, 1.0),
        seed=None,
        use_bias=False,
        activation=None,
    ):
        # Initialize the neuron's weights
        low, high = weight_range
        self.weights = initialize_weights(low, high, n_inputs, seed)

        # Set the optional bias and activation
        self.use_bias = use_bias
        self.bias = 0.0
        self.activation = activation

    def forward(self, x):
        # Convert the input values to a numpy array
        x = np.asarray(x, dtype=float)

        # Use the bias only when it is enabled
        bias = self.bias if self.use_bias else 0.0

        # Calculate the neuron's prediction
        return forward(x, self.weights, bias=bias, activation=self.activation)
