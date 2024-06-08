import numpy as np

'''
1. Convolutional layers convolves the input_array with filters_array to produce feature_map.
2. This layer has two functions forward_pass and backward_pass
3. In forward_pass, we do matrix multiplication of input_array with each
   filter_array and add bias to each element of resultant array'''


class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters,
                                       filter_size,
                                       filter_size,
                                       input_channels) * 0.1

        self.biases = np.zeros(num_filters)
        self.last_input_array = None
        self.last_output_array = None

    def forward_pass(self, input_array):
        self.last_input_array = input_array
        # Calculating width,height of input_img
        h, w = input_array.shape[1], input_array.shape[2]
        # Calculating output_dim
        out_dim = h - self.filter_size + 1
        # Intialize output array
        self.last_output_array = np.zeros((self.num_filters, out_dim, out_dim))
        # Applying convolutional operation by sliding each filter to
        # input_image array region
        for i in range(self.num_filters):
            filter = self.filters[i]
            for j in range(out_dim):
                for k in range(out_dim):
                    region = input[:,
                                   j:j + self.filter_size,
                                   k:k + self.filter_size]
                    # Adding bias to each feature_map_array element
                    self.last_output_array[i, j, k] = np.sum(
                        region * filter) + self.biases[i]
        return self.last_output_array

    def backward_pass(self, d_l_d_out, learning_rate):
        # Gradients of the loss with respect to the filters.
        d_l_d_filters = np.zeros(self.filters.shape)
        # Gradients of the loss with respect to the biases.
        d_l_d_biases = np.zeros(self.biases.shape)
        # Gradients of the loss with respect to the input.
        d_l_d_input = np.zeros(self.last_input_array.shape)

        out_dim = self.last_output_array.shape[1]

        # Apply the derivative of ReLU to the gradients
        # The
        # ReLU
        # derivative is zero
        # for negative values in the output and one for positive values.This
        # masks the gradient appropriately
        d_l_d_out[self.last_output_array <= 0] = 0

        for i in range(self.num_filters):
            for j in range(out_dim):
                for k in range(out_dim):
                    region = self.last_input_array[:,
                                                   j:j + self.filter_size,
                                                   k:k + self.filter_size]

                    d_l_d_filters[i] += d_l_d_out[i, j, k] * region
                    d_l_d_biases[i] += d_l_d_out[i, j, k]
                    d_l_d_input[:,
                                j:j + self.filter_size,
                                k:k + self.filter_size] += d_l_d_out[i,
                                                                     j,
                                                                     k] * self.filters[i]

        # Update weights and biases
        self.filters -= learning_rate * d_l_d_filters
        self.biases -= learning_rate * d_l_d_biases

        return d_l_d_input
