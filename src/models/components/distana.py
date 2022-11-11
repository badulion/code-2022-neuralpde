import torch
import torch.nn as nn

from src.models.components.convLSTM import ConvLSTMCell




class Distana(nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, 
                 input_dim,
                 pre_layer_size,
                 lat_size,
                 lstm_hid_size,
                 kernel_size=3,
                 bias=True
                 ):

        super(Distana, self).__init__()
        #
        # PK specific parameters
        
        # Convolution parameters
        self.pk_kernel_size = kernel_size

        # Input sizes (dimensions)
        self.pk_dyn_in_size = input_dim
        
        # Layer sizes (number of neurons per layer)
        self.pk_pre_layer_size = pre_layer_size
        self.pk_lstm_hid_size = lstm_hid_size

        # Output sizes (dimensions)
        self.pk_dyn_out_size = input_dim

        # Lateral vector size
        self.pk_lat_size = lat_size

        # Lateral input convolution layer
        self.lat_in_conv_layer = nn.Conv2d(
            in_channels=self.pk_lat_size,
            out_channels=self.pk_lat_size,
            kernel_size=self.pk_kernel_size,
            padding='same',
            padding_mode='circular',
            bias=bias
        )

        # Dynamic and lateral input preprocessing layer
        self.pre_layer = nn.Conv2d(
            in_channels=self.pk_dyn_in_size + self.pk_lat_size,
            out_channels=self.pk_pre_layer_size,
            kernel_size=3,
            padding='same',
            padding_mode='circular',
            bias=bias
        )

        # Central LSTM layer
        self.clstm = ConvLSTMCell(
            input_dim=self.pk_pre_layer_size,
            hidden_dim=self.pk_lstm_hid_size,
            kernel_size=self.pk_kernel_size,
            bias=bias
        )

        # Postprocessing layer
        self.post_layer = nn.Conv2d(
            in_channels=self.pk_lstm_hid_size,
            out_channels=self.pk_dyn_out_size + self.pk_lat_size,
            kernel_size=3,
            padding='same',
            padding_mode='circular',
            bias=bias
        )

    def forward(self, dyn_in, lat_out_prev):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param lat_out_prev: Output of the previous lateral layer
        """

        # Compute the lateral input as convolution of the lateral outputs from
        # the previous timestep
        pk_lat_in = self.lat_in_conv_layer(lat_out_prev)

        # Forward the dynamic and lateral inputs through the preprocessing
        # layer
        pk_dynlat_in = torch.cat(tensors=(dyn_in, pk_lat_in), dim=1)
        pre_act = torch.tanh(self.pre_layer(pk_dynlat_in))

        # Feed the preprocessed data through the lstm
        lstm_h = self.clstm(pre_act)

        # Pass the lstm output through the postprocessing layer
        post_act = self.post_layer(lstm_h)

        # Dynamic output
        dyn_out = post_act[:, :self.pk_dyn_out_size]

        # Lateral output
        lat_out = torch.tanh(post_act[:, -self.pk_lat_size:])

        return dyn_out, lat_out

        
    def reset_hidden(self, batch_size, image_size):
        self.clstm.reset_hidden(batch_size, image_size)