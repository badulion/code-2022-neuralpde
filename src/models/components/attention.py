import torch
import torch.nn as nn


class SeqSelfAttention(nn.Module):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 feature_dim,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 **kwargs):
        """Layer initialization.
        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = torch.nn.init.kaiming_uniform_
        self.bias_initializer = torch.nn.init.zeros_
        self.attention_activation = torch.sigmoid

        
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(feature_dim)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(feature_dim)
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': 'kaiming_uniform',
            'bias_initializer': 'zeros',
            'attention_activation': 'sigmoid',
        }
        return list(config.items())

    def _build_additive_attention(self, feature_dim):

        self.Wt = torch.nn.Parameter(torch.empty(size=(feature_dim, self.units)))
        self.kernel_initializer(self.Wt)

        self.Wx = torch.nn.Parameter(torch.empty(size=(feature_dim, self.units)))
        self.kernel_initializer(self.Wx)

        if self.use_additive_bias:
            self.bh = torch.nn.Parameter(torch.empty(size=(self.units,)))
            self.bias_initializer(self.bh)

        self.Wa = torch.nn.Parameter(torch.empty(size=(self.units, 1)))
        self.kernel_initializer(self.Wa)

        if self.use_attention_bias:
            self.ba = torch.nn.Parameter(torch.empty(size=(1,)))
            self.bias_initializer(self.ba)

    def _build_multiplicative_attention(self, feature_dim):
        self.Wa = torch.nn.Parameter(torch.empty(size=(feature_dim, feature_dim)))
        self.kernel_initializer(self.Wa)

        if self.use_attention_bias:
            self.ba = torch.nn.Parameter(torch.empty(size=(1,)))
            self.bias_initializer(self.ba)

    def forward(self, inputs, mask=None, **kwargs):
        input_len = inputs.size(1)
        #K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = torch.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = torch.arange(0, input_len) - self.attention_width // 2
            lower = torch.unsqueeze(lower, dim=-1)
            upper = lower + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, input_len), dim=0)
            e -= 10000.0 * (1.0 - (lower <= indices).float()) * (indices < upper).float()
        if mask is not None:
            mask = torch.unsqueeze(mask.float(), dim=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - torch.permute(mask, (0, 2, 1))))

        # a_{t} = \text{softmax}(e_t)
        max_e, _ = torch.max(e, axis=-1, keepdims=True)
        e = torch.exp(e - max_e)
        a = e / torch.sum(e, dim=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = SeqSelfAttention._bdot(a, inputs)

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = inputs.size()
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = torch.unsqueeze(torch.matmul(inputs, self.Wt), 2)
        k = torch.unsqueeze(torch.matmul(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.bh)
        else:
            h = torch.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = torch.reshape(torch.matmul(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = torch.reshape(torch.matmul(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = SeqSelfAttention._bdot(torch.matmul(inputs, self.Wa), torch.permute(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _bdot(x, y, axes=None):
        if isinstance(axes, int):
            axes = (axes, axes)
        if x.ndim > y.ndim:
            diff = x.ndim - y.ndim
            y = torch.reshape(y, torch.concat([y.shape, [1] * (diff)], dim=0))
        elif y.ndim > x.ndim:
            diff = y.ndim - x.ndim
            x = torch.reshape(x, torch.concat([x.shape, [1] * (diff)], dim=0))
        else:
            diff = 0

        if x.ndim == 2 and y.ndim == 2:
            if axes[0] == axes[1]:
                out = torch.sum(torch.matmul(x, y), dim=axes[0])
            else:
                out = torch.sum(torch.matmul(torch.transpose(x, 1, 0), y), dim=axes[1])
        else:
            if axes is not None:
                adj_x = None if axes[0] == x.ndim - 1 else True
                adj_y = True if axes[1] == y.ndim - 1 else None
                out = torch.matmul(x, y)
            else:
                out = torch.matmul(x, y)
        if diff:
            if x.ndim > y.ndim:
                idx = x.ndim + y.ndim - 3
            else:
                idx = x.ndim - 1
            out = torch.squeeze(out, list(range(idx, idx + diff)))
        if out.ndim == 1:
            out = torch.unsqeeze(out, 1)
        return out

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


if __name__ == '__main__':
    seq_attention = SeqSelfAttention(10)
    x = torch.randn((4,16,10))
    print(seq_attention(x).shape)


