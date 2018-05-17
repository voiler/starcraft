import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ConvLSTMCell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_shape, num_channels, kernel_size, hidden_size):
        super(ConvLSTMCell, self).__init__()

        self.input_shape = input_shape  # H,W
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        # self.batch_size=batch_size
        self.padding = (kernel_size - 1) / 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.num_channels + self.hidden_size, 4 * self.hidden_size, self.kernel_size, 1,
                              self.padding)

    def forward(self, inputs, hidden_state):
        # print(hidden_state)
        hidden, c = hidden_state  # hidden and c are images with several channels
        # print 'hidden ',hidden.size()
        # print 'inputs ',inputs.size()
        combined = torch.cat((inputs, hidden), 1)  # oncatenate in the channels
        # print 'combined',combined.size()
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.hidden_size, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(batch_size, self.hidden_size, self.input_shape[0], self.input_shape[1]).cuda(),
                    torch.zeros(batch_size, self.hidden_size, self.input_shape[0], self.input_shape[1]).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_size, self.input_shape[0], self.input_shape[1]),
                    torch.zeros(batch_size, self.hidden_size, self.input_shape[0], self.input_shape[1]))


class ConvLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      input_shape: int tuple thats the height and width of the hidden states h and c()
      kernel_size: int that is the height and width of the filters
      hidden_size: int thats the num of channels of the states

    """

    def __init__(self, input_shape, num_channels,
                 kernel_size, hidden_size, num_layers,
                 batch_first=False, weights_initializator=weights_init):
        super(ConvLSTM, self).__init__()

        self.input_shape = input_shape  # H,W
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        cell_list = [ConvLSTMCell(self.input_shape, self.num_channels, self.kernel_size, self.hidden_size)]
        # one has a different number of inputs channels

        for id_cell in range(1, self.num_layers):
            cell_list.append(ConvLSTMCell(self.input_shape, self.hidden_size, self.kernel_size, self.hidden_size))
        self.cell_list = nn.ModuleList(cell_list)
        self.apply(weights_initializator)

    def forward(self, inputs, hidden_state=None):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            inputs is the tensor of shape batch,seq_len,channels,H,W
        """

        current_input = inputs
        if not self.batch_first:
            current_input = inputs.transpose(0, 1)
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=current_input.size(0))
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(1)
        for idlayer in range(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            # all_output = []
            output_inner = []
            for t in range(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[:, t, :, :, :],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.stack(output_inner, dim=1)  # seq_len,B,chans,H,W
        if not self.batch_first:
            current_input = current_input.transpose(0, 1)
        return next_hidden, current_input

    def _init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
