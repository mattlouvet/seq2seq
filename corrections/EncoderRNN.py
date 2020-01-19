class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #Embedding Layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        #GRU Layer
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #Embedding the input
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

