class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #Embedding Layer
        self.embedding = nn.Embedding(output_size, hidden_size)
        #GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)
        #Linear layer mapping to the output size
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #Embedding the input and applying relu
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        # We feed the embedded vector as well as the context vector passed as argument into the gru
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
