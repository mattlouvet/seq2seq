class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p #dropout proba
        self.max_length = max_length
        
        #Embedding layers for the decoder input
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        #Attn linear layer
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        #Attn_combine linear layer
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        #Dropout
        self.dropout = nn.Dropout(self.dropout_p)
        
        #GRU layer
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        #Out linear layer
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #First, the input (target sequence) is embedded. Some weights are randomly zeroed out to facilitate
        #learning with the attention mechanism
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        #Attention is computed by combining the context vectors and the embedded input
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #Attention is applied on the encoded original sentence (in english)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        #We retrieve the embedded input and the context vector (with attention applied), and combine the two tensors
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        #The attended part of the input is fed into the lstm, conditioned by the hidden and cell states
        output, hidden = self.gru(output, hidden)
        #Retrieve token probabilities
        output = F.log_softmax(self.out(output[0]), dim=1)
        
        #In addition to the output and hidden states that are necessary for iterating, we return the attention
        #weights, that will provide some form of explainability
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

