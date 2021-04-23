import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import torch.nn.functional as F
from tcn import TemporalConvNet

# ----------------------------------------------------------------------------------------------------------------------
class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, teacher_forcing=0):
        super(DeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        
        # Attention
        self.attention = Attention(128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        hidden_size = 128
        self.attention_linear = nn.Linear(128 + self.num_features, 46)
        self.decoder = nn.GRUCell(
            input_size=128,
            hidden_size=128,
        )
        self.out = nn.Linear(128, 45)
        self.teacher_forcing = teacher_forcing

        self.decoder_cell = AttentionDecoderCell(self.num_features, 128, 46)
        

    def forward(self, x_padded, x_lengths=None, x_labels=None):
        
        #x_packed = packer(x_padded, x_lengths, batch_first=True)
        #print(x_padded.shape)
        # Encode
        output, _ = self.gru1(x_padded)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)
        prev_hidden = hidden[0]
        # Pass to attention with the original padding
        #output_padded, _ = padder(output, batch_first=True)
        output_size = len(x_padded[0])
        #attn_output = self.attention(output, hidden[-1:])       
#        if torch.cuda.is_available():
#            outputs = torch.zeros(x_padded.size(0), output_size, device='cuda')
#        else:
#            outputs = torch.zeros(x_padded.size(0), output_size)

        outputs = []
        y_prev = x_padded[:, -1, :]
        

        
        output = output.permute(1, 0, 2)
        for i in range(output_size):
            rnn_output = self.out(output[i])
            outputs.append(rnn_output)
        """

        for i in range(output_size):
            if self.training == True:
                if (x_labels is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                    y_prev = x_labels[:, i]
            
            attention_input = torch.cat((prev_hidden, y_prev), axis=1)
            attention_weights = F.softmax(self.attention_linear(attention_input),-1).unsqueeze(1)
            attention_combine = torch.bmm(attention_weights, output).squeeze(1)
            prev_hidden = self.decoder(attention_combine, prev_hidden)
            #print("prev hidden: ", prev_hidden.shape)
            rnn_output = self.out(prev_hidden)
            y_prev = rnn_output
            
            #rnn_output, prev_hidden = self.decoder_cell(output, prev_hidden, y_prev)
            #y_prev = rnn_output
            outputs.append(rnn_output)
        """
        
        outputs = torch.stack(outputs).permute(1, 0, 2)
        return outputs
        # Classify
        #return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, sequence_len):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)        
        self.attention_linear = nn.Linear(128 + input_feature_len, 46)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=128,
            hidden_size=128,
        )
        self.out = nn.Linear(128, 45)
        
    def forward(self, encoder_output, prev_hidden, y):
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input),-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden



# ----------------------------------------------------------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2))
        return self.linear(y1.transpose(1, 2))        
