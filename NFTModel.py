import torch
import torch.nn as nn
import numpy as np

class ConvLSTMBlock(nn.Module): # iNFT is sensitive to pulse power, so we'll not use normalization layers
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 2,
                 padding = 1,
                 padding_mode = "circular",
                 activation = 'LeakyReLU',
                 lstm_layers = 3
                 ):
        super(ConvLSTMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.Tanh() if activation == 'Tanh' else nn.LeakyReLU(0.2) if activation == 'LeakyReLU' else nn.Identity()
        )
        self.lstm = nn.LSTM(out_channels, out_channels, num_layers=lstm_layers, dropout=(0.2 if lstm_layers>1 else 0.0), batch_first=True) if lstm_layers > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,1)
        x = self.lstm(x)[0]
        x = x.permute(0,2,1)
        return x

class UpSampleLSTMBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 2,
                 padding = 1,
                 output_padding = 1,
                 activation = 'Tanh',
                 lstm_layers = 3
                 ):
        super(UpSampleLSTMBlock, self).__init__()
        self.lstm_layers = lstm_layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.Tanh() if activation == 'Tanh' else nn.LeakyReLU(0.2) if activation == 'LeakyReLU' else nn.Identity()
        )
        self.lstm = nn.LSTM(out_channels, out_channels, num_layers=lstm_layers, dropout=(0.2 if lstm_layers>1 else 0.0), batch_first=True) if lstm_layers > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.upsample(x)
        if self.lstm_layers > 0:
            x = x.permute(0,2,1)
            x = self.lstm(x)[0]
            x = x.permute(0,2,1)
        return x
    
class ConvLSTMAutoEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 sequence_length,
                 kernel_size = 3,
                 stride = 2,
                 padding = 1,
                 features = [16, 32, 64, 128],
                 lstm_layers = 1,
                 forward = False
                 ):
        super(ConvLSTMAutoEncoder, self).__init__()
        layers = []
        layer_lens = []
        for feature in features:
            layers.append(ConvLSTMBlock(in_channels=in_channels, out_channels=feature, kernel_size=kernel_size, stride=stride, padding=padding, activation=('Tanh' if forward else 'LeakyReLU'), lstm_layers=lstm_layers))
            in_channels = feature
            layer_lens.append( sequence_length )
            sequence_length = np.floor((sequence_length + 2 * padding - kernel_size) / stride + 1)
        self.encode = nn.Sequential(*layers)
        layers = []
        layer_idx = len(features) - 1
        for feature in features[::-1][1:]+[out_channels,]:
            padding = int(np.ceil(((sequence_length - 1) * stride - layer_lens[layer_idx] + kernel_size) / 2))
            output_padding = int(layer_lens[layer_idx] - (sequence_length - 1) * stride + 2 * padding - kernel_size)
            layers.append(UpSampleLSTMBlock(in_channels=in_channels, out_channels=feature, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, activation=(None if feature==out_channels else 'LeakyReLU' if forward else 'Tanh'), lstm_layers=(0 if feature==out_channels else lstm_layers)))
            in_channels = feature
            sequence_length = layer_lens[layer_idx]
            layer_idx -= 1
        self.decode = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def test_model():
    x = torch.randn((64, 2, 2048))
    model = ConvLSTMAutoEncoder(2, 2, 2048, features=[64,64,64])
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == "__main__":
    test_model()
