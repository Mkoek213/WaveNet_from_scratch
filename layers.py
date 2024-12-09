from torch import nn
import numpy as np
import torch

class Tanh():
    def __call__(self, x):
        return torch.tanh(x)
    
    def parameters(self):
        return []
    
class Sigmoid():
    def __call__(self, x):
        return torch.sigmoid(x)
    
    def parameters(self):
        return []

class Softmax():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return torch.softmax(x, dim=self.dim)
    
    def parameters(self):
        return []

class ReLU():
    def __call__(self, x):
        return torch.relu(x)
    
    def parameters(self):
        return []
    
class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias) #do zaimplementowania
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])
    
    def forward(self, x):
        return self.conv(x)

    def __call__(self, x):
        return self.forward(x)

class CasualDilatedConv1d():
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias) #do zaimplementowania
        self.ignore_idx = (kernel_size-1) * dilation
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])
        
    def forward(self, x):
        return self.conv(x)[..., :-self.ignore_idx]

    def __call__(self, x):
        return self.forward(x)
    

class ResidualBlock():
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.casualDilatedConv1d1 = CasualDilatedConv1d(in_channels, out_channels, kernel_size, dilation)
        self.resConv1d = Conv1d(out_channels, out_channels, kernel_size=1)
        self.skipConv1d = Conv1d(in_channels, skip_channels, kernel_size=1)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid() 

    def parameters(self):
        return self.casualDilatedConv1d1.parameters() + self.resConv1d.parameters() + self.skipConv1d.parameters()
    
    def forward(self, input):
        x = self.casualDilatedConv1d1(input)
        x1 = self.tanh(x)
        x2 = self.sigmoid(x)
        x = x1 * x2
        residual_output = self.resConv1d(x) + input
        skip_output = self.skipConv1d(x)
        return residual_output, skip_output

    def __call__(self, x):
        return self.forward(x)
    

class StackResidualBlock():
    def __init__(self, stack_size, layer_size, in_channels, out_channels, skip_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.layer_size = layer_size
        buildDilationFunc = np.vectorize(self.buildDilation)
        dilations = buildDilationFunc(stack_size, layer_size)
        self.residual_blocks = []
        for dilation in dilations:
            self.residual_blocks.append(ResidualBlock(in_channels, out_channels, skip_channels, kernel_size, dilation))

    
    def buildDilation(self, stack_size, layer_size):
        return [[2**layer for layer in range(layer_size)] for _ in range(stack_size)]
        
    def parameters(self):
        return [param for block in self.residual_blocks for param in block.parameters()]
    
    def forward(self, input):
        residual_output = input
        skip_outputs = []
        for res_block in self.residual_blocks:
            residual_output, skip_output = res_block.forward(residual_output)
            skip_outputs.append(skip_output)
        return residual_output, torch.stack(skip_outputs)

    def __call__():
        return self.forward(x)


class DenseLayer():
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        self.conv1d = Conv1d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def parameters(self):
        return self.conv1d.parameters()

    def forward(self, x):
        out = torch.sum(x, dim=2)
        for i in range(2):
            out = self.relu(out)
            out = self.conv1d(out)
        
        return self.softmax(out)

    def __call__(self, x):
        return self.forward(x)
    

class WaveNet():
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, stack_size, layer_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.layer_size = layer_size
        self.casualConv1d = CasualDilatedConv1d(in_channels, out_channels, kernel_size, dilation=1)
        self.stackResidualBlock = StackResidualBlock(stack_size, layer_size, in_channels, out_channels, skip_channels, kernel_size)
        self.denseLayer = DenseLayer(skip_channels, out_channels)

    def parameters(self):
        return self.stackResidualBlock.parameters() + self.denseLayer.parameters()

    def forward(self, x):
        x = self.casualConv1d(x)
        residual_output, skip_outputs = self.stackResidualBlock(x)
        return self.denseLayer(skip_outputs)

    def __call__(self, x):
        return self.forward(x)
