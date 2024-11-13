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
    
class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False) #do zaimplementowania
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])
    
    def forward(self, x):
        return self.conv(x)

class CasualDilatedConv1d():
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False) #do zaimplementowania
        self.ignore_idx = (kernel_size-1) * dilation
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])
        
    def forward(self, x):
        return self.conv(x)[..., :-self.ignore_idx]
    

class ResidualBlock():
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.casualDilatedConv1d1 = CasualDilatedConv1d(in_channels, out_channels, kernel_size, dilation)
        self.resConv1d = Conv1d(out_channels, out_channels, kernel_size=1, dilation=1)
        self.skipConv1d = Conv1d(in_channels, skip_channels, kernel_size=1, dilation=1)
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