from torch import nn
import numpy as np
import torch

class Tanh():
    def __call__(self, x) -> torch.Tensor:
        return torch.tanh(input=x)
    
    def parameters(self):
        return []
    
class Sigmoid():
    def __call__(self, x) -> torch.Tensor:
        return torch.sigmoid(input=x)
    
    def parameters(self):
        return []

class Softmax():
    def __init__(self, dim) -> None:
        self.dim = dim

    def __call__(self, x) -> torch.Tensor:
        return torch.softmax(input=x, dim=self.dim)
    
    def parameters(self):
        return []

class ReLU():
    def __call__(self, x) -> torch.Tensor:
        return torch.relu(input=x)
    
    def parameters(self):
        return []
    
class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False, padding='same') -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, padding=padding) #do zaimplementowania
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])
    
    def forward(self, x):
        return self.conv(x)

    def __call__(self, x):
        return self.forward(x)

class CasualDilatedConv1d():
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, padding='same') #do zaimplementowania
        self.ignore_idx = (kernel_size-1) * dilation
        
    def parameters(self):
        return [self.conv.weight] + ([] if self.conv.bias is None else [self.conv.bias])

    def forward(self, x):
        return self.conv(x)[..., :-self.ignore_idx]

    def __call__(self, x):
        return self.forward(x)
    

class ResidualBlock():
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        self.out_channels = res_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.casualDilatedConv1d1 = CasualDilatedConv1d(res_channels, res_channels, kernel_size, dilation=dilation)
        self.resConv1d = Conv1d(res_channels, res_channels, kernel_size=1, dilation=dilation)
        self.skipConv1d = Conv1d(res_channels, skip_channels, kernel_size=1, dilation=dilation)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid() 

    def parameters(self):
        return self.casualDilatedConv1d1.parameters() + self.resConv1d.parameters() + self.skipConv1d.parameters()
        
    
    def forward(self, inputX, skipSize):
        x = self.casualDilatedConv1d1(inputX)
        x1 = self.tanh(x)
        x2 = self.sigmoid(x)
        x = x1 * x2
        residual_output = self.resConv1d(x)
        residual_output += inputX[..., -residual_output.size(2):]
        skip_output = self.skipConv1d(x)
        skip_output = skip_output[..., -skipSize:]
        return residual_output, skip_output


    def __call__(self, x, skipSize):
        return self.forward(x, skipSize)
    

class StackResidualBlock():
    def __init__(self, stack_size, layer_size, res_channels, skip_channels, kernel_size):
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.layer_size = layer_size
        buildDilationFunc = np.vectorize(self.buildDilation)
        self.dilations = buildDilationFunc(stack_size, layer_size)
        self.residual_blocks = []
        for dilation_list in self.dilations:
            for dilation in dilation_list:
                self.residual_blocks.append(ResidualBlock(res_channels, skip_channels, kernel_size, dilation))

    
    def buildDilation(self, stack_size, layer_size):
        return [[2**layer for layer in range(layer_size)] for _ in range(stack_size)]
        
    def parameters(self):
        return [param for block in self.residual_blocks for param in block.parameters()]
    
    def forward(self, inputX, skipSize):
        residual_output = inputX
        skip_outputs = []
        for res_block in self.residual_blocks:
            residual_output, skip_output = res_block(residual_output, skipSize)
            skip_outputs.append(skip_output)
        return residual_output, torch.stack(skip_outputs)

    def __call__(self, x, skipSize):
        return self.forward(x, skipSize)


class DenseLayer():
    def __init__(self, in_channels):
        self.in_channels = in_channels
        self.relu = ReLU()
        self.softmax = Softmax(dim=2)
        self.conv1d = Conv1d(in_channels, in_channels, kernel_size=1, dilation=1, bias=False)

    def parameters(self):
        return self.conv1d.parameters()

    def forward(self, x):
        out = torch.sum(x, dim=1)
        for i in range(2):
            out = self.relu(out)
            out = self.conv1d(out)
        
        return self.softmax(out)

    def __call__(self, x):
        return self.forward(x)
    

class WaveNet():
    def __init__(self, in_channels, out_channels, kernel_size, stack_size, layer_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.layer_size = layer_size
        self.casualConv1d = CasualDilatedConv1d(in_channels, in_channels, kernel_size, dilation=1)
        self.stackResidualBlock = StackResidualBlock(self.stack_size, self.layer_size, in_channels, out_channels, kernel_size)
        self.denseLayer = DenseLayer(out_channels)

    def parameters(self):
        return self.stackResidualBlock.parameters() + self.denseLayer.parameters()

    def calculateReceptiveField(self):
        return np.sum([(self.kernel_size - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)

    def calculateOutputSize(self, x):
        return int(x.size(2)) - self.calculateReceptiveField()

    def forward(self, x):
        x = self.casualConv1d(x)
        skipSize = self.calculateOutputSize(x)
        residual_output, skip_outputs = self.stackResidualBlock(x, skipSize)
        return self.denseLayer(skip_outputs)

    def __call__(self, x):
        return self.forward(x)
