import torch
from torch import nn
import numpy as np


class CasualDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding=1):
        super().__init__()
        self.conv1D = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False, padding='same')
        self.ignoreOutIndex = (kernel_size - 1) * dilation

    def forward(self, x):
        # print("-----------CasualDilatedConv1D-----------")
        # print("x size:", x.size())
        output = self.conv1D(x)[..., :-self.ignoreOutIndex]
        # print("output size:", output.size())
        return output


class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, skipConnection):
        # print("-----------DenseLayer-----------")
        # print("skipConnection size:", skipConnection.size())
        out = torch.mean(skipConnection, dim=0)  # Mean over batch dimension (dim=0)
        # print("out size:", out.size())
        for i in range(2):
            out = self.relu(out)
            out = self.conv1d(out)
        # print("out size:", out.size())
        output = self.softmax(out)
        # print("output size:", output.size())
        return output


class ResBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.casualDilatedConv1D = CasualDilatedConv1D(res_channels, res_channels, kernel_size, dilation=dilation)
        self.resConv1D = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.skipConv1D = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputX, skipSize):
        #print("-----------ResBlock-----------")
        #print("inputX size:", inputX.size())
        #print("skipSize:", skipSize)
        x = self.casualDilatedConv1D(inputX)
        #print("x size:", x.size())
        x1 = self.tanh(x)
        #print("x1 size:", x1.size())
        x2 = self.sigmoid(x)
        #print("x2 size:", x2.size())
        x = x1 * x2
        #print("x size:", x.size())
        resOutput = self.resConv1D(x)
        #print("resOutput size:", resOutput.size())
        resOutput = resOutput + inputX[..., -resOutput.size(2):]
        #print("resOutput size:", resOutput.size())
        skipOutput = self.skipConv1D(x)
        #print("skipOutput size:", skipOutput.size())
        skipOutput = skipOutput[..., -skipSize:]
        #print("skipOutput size:", skipOutput.size())
        return resOutput, skipOutput


class StackOfResBlocks(nn.Module):

    def __init__(self, stack_size, layer_size, res_channels, skip_channels, kernel_size):
        super().__init__()
        buildDilationFunc = np.vectorize(self.buildDilation)
        dilations = buildDilationFunc(stack_size, layer_size)
        self.resBlocks = []
        for s, dilationPerStack in enumerate(dilations):
            for l, dilation in enumerate(dilationPerStack):
                resBlock = ResBlock(res_channels, skip_channels, kernel_size, dilation)
                self.add_module(f'resBlock_{s}_{l}', resBlock)  # Add modules manually
                self.resBlocks.append(resBlock)

    def buildDilation(self, stack_size, layer_size):
        # stack1=[1,2,4,8,16,...512]
        dilationsForAllStacks = []
        for stack in range(stack_size):
            dilations = []
            for layer in range(layer_size):
                dilations.append(2 ** layer)
            dilationsForAllStacks.append(dilations)
        return dilationsForAllStacks

    def forward(self, x, skipSize):
        #print("-----------StackOfResBlocks-----------")
        #print("x size:", x.size())
        #print("skipSize:", skipSize)
        resOutput = x
        skipOutputs = []
        for resBlock in self.resBlocks:
            resOutput, skipOutput = resBlock(resOutput, skipSize)
            skipOutputs.append(skipOutput)
        #print("resOutput size:", resOutput.size())
        #print("skipOutputs size:", skipOutputs[0].size())
        return resOutput, torch.stack(skipOutputs)


class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stack_size, layer_size):
        super().__init__()
        self.stack_size = stack_size
        self.layer_size = layer_size
        self.kernel_size = kernel_size
        self.casualConv1D = CasualDilatedConv1D(in_channels, in_channels, kernel_size, dilation=1)
        self.stackResBlock = StackOfResBlocks(self.stack_size, self.layer_size, in_channels, out_channels, kernel_size)
        self.denseLayer = DenseLayer(out_channels)

    def calculateReceptiveField(self):
        return np.sum([(self.kernel_size - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)


    def calculateOutputSize(self, x):
        return int(x.size(2)) - self.calculateReceptiveField()

    def forward(self, x):
        # x: b c t -> input data size
        x = self.casualConv1D(x)
        skipSize = self.calculateOutputSize(x)
        _, skipConnections = self.stackResBlock(x, skipSize)
        dense = self.denseLayer(skipConnections)
        return dense
    

class WaveNetClassifier(nn.Module):
    def __init__(self, seqLen, output_size):
        super().__init__()
        self.output_size = output_size
        self.wavenet = WaveNet(1, 1, 2, 3, 4)  # Adjusted to work with input size 256
        self.liner = nn.Linear(seqLen - self.wavenet.calculateReceptiveField() - 1, output_size)
        # No softmax here, because the final output is a scalar
        self.softmax = nn.Softmax(dim=-1)  # Optional, but not typically needed for a single output

    def forward(self, x):
        # print("Input shape:", x.shape)return np.sum([(self.kernel_size - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)
        x = self.wavenet(x)
        # print("After WaveNet shape:", x.shape)
        # x = x.view(x.size(0), -1)  # Flatten before Linear
        # print("Before Linear shape:", x.shape)
        x = self.liner(x)
        return x

















# from torch import nn
# import numpy as np
# import torch

# class Tanh():
#     def __call__(self, x) -> torch.Tensor:
#         return torch.tanh(input=x)
    
#     def parameters(self):
#         return []
    
# class Sigmoid():
#     def __call__(self, x) -> torch.Tensor:
#         return torch.sigmoid(input=x)
    
#     def parameters(self):
#         return []

# class Softmax():
#     def __init__(self, dim) -> None:
#         self.dim = dim

#     def __call__(self, x) -> torch.Tensor:
#         return torch.softmax(input=x, dim=self.dim)
    
#     def parameters(self):
#         return []

# class ReLU():
#     def __call__(self, x) -> torch.Tensor:
#         return torch.relu(input=x)
    
#     def parameters(self):
#         return []
    
# class Conv1d():
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False, padding='same') -> None:
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.bias = bias
#         self.padding = padding
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=1, bias=bias, padding=1) #do zaimplementowania

#     def to(self, device):
#         self.conv = self.conv.to(device)
        
#     def parameters(self):
#         return self.conv.parameters()
    
#     def forward(self, x):
#         return self.conv(x)[:,:,:-1]

#     def __call__(self, x):
#         return self.forward(x)

# class CasualDilatedConv1d():
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False, padding=1):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.padding = (kernel_size - 1) * dilation
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation, bias=bias, padding=1) #do zaimplementowania
#         self.ignore_idx = (kernel_size-1) * dilation

#     def to(self, device):
#         self.conv = self.conv.to(device)
        
#     def parameters(self):
#         return self.conv.parameters()

#     def forward(self, x):
#         print("----------CASUAL DILATED CONV--------")
#         print(f"Input size: {x.size()}")
#         output = self.conv(x)[..., :-1]
#         print(f"Output size after conv: {output.size()}")
#         print("----------CASUAL DILATED CONV--------")
#         return output

#     def __call__(self, x):
#         return self.forward(x)
    

# class ResidualBlock():
#     def __init__(self, res_channels, skip_channels, kernel_size, dilation):
#         self.out_channels = res_channels
#         self.skip_channels = skip_channels
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.casualDilatedConv1d1 = CasualDilatedConv1d(res_channels, res_channels, kernel_size=2, dilation=dilation)
#         self.resConv1d = nn.Conv1d(res_channels, res_channels, 1)
#         self.skipConv1d = nn.Conv1d(res_channels, skip_channels, 1)
#         self.tanh = Tanh()
#         self.sigmoid = Sigmoid() 

#     def to(self, device):
#         self.casualDilatedConv1d1 = self.casualDilatedConv1d1.to(device)
#         self.resConv1d = self.resConv1d.to(device)
#         self.skipConv1d = self.skipConv1d.to(device)

#     def parameters(self):
#         parameters = []
#         parameters += self.casualDilatedConv1d1.parameters()
#         parameters += self.resConv1d.parameters()
#         parameters += self.skipConv1d.parameters()
#         return parameters
        
    
#     def forward(self, inputX, skipSize):
#         print("----------RESIDUAL BLOCK--------")
#         print(f"Input size: {inputX.size()}")
#         x = self.casualDilatedConv1d1(inputX)
#         print(f"Output size after casual dilated conv: {x.size()}")
#         x1 = self.tanh(x)
#         print(f"Output size after tanh: {x1.size()}")
#         x2 = self.sigmoid(x)
#         print(f"Output size after sigmoid: {x2.size()}")
#         gated = x1 * x2
#         print(f"Output size after gated: {gated.size()}")
#         output = self.resConv1d(gated)
#         print(f"Output size after res conv1d: {output.size()}")
#         input_cut = inputX[:,:, -output.size(2):]
#         print(f"Input cut size: {input_cut.size()}")
#         residual_output = output + input_cut
#         print(f"Residual output size: {residual_output.size()}")
# # Calculate skip size dynamically based on the output size
#         skip = self.skipConv1d(gated)
#         print(f"Skip size before slicing: {skip.size()}")
        
#         # Dynamically adjust skip size based on available output size
#         skip_size = min(skip.size(2), skipSize)

#         # Ensure the skip size is the same by padding the skip connection if needed
#         if skip.size(2) < skipSize:
#             padding = skipSize - skip.size(2)
#             skip = F.pad(skip, (0, padding))  # Pad along the last dimension
#         else:
#             skip = skip[:, :, -skip_size:]  # Slice the skip connection
#         print(f"Skip size = {-skipSize}")
#         print(f"Skip size after slicing: {skip.size()}")

#         print("----------RESIDUAL BLOCK--------")
#         return residual_output, skip
#         # skip = self.skipConv1d(gated)
#         # print(f"Skip size: {skip.size()}")
#         # skip = skip[:,:,-skipSize:]
#         # print(f"Residual output size: {residual_output.size()}")
#         # print(f"Skip size: {skip.size()}")
#         # print("----------RESIDUAL BLOCK--------")

#         return residual_output, skip


#     def __call__(self, x, skipSize):
#         return self.forward(x, skipSize)
    

# class StackResidualBlock():
#     def __init__(self, stack_size, layer_size, res_channels, skip_channels, kernel_size):
#         self.res_channels = res_channels
#         self.skip_channels = skip_channels
#         self.kernel_size = kernel_size
#         self.stack_size = stack_size
#         self.layer_size = layer_size
#         buildDilationFunc = np.vectorize(self.buildDilation)
#         self.dilations = buildDilationFunc(stack_size, layer_size)
#         self.residual_blocks = []
#         for dilation_list in self.dilations:
#             for dilation in dilation_list:
#                 self.residual_blocks.append(ResidualBlock(res_channels, skip_channels, kernel_size, dilation))

#     def to(self, device):
#         for idx, res_block in enumerate(self.residual_blocks):
#             self.residual_blocks[idx] = res_block.to(device)

#     def buildDilation(self, stack_size, layer_size):
#         return [[2**layer for layer in range(layer_size)] for _ in range(stack_size)]
        
#     def parameters(self):
#         parameters = []
#         for res_block in self.residual_blocks:
#             parameters += res_block.parameters()
#         return parameters
    
#     # def forward(self, inputX, skipSize):
#     #     residual_output = inputX
#     #     skip_outputs = []
#     #     for res_block in self.residual_blocks:
#     #         residual_output, skip_output = res_block(residual_output, skipSize)
#     #         skip_outputs.append(skip_output)
#     #     return residual_output, torch.stack(skip_outputs)

#     def forward(self, inputX, skipSize):
#         residual_output = inputX
#         skip_outputs = []

#         # Loop through each residual block
#         for res_block in self.residual_blocks:
#             residual_output, skip_output = res_block(residual_output, skipSize)
#             skip_outputs.append(skip_output)

#         # Stack all the skip outputs
#         skip_connections = torch.cat(skip_outputs, dim=-1)  # Concatenate along the skip_channels axis
#         return residual_output, skip_connections

#     def __call__(self, x, skipSize):
#         return self.forward(x, skipSize)


# class DenseLayer():
#     def __init__(self, in_channels):
#         self.in_channels = in_channels
#         self.relu = ReLU()
#         self.softmax = Softmax(dim=1)
#         self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

#     def to(self, device):
#         self.conv1d = self.conv1d.to(device)

#     def parameters(self):
#         return self.conv1d.parameters()

#     def forward(self, x):
#         output = self.relu(x)
#         out = self.conv1d(output)
#         out = self.relu(out)
#         out = self.conv2d(out)
#         return self.softmax(out)

#     def __call__(self, x):
#         return self.forward(x)
    

# class WaveNet():
#     def __init__(self, layer_size, stack_size, in_channels, res_channels):
#         self.layer_size = layer_size
#         self.stack_size = stack_size
#         self.in_channels = in_channels
#         self.res_channels = res_channels
#         self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)
#         self.casualConv1d = CasualDilatedConv1d(in_channels, res_channels, kernel_size=2, dilation=1)
#         self.stackResidualBlock = StackResidualBlock(self.stack_size, self.layer_size, res_channels, in_channels, kernel_size=2)
#         self.denseLayer = DenseLayer(in_channels)

#     @staticmethod
#     def calc_receptive_fields(layer_size, stack_size):
#         layers = [2 ** i for i in range(0, layer_size)] * stack_size
#         num_receptive_fields = np.sum(layers)

#         return int(num_receptive_fields)

#     def calc_output_size(self, x):
#         output_size = int(x.size(2)) - self.receptive_fields

#         # self.check_input_size(x, output_size)

#         return output_size

#     # def check_input_size(self, x, output_size):
#     #     if output_size < 1:
#     #         raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

#     def to(self, device):
#         self.casualConv1d = self.casualConv1d.to(device)
#         self.stackResidualBlock = self.stackResidualBlock.to(device)
#         self.denseLayer = self.denseLayer.to(device)

#     def parameters(self):
#         # Ensure that all submodule parameters are concatenated correctly
#         params = []
#         params += self.casualConv1d.parameters()
#         params += self.stackResidualBlock.parameters()
#         params += self.denseLayer.parameters()
#         return params  # Returns a list of all parameters

#     def calculateReceptiveField(self):
#         return np.sum([(2 - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)

#     def calculateOutputSize(self, x):
#         return int(x.size(2)) - self.calculateReceptiveField()

#     # def forward(self, x):
#     #     x = self.casualConv1d(x)
#     #     skipSize = self.calculateOutputSize(x)
#     #     if self.out_size > skipSize:
#     #         raise ValueError(f"Output size {self.out_size} is bigger than receptive field {skipSize}.")
#     #     residual_output, skip_outputs = self.stackResidualBlock(x, self.out_size)
#     #     return self.denseLayer(skip_outputs)

#     def forward(self, x):
#         print(f"Input size: {x.size()}")
#         # output = x.transpose(1, 2)
#         # print(f"Output size after transpose: {output.size()}")
#         output_size = self.calculateOutputSize(x)
#         output = self.casualConv1d(x)
#         print(f"Output size after casual conv: {output.size()}")
#         skip_connections = self.stackResidualBlock(output, output_size)
#         # print(f"skip connections size after stack residual block: {skip_connections.size()}")
#         for skip_connection in skip_connections:
#             print(f"Skip connection size: {skip_connection.size()}")
#         output = torch.sum(skip_connections, dim=0)
#         print(f"Output size after sum: {output.size()}")
#         output = self.denseLayer(output)
#         print(f"Output size after dense layer: {output.size()}")
#         return output.transpose(1, 2).contiguous()

#     def __call__(self, x):
#         return self.forward(x)


# class WaveNetClassifier(nn.Module):
#     def __init__(self,seqLen,output_size):
#         super().__init__()
#         self.output_size=output_size
#         self.wavenet=WaveNet(1,1,2,3,4)
#         self.liner=nn.Linear(seqLen-self.wavenet.calculateReceptiveField(),output_size)
#         self.softmax=nn.Softmax(-1)
    
#     def forward(self,x):
#         x=self.wavenet(x)
#         x=self.liner(x)
#         return self.softmax(x)
