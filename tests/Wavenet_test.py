from unittest import TestCase
from layers import WaveNet
from torch import nn
import numpy as np
from einops import repeat
import torch

class TestWavenet(TestCase):
    def setUp(self) -> None:
        super().setUp()
        input_channels = 1
        output_channels = 1
        sample_size = 2000
        batch_size = 8
        self.wavenet = WaveNet(input_channels,input_channels, output_channels, kernel_size=3, stack_size=2, layer_size=5)
        self.input = nn.Parameter(torch.randn((sample_size, input_channels)))
        self.input = repeat(self.input, 'c t->b c t', b=batch_size)

    def test_runWavenet(self):
        output = self.wavenet(self.input)
        pass
