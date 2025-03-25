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
        self.wavenet = WaveNet(10, 5, 256, 512)
        self.input = nn.Parameter(torch.randn((input_channels, sample_size)))
        self.input = repeat(self.input, 'c t->b c t', b=batch_size)

    def test_runWavenet(self):
        output = self.wavenet(self.input)
        pass

    def test_calculateReceptiveField(self):
        print(self.wavenet.calculateReceptiveField())
        # self.assertEqual(receptive_field, 62)
