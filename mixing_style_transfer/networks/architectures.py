""" 
"Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"

    Implementation of neural networks used in the task 'Music Mixing Style Transfer'
        - 'FXencoder'
        - TCN based 'MixFXcloner'

    We modify the TCN implementation from: https://github.com/csteinmetz1/micro-tcn
    which was introduced in the work "Efficient neural networks for real-time modeling of analog dynamic range compression" by Christian J. Steinmetz, and Joshua D. Reiss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))

from networks.network_utils import *



# FXencoder that extracts audio effects from music recordings trained with a contrastive objective
class FXencoder(nn.Module):
    def __init__(self, config):
        super(FXencoder, self).__init__()
        # input is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        for i in range(len(config["kernels"])):
            if config["conv_block"]=='res':
                encoder.append(Res_ConvBlock(dimension=1, \
                                                in_channels=config["channels"][i], \
                                                out_channels=config["channels"][i+1], \
                                                kernel_size=config["kernels"][i], \
                                                stride=config["strides"][i], \
                                                padding="SAME", \
                                                dilation=config["dilation"][i], \
                                                norm=config["norm"], \
                                                activation=config["activation"], \
                                                last_activation=config["activation"]))
            elif config["conv_block"]=='conv':
                encoder.append(ConvBlock(dimension=1, \
                                            layer_num=1, \
                                            in_channels=config["channels"][i], \
                                            out_channels=config["channels"][i+1], \
                                            kernel_size=config["kernels"][i], \
                                            stride=config["strides"][i], \
                                            padding="VALID", \
                                            dilation=config["dilation"][i], \
                                            norm=config["norm"], \
                                            activation=config["activation"], \
                                            last_activation=config["activation"], \
                                            mode='conv'))
        self.encoder = nn.Sequential(*encoder)

        # pooling method
        self.glob_pool = nn.AdaptiveAvgPool1d(1)

    # network forward operation
    def forward(self, input):
        enc_output = self.encoder(input)
        glob_pooled = self.glob_pool(enc_output).squeeze(-1)

        # outputs c feature
        return glob_pooled


# MixFXcloner which is based on a Temporal Convolutional Network (TCN) module
    # original implementation : https://github.com/csteinmetz1/micro-tcn
import pytorch_lightning as pl
class TCNModel(pl.LightningModule):
    """ Temporal convolutional network with conditioning module.
        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
            skip_connections (bool): Skip connections from each block to the output. Default: False
            num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=3, 
                 dilation_growth=1, 
                 channel_growth=1, 
                 channel_width=32, 
                 stack_size=10,
                 cond_dim=2048,
                 grouped=False,
                 causal=False,
                 skip_connections=False,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(TCNModel, self).__init__()
        self.save_hyperparameters()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.hparams.kernel_size, 
                                        dilation=dilation,
                                        padding="same" if self.hparams.causal else "valid",
                                        causal=self.hparams.causal,
                                        cond_dim=cond_dim, 
                                        grouped=self.hparams.grouped,
                                        conditional=True if self.hparams.nparams > 0 else False))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, cond):
        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            # for SeFa
            if isinstance(cond, list):
                x = block(x, cond[idx])
            else:
                x = block(x, cond)
            skips = 0

        out = torch.clamp(self.output(x + skips), min=-1, max=1)

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=1)
        parser.add_argument('--nblocks', type=int, default=4)
        parser.add_argument('--kernel_size', type=int, default=5)
        parser.add_argument('--dilation_growth', type=int, default=10)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=32)
        parser.add_argument('--stack_size', type=int, default=10)
        parser.add_argument('--grouped', default=False, action='store_true')
        parser.add_argument('--causal', default=False, action="store_true")
        parser.add_argument('--skip_connections', default=False, action="store_true")

        return parser


class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                dilation=1, 
                cond_dim=2048, 
                grouped=False, 
                causal=False,
                conditional=False, 
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        self.pad_length = ((kernel_size-1)*dilation) if self.causal else ((kernel_size-1)*dilation)//2
        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=self.pad_length,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(cond_dim, out_ch)
        self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.LeakyReLU()
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x, p):
        x_in = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.film(x, p)

        x_res = self.res(x_in)

        if self.causal:
            x = x[..., :-self.pad_length]
        x += x_res

        return x



if __name__ == '__main__':
    ''' check model I/O shape '''
    import yaml
    with open('networks/configs.yaml', 'r') as f:
        configs = yaml.full_load(f)

    batch_size = 32
    sr = 44100
    input_length = sr*5
    
    input = torch.rand(batch_size, 2, input_length)
    print(f"Input Shape : {input.shape}\n")
    

    print('\n========== Audio Effects Encoder (FXencoder) ==========')
    model_arc = "FXencoder"
    model_options = "default"

    config = configs[model_arc][model_options]
    print(f"configuration: {config}")

    network = FXencoder(config)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    # model inference
    output_c = network(input)
    print(f"Output Shape : {output_c.shape}")


    print('\n========== TCN based MixFXcloner ==========')
    model_arc = "TCN"
    model_options = "default"

    config = configs[model_arc][model_options]
    print(f"configuration: {config}")

    network = TCNModel(nparams=config["condition_dimension"], ninputs=2, noutputs=2, \
                        nblocks=config["nblocks"], \
                        dilation_growth=config["dilation_growth"], \
                        kernel_size=config["kernel_size"], \
                        channel_width=config["channel_width"], \
                        stack_size=config["stack_size"], \
                        cond_dim=config["condition_dimension"], \
                        causal=config["causal"])
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}\tReceptive field duration : {network.compute_receptive_field() / sr:.3f}")

    ref_embedding = output_c
    # model inference
    output = network(input, output_c)
    print(f"Output Shape : {output.shape}")

