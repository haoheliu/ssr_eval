import sys     
sys.path.append("/vol/research/dcase2022/sr_eval_vctk/testees")

import torch.utils
import torch.nn as nn
import torch.utils.data
from voicefixer import Vocoder
import os
import pytorch_lightning as pl
from tools.pytorch.modules.fDomainHelper import FDomainHelper
from tools.pytorch.mel_scale import MelScale
from voicefixer_local.models.components.unet import UNetResComplex_100Mb
import numpy as np   

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BN_GRU(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,layer=1, bidirectional=False, batchnorm=True, dropout=0.0):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if(batchnorm):self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self,inputs):
        # (batch, 1, seq, feature)
        if(self.batchnorm):inputs = self.bn(inputs)
        out,_ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)

class VoiceFixer(pl.LightningModule):
    def __init__(self, channels):
        super(VoiceFixer, self).__init__()
        
        if(torch.cuda.is_available()): device = "cuda"
        else: device="cpu"
        
        model_name="unet"
        
        self.channels = channels
        
        self.vocoder = Vocoder(sample_rate=44100).to(device)

        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.f_helper = FDomainHelper(
            window_size=2048,
            hop_size=441,
            center=True,
            pad_mode="reflect",
            window="hann",
            freeze_parameters=True,
        ).to(device)

        self.mel = MelScale(n_mels=128,
                            sample_rate=44100,
                            n_stft=2048 // 2 + 1).to(device)

        # masking
        self.generator = Generator(model_name).to(device)

    def get_vocoder(self):
        return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def forward(self, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig)

def to_log(input):
    assert torch.sum(input < 0) == 0, str(input)+" has negative values counts "+str(torch.sum(input < 0))
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input

class BN_GRU(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,layer=1, bidirectional=False, batchnorm=True, dropout=0.0):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if(batchnorm):self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self,inputs):
        # (batch, 1, seq, feature)
        if(self.batchnorm):inputs = self.bn(inputs)
        out,_ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)

class Generator(nn.Module):
    def __init__(self, model_name="unet"):
        super(Generator, self).__init__()
        if(model_name == "unet"):
            from models.components.unet import UNetResComplex_100Mb
            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif(model_name=="unet_small"):
            from models.components.unet_small import UNetResComplex_100Mb
            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif(model_name=="bigru"):
            n_mel = 128
            self.analysis_module = nn.Sequential(
                    nn.BatchNorm2d(1),
                    nn.Linear(n_mel, n_mel * 2),
                    BN_GRU(input_dim=n_mel*2, hidden_dim=n_mel*2, bidirectional=True, layer=2),
                    nn.ReLU(),
                    nn.Linear(n_mel*4, n_mel*2),
                    nn.ReLU(),
                    nn.Linear(n_mel*2, n_mel),
                )
        elif(model_name=="dnn"):
            n_mel = 128
            self.analysis_module = nn.Sequential(
                    nn.Linear(n_mel, n_mel * 2),
                    nn.ReLU(),
                    nn.BatchNorm2d(1),
                    nn.Linear(n_mel * 2, n_mel * 4),
                    nn.ReLU(),
                    nn.BatchNorm2d(1),
                    nn.Linear(n_mel * 4, n_mel * 8),
                    nn.ReLU(),
                    nn.BatchNorm2d(1),
                    nn.Linear(n_mel * 8, n_mel * 4),
                    nn.ReLU(),
                    nn.BatchNorm2d(1),
                    nn.Linear(n_mel * 4, n_mel * 2),
                    nn.ReLU(),
                    nn.Linear(n_mel * 2, n_mel),
                )
        else:
            pass # todo warning
    def forward(self, mel_orig):
        out = self.analysis_module(to_log(mel_orig))
        if(type(out) == type({})):
            out = out['mel']
        mel = out + to_log(mel_orig)
        return {'mel': mel}
