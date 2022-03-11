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

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()

    def __call__(self, output, target):
        return self.loss(output,target)
    
class L1_Sp(nn.Module):
    def __init__(self):
        super(L1_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        self.l1 = L1()

    def __call__(self, output, target, log_op=False):
        sp_loss = self.l1(
                self.f_helper.wav_to_spectrogram(output, eps=1e-8),
                self.f_helper.wav_to_spectrogram(target, eps=1e-8)
            )
        return sp_loss
    
class SSR_DNN(pl.LightningModule):
    def __init__(self, channels):
        super(SSR_DNN, self).__init__()
        
        if(torch.cuda.is_available()): device = "cuda"
        else: device="cpu"
        
        model_name="dnn"

        self.valid = None
        self.fake = None

        self.train_step = 0
        self.val_step = 0
        self.val_result_save_dir = None
        self.val_result_save_dir_step = None
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}


        self.f_loss = L1_Sp()

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

        self.vocoder = Vocoder(sample_rate=44100).to(device)

        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        # masking
        self.generator = Generator().to(device)

    def get_vocoder(self):
        return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def forward(self, mel_orig, wav):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig, wav)

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
    def __init__(self):
        super(Generator, self).__init__()
        self.unet = DNN(channels=1)

    def forward(self,sp, noisy_wav):
        # Denoising
        unet_out = self.unet(sp, noisy_wav)['wav']
        return {'wav': unet_out, "clean": sp}
    
class DNN(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(DNN, self).__init__()
        window_size = 2048
        hop_size = 441
        activation = 'relu'
        momentum = 0.01
        center = True,
        pad_mode = 'reflect'
        window = 'hann'
        freeze_parameters = True

        self.nsrc = nsrc
        self.channels = channels
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
        )
        n_mel = 1025
        # n_mel = 128
        self.lstm = nn.Sequential(
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

    def forward(self, sp, wav):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        _, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(wav)
        # shapes: (batch_size, channels_num, time_steps, freq_bins)

        out_mag = self.lstm(sp)
        
        out_mag = torch.relu(out_mag) + sp

        out_real = out_mag * cos_in
        out_imag = out_mag * sin_in

        length = wav.shape[2]

        wav_out = self.f_helper.istft(out_real, out_imag, length)
        output_dict = {'wav': wav_out[:, None, :]}

        # wav_out = self.conv(wav)
        #
        # output_dict = {'wav': wav_out}

        return output_dict
