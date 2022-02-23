import sys
sys.path.append("/vol/research/dcase2022/sr_eval_vctk")
sys.path.append("/vol/research/dcase2022/sr_eval_vctk/testees")
import time
import librosa
import torch
from testees.voicefixer_local.models.gsr_voicefixer_unet import VoiceFixer as Model
from omegaconf import OmegaConf as OC
import numpy as np 
from utils import *
from sr_eval_vctk import SR_Eval, BasicTestee

model = None
EPS=1e-9


class VoiceFixerTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
    def pre(self, input):
        self.checkpoint("pre 1")
        input = input[None, ...].to(self.device)
        self.checkpoint("pre 2")
        sp, _, _ = self.model.f_helper.wav_to_spectrogram_phase(input)
        self.checkpoint("pre 3")
        mel_orig = self.model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def checkpoint(self, name):
        # print(name, time.time()-self.current)
        # self.current = time.time()
        pass

    def infer(self, x, target):
        metrics = {}
        self.checkpoint("start infer")
        with torch.no_grad():
            self.checkpoint("to cuda")
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            self.checkpoint("pre 0")
            _, mel_noisy = self.pre(segment)
            self.checkpoint("pre end")
            out = self.model(mel_noisy)
            self.checkpoint("analysis end")
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            self.checkpoint("vocoder end")
            # import ipdb; ipdb.set_trace()
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        self.checkpoint("end infer")
        return self.tensor2numpy(out), metrics

if __name__ == '__main__':
    # import soundfile as sf
    testee = VoiceFixerTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/voicefixer_local/ckpt/unet_butter_2k_44k/epoch=11-step=22499-val_l=0.27.ckpt",
                              model_name="unet", 
                              device="cuda")
    
    test_name = "VF_UNet_SRONLY_Butter_Rand_Order_2k_44k"
    
    sr_eval = SR_Eval(testee, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      sr=44100,
                        setting_lowpass_filtering = {
                          "filter":["cheby","butter"],
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                          "filter_order": [3,6,9]
                      }, 
                      setting_subsampling = {
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                      }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                      }, 
                      setting_mp3_compression = {
                          "original_low_kbps": [32, 48, 64, 96, 128],
                      } 
    )
    
    sr_eval.evaluate(limit_test_nums=10, limit_speaker=-1)
    
    



