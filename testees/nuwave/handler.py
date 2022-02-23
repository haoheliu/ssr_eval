import sys
sys.path.append("/vol/research/dcase2022/sr_eval_vctk")

import librosa
import torch
from lightning_model import NuWave
from omegaconf import OmegaConf as OC
import numpy as np 
from utils import trim_center
from sr_eval_vctk import SR_Eval, BasicTestee
import time 
model = None


class NuWaveTestee(BasicTestee):
    def __init__(self, ckpt, device) -> None:
        self.hparams = OC.load('hparameter.yaml')
        self.model = NuWave(self.hparams, False).to(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=torch.device(device))['state_dict'])
        self.model.eval()
        self.model.freeze()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
    
    def checkpoint(self, name):
        print(name, time.time()-self.current)
        self.current = time.time()
        # pass
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            out = self.model.sample(segment, self.hparams.ddpm.infer_step)
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        return self.tensor2numpy(out), metrics

if __name__ == '__main__':
    # import soundfile as sf
    # testee = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/0.5/nuwave_x2_02_22_02_epoch=121.ckpt", device="cuda")
    # test_name = "nuwave_24k"
    testee = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/random_cutoff_fft/nuwave_random_fft_filter_02_22_15_epoch=91.ckpt", device="cuda")
    test_name = "nuwave_random_cutoff_fft"
    
    sr_eval = SR_Eval(testee, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      sr=48000,
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
    
    sr_eval.evaluate(limit_test_nums=2, limit_speaker=2)
    
    



