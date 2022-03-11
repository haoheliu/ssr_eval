import sys
sys.path.append("/vol/research/dcase2022/sr_eval_vctk")
sys.path.append("/vol/research/dcase2022/sr_eval_vctk/testees")

import time
import librosa
import torch
from e2e_models.models.gsr_voicefixer_unet import SSR_DNN as Model

from omegaconf import OmegaConf as OC
import numpy as np 
from utils import *
from sr_eval_vctk import SR_Eval, BasicTestee

model = None
EPS=1e-9

class SR_DNN_Rand_Testee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
    def pre(self, input):
        input = input[None, ...].to(self.device)
        sp, _, _ = self.model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            sp, mel_noisy = self.pre(segment)
            out = self.model(sp, segment[None,...])
            out = out['wav']
            # import ipdb; ipdb.set_trace()
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        return self.tensor2numpy(out), metrics

if __name__ == '__main__':
    
    # import soundfile as sf
    
    for test_name  in ["SR_DNN_Rand_Testee"]:
    
        testee = eval(test_name)(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/e2e_models/ckpt/dnn/epoch=17-step=12671-val_l=0.00.ckpt",
                                model_name="unet", 
                                device="cuda")
        
        sr_eval = SR_Eval(testee, 
                        test_name=test_name, 
                        test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                        model_input_sr=44100,
                        model_output_sr=44100,
                        evaluationset_sr=44100,
                        setting_lowpass_filtering = {
                            "filter":["cheby","butter"],
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                            "filter_order": [3,6,9]
                        }, 
                        setting_subsampling = {
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        }, 
                        setting_fft = {
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        }, 
                        setting_mp3_compression = {
                            "original_low_kbps": [32, 48, 64, 96, 128],
                        },
                        save_processed_result=True,
        )
        
        sr_eval.evaluate(limit_test_nums=10, limit_speaker=-1)
    
    

    