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
    limit_test_nums=2
    limit_speaker=-1
    
    testee = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/16/nuwave_x16_02_28_13_epoch=485.ckpt", device="cuda")
    test_name = "nuwave_16"
    
    sr_eval = SR_Eval(testee, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=48000,
                      model_output_sr=48000,
                      evaluationset_sr=44100,
                    #     setting_lowpass_filtering = {
                    #       "filter":["cheby","butter"],
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                    #       "filter_order": [3,6,9]
                    #   }, 
                    #   setting_subsampling = {
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                    #   }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 3000, 4000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                    #   setting_mp3_compression = {
                    #       "original_low_kbps": [32, 48, 64, 96, 128],
                    #   } 
    )
    
    try: 
        sr_eval.evaluate(limit_test_nums=limit_test_nums, limit_speaker=limit_speaker)
    except Exception as e:
        print(e)
    
    testee = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/8/nuwave_x8_02_24_01_epoch=197.ckpt", device="cuda")
    test_name = "nuwave_8"
    
    sr_eval = SR_Eval(testee, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=48000,
                      model_output_sr=48000,
                      evaluationset_sr=44100,
                    #     setting_lowpass_filtering = {
                    #       "filter":["cheby","butter"],
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                    #       "filter_order": [3,6,9]
                    #   }, 
                    #   setting_subsampling = {
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                    #   }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 6000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                    #   setting_mp3_compression = {
                    #       "original_low_kbps": [32, 48, 64, 96, 128],
                    #   } 
    )
    
    try: 
        sr_eval.evaluate(limit_test_nums=limit_test_nums, limit_speaker=limit_speaker)
    except Exception as e:
        print(e)
        
        
        
    testee_ema = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/4/nuwave_x4_02_24_01_epoch=307.ckpt", device="cuda")
    test_name = "nuwave_4"
    
    sr_eval = SR_Eval(testee_ema, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=48000,
                      model_output_sr=48000,
                      evaluationset_sr=44100,
                    #     setting_lowpass_filtering = {
                    #       "filter":["cheby","butter"],
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                    #       "filter_order": [3,6,9]
                    #   }, 
                    #   setting_subsampling = {
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                    #   }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                    #   setting_mp3_compression = {
                    #       "original_low_kbps": [32, 48, 64, 96, 128],
                    #   } 
    )
    
    try: 
        sr_eval.evaluate(limit_test_nums=limit_test_nums, limit_speaker=limit_speaker)
    except Exception as e:
        print(e)
        
        
    testee_rand_ema = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/random_cutoff_fft/nuwave_random_fft_filter_02_22_15_epoch=126_EMA", device="cuda")
    test_name = "nuwave_random_cutoff_fft_ema"
    
    sr_eval = SR_Eval(testee_rand_ema, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=48000,
                      model_output_sr=48000,
                      evaluationset_sr=44100,
                    #     setting_lowpass_filtering = {
                    #       "filter":["cheby","butter"],
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                    #       "filter_order": [3,6,9]
                    #   }, 
                    #   setting_subsampling = {
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                    #   }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                    #   setting_mp3_compression = {
                    #       "original_low_kbps": [32, 48, 64, 96, 128],
                    #   } 
    )
    
    try: 
        sr_eval.evaluate(limit_test_nums=limit_test_nums, limit_speaker=limit_speaker)
    except Exception as e:
        print(e)
    
    testee_rand_ema = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/0.5/nuwave_x2_02_22_02_epoch=121.ckpt", device="cuda")
    test_name = "nuwave_2"
    
    sr_eval = SR_Eval(testee_rand_ema, 
                      test_name=test_name, 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=48000,
                      model_output_sr=48000,
                      evaluationset_sr=44100,
                    #     setting_lowpass_filtering = {
                    #       "filter":["cheby","butter"],
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                    #       "filter_order": [3,6,9]
                    #   }, 
                    #   setting_subsampling = {
                    #       "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                    #   }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                    #   setting_mp3_compression = {
                    #       "original_low_kbps": [32, 48, 64, 96, 128],
                    #   } 
    )
    
    try: 
        sr_eval.evaluate(limit_test_nums=limit_test_nums, limit_speaker=limit_speaker)
    except Exception as e:
        print(e)
    
    # testee_rand = NuWaveTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/nuwave/ckpt/random_cutoff_fft/nuwave_random_fft_filter_02_22_15_epoch=91.ckpt", device="cuda")
    # test_name = "nuwave_random_cutoff_fft"
    
    # import soundfile as sf
    # path = "wukong3_17_orig.wav"
    # audio,_ = librosa.load(path, sr=44100)
    # res, _ = testee_rand.infer(audio, audio)
    # sf.write("temp.wav", res, 44100)



