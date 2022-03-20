import sys
sys.path.append("/vol/research/dcase2022/project/ssr_eval")
sys.path.append("/vol/research/dcase2022/project/ssr_eval/examples")
import time
import librosa
import torch
import os
from examples.NVSR.nvsr_unet import NVSR as Model
import numpy as np 
from ssr_eval import SR_Eval, BasicTestee

torch.manual_seed(234)
EPS=1e-9

def to_log(input):
    assert torch.sum(input < 0) == 0, str(input)+" has negative values counts "+str(torch.sum(input < 0))
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input
    
def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if (est.shape[-1] == ref.shape[-1]):
        return est, ref
    elif (est.shape[-1] > ref.shape[-1]):
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2):-int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2):-int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class NVSRBaseTestee(BasicTestee):
    def __init__(self, device) -> None:
        self.model_name = "unet"
        self.ckpt = os.path.join(os.path.expanduser('~'),".cache/ssr_eval/NVSR/epoch=11-step=22499-val_l=0.27.pth")
        self.download_pretrained()
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
    def download_pretrained(self):
        import urllib.request
        if (not os.path.exists(self.ckpt)):
            os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
            print("Downloading the weight of pretrained speech super resolution baseline model NVSR")
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6370601/files/epoch%3D11-step%3D22499-val_l%3D0.27.pth?download=1",
                self.ckpt
            )
            print('Weights downloaded in: {} Size: {}'.format(self.ckpt, os.path.getsize(self.ckpt)))
    
    def pre(self, input):
        input = input[None, ...].to(self.device)
        sp, _, _ = self.model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig
    
    def find_cutoff(self, x, threshold=0.95):
        threshold = x[-1] * threshold
        for i in range(1, x.shape[0]):
            if(x[-i] < threshold): 
                return x.shape[0]-i 
        return 0
    
    def get_cutoff_index(self, x):
        stft_x = np.abs(librosa.stft(x))
        energy = np.cumsum(np.sum(stft_x,axis=-1))
        return self.find_cutoff(energy, 0.97), stft_x
    
    def replace_lr(self, x, out):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        stft_gt =librosa.stft(x)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed

    def infer(self, x):
        return x
    
class NVSRTestee(NVSRBaseTestee):
    def __init__(self, device) -> None:
        super(NVSRTestee, self).__init__(device)

    def infer(self, x):
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        return self.tensor2numpy(out)
    
class NVSRPostProcTestee(NVSRBaseTestee):
    def __init__(self, device) -> None:
        super(NVSRPostProcTestee, self).__init__(device)
        
    def infer(self, x):
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out)
        return out
  
class NVSRPaddingPostProcTestee(NVSRBaseTestee):
    def __init__(self, device) -> None:
        super(NVSRPaddingPostProcTestee, self).__init__(device)
        
    def get_cutoff_index_v2(self, x):
        energy = np.cumsum(np.sum(x,axis=-1))
        return self.find_cutoff(energy, 0.97)
    
    def add_segment_to_higher_freq(self, mel_lr):
        # mel_lr: [128, t-steps]
        size = mel_lr.size()
        mel_lr=mel_lr.squeeze().transpose(0,1).cpu().numpy()
        cutoffratio = self.get_cutoff_index_v2(mel_lr)        
        avg_energy = np.tile(mel_lr[cutoffratio,:], (mel_lr.shape[0],1))
        mel_lr[cutoffratio:,...] = 0
        avg_energy[:cutoffratio,...] = 0  
        mel_lr = mel_lr + avg_energy
        mel_lr = torch.Tensor(mel_lr.copy()).transpose(0,1)[None,None,...].to(self.device)
        assert size == mel_lr.size()
        return mel_lr
    
    def infer(self, x):
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            denoised_mel = self.add_segment_to_higher_freq(mel_noisy)
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out)
        return out    
  
if __name__ == '__main__':
    import soundfile as sf
    for test_name  in ["NVSRPostProcTestee"]:
        testee = eval(test_name)(device="cuda")
        sr_eval = SR_Eval(testee, 
                        test_name=test_name, 
                        test_data_root="/vol/research/dcase2022/project/ssr_eval/vctk_test", 
                        model_input_sr=44100,
                        model_output_sr=44100,
                        evaluationset_sr=44100,
                        setting_fft = {
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        }, 
                        save_processed_result=False,
        )
        sr_eval.evaluate(limit_test_nums=10, limit_speaker=-1)  
        
    
    

    