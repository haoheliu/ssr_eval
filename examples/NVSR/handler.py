import sys
# sys.path.append("/vol/research/dcase2022/project/sr_eval_vctk")
# sys.path.append("/vol/research/dcase2022/project/sr_eval_vctk/examples")
import time
import librosa
import torch
from examples.NVSR.nvsr_unet import NVSR as Model
import numpy as np 
from sr_eval_vctk import SR_Eval, BasicTestee

torch.manual_seed(0)
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

class NVSRTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
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
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            # import ipdb; ipdb.set_trace()
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        return self.tensor2numpy(out), metrics
    
class NVSRUnprocessedMelReplaceTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        self.avg_mel_energy = np.load("mel_energy_avg.npy")
        
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
        
    def get_cutoff_index_v2(self, x):
        energy = np.cumsum(np.sum(x,axis=-1))
        return self.find_cutoff(energy, 0.97)
    
    def get_cutoff_index(self, x):
        stft_x = np.abs(librosa.stft(x))
        energy = np.cumsum(np.sum(stft_x,axis=-1))
        return self.find_cutoff(energy, 0.97), stft_x
        
    def replace_lr(self, x, out, target):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        stft_gt =librosa.stft(target)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
    def get_energy_distribution(self, mel_x):
        # mel_lr: [128, t-steps]
        energy = np.sum(mel_x, axis=0)
        max_energy = np.max(energy)
        return energy / max_energy
    
    def add_segment_to_higher_freq(self, mel_lr):
        # mel_lr: [128, t-steps]
        size = mel_lr.size()
        mel_lr=mel_lr.squeeze().transpose(0,1).cpu().numpy()
        cutoffratio = self.get_cutoff_index_v2(mel_lr)        
        # import ipdb; ipdb.set_trace()
        avg_energy = np.tile(mel_lr[cutoffratio,:], (mel_lr.shape[0],1))
        mel_lr[cutoffratio:,...] = 0
        avg_energy[:cutoffratio,...] = 0  
        mel_lr = mel_lr + avg_energy
        
        mel_lr = torch.Tensor(mel_lr.copy()).transpose(0,1)[None,None,...].to(self.device)
        assert size == mel_lr.size()
        return mel_lr
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            denoised_mel = mel_noisy
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out, target)
        return out, metrics    
    
class NVSRPaddingPostProcTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
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
        
    def get_cutoff_index_v2(self, x):
        energy = np.cumsum(np.sum(x,axis=-1))
        return self.find_cutoff(energy, 0.97)
    
    def get_cutoff_index(self, x):
        stft_x = np.abs(librosa.stft(x))
        energy = np.cumsum(np.sum(stft_x,axis=-1))
        return self.find_cutoff(energy, 0.97), stft_x
        
    def replace_lr(self, x, out, target):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        stft_gt =librosa.stft(target)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
    def get_energy_distribution(self, mel_x):
        # mel_lr: [128, t-steps]
        energy = np.sum(mel_x, axis=0)
        max_energy = np.max(energy)
        return energy / max_energy
    
    def add_segment_to_higher_freq(self, mel_lr):
        # mel_lr: [128, t-steps]
        size = mel_lr.size()
        mel_lr=mel_lr.squeeze().transpose(0,1).cpu().numpy()
        cutoffratio = self.get_cutoff_index_v2(mel_lr)        
        # import ipdb; ipdb.set_trace()
        avg_energy = np.tile(mel_lr[cutoffratio,:], (mel_lr.shape[0],1))
        # todo
        avg_energy = avg_energy + np.random.normal(*avg_energy.shape) * avg_energy
        
        mel_lr[cutoffratio:,...] = 0
        avg_energy[:cutoffratio,...] = 0  
        mel_lr = mel_lr + avg_energy
        
        mel_lr = torch.Tensor(mel_lr.copy()).transpose(0,1)[None,None,...].to(self.device)
        assert size == mel_lr.size()
        return mel_lr
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            denoised_mel = self.add_segment_to_higher_freq(mel_noisy)
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out, target)
        return out, metrics    

class NVSRPostProcTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        self.sr = 44100
        
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
        
    def replace_lr(self, x, out, target):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        # print(target.shape, out.shape)
        stft_gt =librosa.stft(target)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
    def proc(self, fname, save_fname):
        x,_ = librosa.load(fname, sr=self.sr)
        res,_ = self.infer(x, x)
        sf.write(save_fname, res, samplerate=44100)
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
        
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out, target)
        return out, metrics

class GroudTruthMelNVSRPostProcTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
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
        
    def replace_lr(self, x, out, target):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        stft_gt =librosa.stft(target)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(target.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            out = self.replace_lr(x, out, target)
        return out, metrics

class GroudTruthMelNVSRTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        
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
        
    def replace_lr(self, x, out, target):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio, stft_x = self.get_cutoff_index(x)
        stft_gt =librosa.stft(target)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(target.copy()).to(self.device)[None,...]
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            out, _ = trim_center(out, segment)
            out = self.tensor2numpy(out)
            out = np.squeeze(out)
            # out = self.replace_lr(x, out, target)
        return out, metrics

if __name__ == '__main__':

    import soundfile as sf
    
    for test_name  in ["NVSRPostProcTestee"]:
        testee = eval(test_name)(ckpt="/vol/research/dcase2022/project/sr_eval_vctk/examples/NVSR/ckpt/epoch=11-step=22499-val_l=0.27.pth",
                                model_name="unet", 
                                device="cuda",
                                )
        sr_eval = SR_Eval(testee, 
                        test_name=test_name, 
                        test_data_root="/vol/research/dcase2022/project/sr_eval_vctk/vctk_test", 
                        model_input_sr=44100,
                        model_output_sr=44100,
                        evaluationset_sr=44100,
                        setting_fft = {
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000], 
                        }, 
                        save_processed_result=False,
        )
        sr_eval.evaluate(limit_test_nums=10, limit_speaker=-1)  
    
    

    