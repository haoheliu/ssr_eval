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

torch.manual_seed(0)

model = None
EPS=1e-9

def statistic_mel_energy():
    # Calculate mel energy average
    import os  
    from tqdm import tqdm
    from glob import glob
    testee = VoiceFixerTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/voicefixer_local/ckpt/unet_butter_2k_44k/epoch=11-step=22499-val_l=0.27.ckpt",
                            model_name="unet", 
                            device="cuda")
    path = "/vol/research/dcase2022/sr_eval_vctk/vctk_test/"
    res =  []
    for file in tqdm(glob(os.path.join(path,"*/*.flac"))):
        file = os.path.join(path, file)
        audio,_ = librosa.load(file, sr=44100)
        audio = torch.Tensor(audio.copy()).to("cuda")[None,...]
        _,mel = testee.pre(audio)
        mel = mel.squeeze()
        step_energy = torch.sum(mel, dim=1)
        step_energy_threshold = torch.mean(step_energy)
        mask = step_energy > step_energy_threshold
        mel_energy = torch.mean(mel[mask], dim=0)
        res.append(mel_energy)
    res_s = torch.stack(res)
    mean_res = torch.mean(res_s, dim=0).cpu().numpy()
    np.save("mel_energy_avg.npy", mean_res)


def statistic_stft_energy():
    # Calculate mel energy average
    import os  
    from tqdm import tqdm
    from glob import glob
    testee = VoiceFixerTestee(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/voicefixer_local/ckpt/unet_butter_2k_44k/epoch=11-step=22499-val_l=0.27.ckpt",
                            model_name="unet", 
                            device="cuda")
    path = "/vol/research/dcase2022/sr_eval_vctk/vctk_test/"
    res =  []
    for file in tqdm(glob(os.path.join(path,"*/*.flac"))):
        file = os.path.join(path, file)
        audio,_ = librosa.load(file, sr=44100)
        audio = torch.Tensor(audio.copy()).to("cuda")[None,...]
        mel,mel_real = testee.pre(audio)
        print(mel.size(),mel_real.size())
        mel = mel.squeeze()
        step_energy = torch.sum(mel, dim=1)
        step_energy_threshold = torch.mean(step_energy)
        mask = step_energy > step_energy_threshold
        mel_energy = torch.mean(mel[mask], dim=0)
        res.append(mel_energy)
    res_s = torch.stack(res)
    mean_res = torch.mean(res_s, dim=0).cpu().numpy()
    np.save("stft_energy_avg.npy", mean_res)

class VoiceFixerTestee(BasicTestee):
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
            _, mel_noisy = self.pre(segment)
            out = self.model(mel_noisy)
            denoised_mel = from_log(out['mel'])
            out = self.model.vocoder(denoised_mel, cuda=True)
            # import ipdb; ipdb.set_trace()
            out, _ = trim_center(out, segment)
        out = out.squeeze()
        return self.tensor2numpy(out), metrics
    
class ArtificialHRReplaceTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.current = time.time()
        self.avg_stft_energy = np.load("stft_energy_avg.npy")
        
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
        # print(mel_lr.shape)
        # mel_lr=mel_lr.squeeze().transpose(0,1).cpu().numpy()
        cutoffratio = self.get_cutoff_index_v2(mel_lr)        
        energyratio = self.get_energy_distribution(mel_lr)[None,...]
        avg_energy = np.tile(mel_lr[cutoffratio,:], (mel_lr.shape[0],1))
        # avg_energy = np.tile(self.avg_stft_energy, (mel_lr.shape[1],1)).T
        # avg_energy = avg_energy * (enesrgyratio * 2)
        mel_lr[cutoffratio:,...] = 0
        avg_energy[:cutoffratio,...] = 0  
        mel_lr = mel_lr + avg_energy
        # import matplotlib.pyplot as plt
        # import librosa.display
        # librosa.display.specshow(np.log(mel_lr))
        # plt.savefig("temp.png")
        # import ipdb; ipdb.set_trace()
        return mel_lr
    
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            sp = librosa.stft(x)
            denoised_sp = self.add_segment_to_higher_freq(sp)
            out = librosa.istft(denoised_sp, length=x.shape[0])
            out, _ = trim_center(out, x)
            out = np.squeeze(out)
            out = self.replace_lr(x, out, target)
        return out, metrics    
    
class VoiceFixerArtificialHRReplaceTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device, lamb = 1.0) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
        self.lamb = lamb
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
        energyratio = self.get_energy_distribution(mel_lr)[None,...]
        avg_energy = np.tile(self.avg_mel_energy, (mel_lr.shape[1],1)).T
        avg_energy = avg_energy * (energyratio * self.lamb)
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


class VoiceFixerUnprocessedMelReplaceTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
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
    
class VoiceFixerCopyFreqHRReplaceTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
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

class VoiceFixerCopyFreqHRTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
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
        import matplotlib.pyplot as plt
        import librosa.display
        librosa.display.specshow(np.log(mel_lr))
        plt.savefig("temp.png")
        
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
            # out = self.replace_lr(x, out, target)
        return out, metrics   

class VoiceFixerArtificialHRTestee(BasicTestee):
    def __init__(self, ckpt, model_name, device) -> None:
        self.model_name = model_name
        self.model = Model(channels=1).load_from_checkpoint(ckpt)
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
        import matplotlib.pyplot as plt 
        
        size = mel_lr.size()
        mel_lr=mel_lr.squeeze().transpose(0,1).cpu().numpy()
        np.save("1mel.npy", mel_lr)
        cutoffratio = self.get_cutoff_index_v2(mel_lr)        
        energyratio = self.get_energy_distribution(mel_lr)[None,...]
        np.save("2energyratio.npy",energyratio)
        avg_energy = np.tile(self.avg_mel_energy, (mel_lr.shape[1],1)).T
        np.save("3avg_energy.npy",avg_energy)
        avg_energy = avg_energy * energyratio
        np.save("4avg_energy_weighted.npy",avg_energy)
        mel_lr[cutoffratio:,...] = 0
        avg_energy[:cutoffratio,...] = 0  
        mask = np.ones_like(mel_lr)
        mask[cutoffratio:,...] *= 0
        np.save("5cutoff.npy", mask)
        np.save("5avg_cutted.npy", avg_energy)
        np.save("6mel_lr.npy", mel_lr)
        mel_lr = mel_lr + avg_energy
        np.save("7mel_final.npy",mel_lr)
        mel_lr = torch.Tensor(mel_lr.copy()).transpose(0,1)[None,None,...].to(self.device)
        assert size == mel_lr.size()
        exit(0)
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
        return out, metrics

class VoiceFixerReplaceLRRescaleTestee(BasicTestee):
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
        rescale_ratio = np.mean(stft_gt[:cutoffratio,...]) / np.mean(stft_out[:cutoffratio,...])
        stft_out *= rescale_ratio
        stft_out[:cutoffratio,...] = stft_gt[:cutoffratio,...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed
    
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
    
class VoiceFixerReplaceLRTestee(BasicTestee):
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

class OracleVoiceFixerReplaceLRTestee(BasicTestee):
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

class OracleVoiceFixerTestee(BasicTestee):
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
    
    for test_name  in ["VoiceFixerReplaceLRRescaleTestee"]:
        testee = eval(test_name)(ckpt="/vol/research/dcase2022/sr_eval_vctk/testees/voicefixer_local/ckpt/unet_butter_2k_44k/epoch=11-step=22499-val_l=0.27.ckpt",
                                model_name="unet", 
                                device="cuda",
                                )
        sr_eval = SR_Eval(testee, 
                        test_name=test_name, 
                        test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                        model_input_sr=44100,
                        model_output_sr=44100,
                        evaluationset_sr=44100,
                        # setting_lowpass_filtering = {
                        #     "filter":["cheby","butter"],
                        #     "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        #     "filter_order": [3,6,9]
                        # }, 
                        # setting_subsampling = {
                        #     "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        # }, 
                        setting_fft = {
                            "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                        }, 
                        # setting_mp3_compression = {
                        #     "original_low_kbps": [32, 48, 64, 96, 128],
                        # },
                        save_processed_result=False,
        )
        
        sr_eval.evaluate(limit_test_nums=3, limit_speaker=-1)
    
    

    