import librosa
import torch
from lightning_model import NuWave
from omegaconf import OmegaConf as OC
import numpy as np  
model = None

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

class NuWaveTestee:
    def __init__(self, ckpt, device) -> None:
        self.hparams = OC.load('hparameter.yaml')
        self.model = NuWave(self.hparams, False).to(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=torch.device(device))['state_dict'])
        self.model.eval()
        self.model.freeze()
        self.device = device
        self.model = self.model.to(self.device)
        
    def infer(self, x, target):
        metrics = {}
        with torch.no_grad():
            segment = torch.Tensor(x).to(self.device)[None,...]
            out = self.model.sample(segment, self.hparams.ddpm.infer_step)
            out, _ = trim_center(out, segment)
        return out.squeeze().numpy(), metrics

if __name__ == '__main__':
    import soundfile as sf
    wav = "/Users/liuhaohe/Downloads/p232_001.wav"
    wav_lr = "/Users/liuhaohe/Downloads/p232_001_lr.wav"
    wav,_ = librosa.load(wav, sr=48000)
    wav_lr,_ = librosa.load(wav_lr, sr=48000)
    testee = NuWaveTestee(ckpt="/Users/liuhaohe/Downloads/nuwave_x2_02_22_02_epoch=121.ckpt", device="cuda")
    res,_ = testee.infer(wav, wav_lr)
    import ipdb; ipdb.set_trace()
    sf.write("temp.wav", res, 48000)
    print(res.shape, wav.shape)
    
    



