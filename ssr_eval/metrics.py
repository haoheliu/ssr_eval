# import git
# git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
# import sys
# sys.path.append(git_root)

import librosa
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ssr_eval.utils import *

EPS = 1e-12


class AudioMetrics:
    def __init__(self, rate):
        self.rate = rate
        self.hop_length = int(rate / 100)
        self.n_fft = int(2048 / (44100 / rate))

    def read(self, est, target):
        est, _ = librosa.load(est, sr=self.rate, mono=True)
        target, _ = librosa.load(target, sr=self.rate, mono=True)
        return est, target

    def wav_to_spectrogram(self, wav):
        f = np.abs(librosa.stft(wav, hop_length=self.hop_length, n_fft=self.n_fft))
        f = np.transpose(f, (1, 0))
        f = torch.tensor(f[None, None, ...])
        return f

    def center_crop(self, x, y):
        dim = 2
        if x.size(dim) == y.size(dim):
            return x, y
        elif x.size(dim) > y.size(dim):
            offset = x.size(dim) - y.size(dim)
            start = offset // 2
            end = offset - start
            x = x[:, :, start:-end, :]
        elif x.size(dim) < y.size(dim):
            offset = y.size(dim) - x.size(dim)
            start = offset // 2
            end = offset - start
            y = y[:, :, start:-end, :]
        assert (
            offset < 10
        ), "Error: the offset %s is too large, check the code please" % (offset)
        return x, y

    def evaluation(self, est, target, file):
        """evaluate between two audio
        Args:
            est (str or np.array): _description_
            target (str or np.array): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # import time; start = time.time()
        if type(est) != type(target):
            raise ValueError(
                "The input value should either both be numpy array or strings"
            )
        if type(est) == type(""):
            est_wav, target_wav = self.read(est, target)
        else:
            assert len(list(est.shape)) == 1 and len(list(target.shape)) == 1, (
                "The input numpy array shape should be [samples,]. Got input shape %s and %s. "
                % (est.shape, target.shape)
            )
            est_wav, target_wav = est, target

        # target_spec_path = os.path.join(os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]+"_proc_%s.pt" % (self.rate))
        # if(os.path.exists(target_spec_path)):
        #     target_sp = torch.load(target_spec_path)
        # else:

        assert (
            abs(target_wav.shape[0] - est_wav.shape[0]) < 100
        ), "Error: Shape mismatch between target and estimation %s and %s" % (
            str(target_wav.shape),
            str(est_wav.shape),
        )

        min_len = min(target_wav.shape[0], est_wav.shape[0])
        target_wav, est_wav = target_wav[:min_len], est_wav[:min_len]

        target_sp = self.wav_to_spectrogram(target_wav)
        est_sp = self.wav_to_spectrogram(est_wav)

        result = {}

        # frequency domain
        result["lsd"] = self.lsd(est_sp.clone(), target_sp.clone())
        result["log_sispec"] = self.sispec(
            to_log(est_sp.clone()), to_log(target_sp.clone())
        )
        result["sispec"] = self.sispec(est_sp.clone(), target_sp.clone())
        result["ssim"] = self.ssim(est_sp.clone(), target_sp.clone())

        for key in result:
            result[key] = float(result[key])
        return result

    def lsd(self, est, target):
        lsd = torch.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
        return lsd[..., None, None]

    def sispec(self, est, target):
        # in log scale
        output, target = energy_unify(est, target)
        noise = output - target
        sp_loss = 10 * torch.log10(
            (pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
        )
        return torch.sum(sp_loss) / sp_loss.size()[0]

    def ssim(self, est, target):
        if "cuda" in str(target.device):
            target, output = target.detach().cpu().numpy(), est.detach().cpu().numpy()
        else:
            target, output = target.numpy(), est.numpy()
        res = np.zeros([output.shape[0], output.shape[1]])
        for bs in range(output.shape[0]):
            for c in range(output.shape[1]):
                res[bs, c] = ssim(output[bs, c, ...], target[bs, c, ...], win_size=7)
        return torch.tensor(res)[..., None, None]


if __name__ == "__main__":
    import numpy as np

    au = AudioMetrics(rate=44100)
    # path1 = "old/out.wav"
    path1 = "eeeeee.wav"
    # path2 = "old/target.wav"
    path2 = "targete.wav"
    result = au.evaluation(path2, path1, path1)
    print(result)
