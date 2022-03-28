import numpy as np
import os
import librosa
import soundfile as sf
from scipy.signal import correlate

from ssr_eval.metrics import AudioMetrics
from ssr_eval.utils import *
from ssr_eval.lowpass import lowpass

# ----------------------------------------------------------------
# README:
# Define several functions for data generation.
# ----------------------------------------------------------------


class BasicTestee:
    def __init__(self) -> None:
        pass

    def _find_cutoff(self, x, threshold=0.95):
        threshold = x[-1] * threshold
        for i in range(1, x.shape[0]):
            if x[-i] < threshold:
                return x.shape[0] - i
        return 0

    def _get_cutoff_index(self, x):
        stft_x = np.abs(librosa.stft(x))
        energy = np.cumsum(np.sum(stft_x, axis=-1))
        return self._find_cutoff(energy, 0.97)

    def postprocessing(self, x, out):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio = self._get_cutoff_index(x)
        stft_gt = librosa.stft(x)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio, ...] = stft_gt[:cutoffratio, ...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed

    def tensor2numpy(self, tensor):
        if "cuda" in str(tensor.device):
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()

    def infer(self, x):
        # x: [sample,]
        # return: [sample, sample]
        return x


"""_summary_
test
    -p360
    -p361
    ...
"""


class SSR_Eval_Helper:
    def __init__(
        self,
        testee,
        input_sr,
        output_sr,
        evaluation_sr=44100,
        test_name="test",
        test_data_root="./datasets/vctk_test",
        setting_lowpass_filtering=None,
        setting_subsampling=None,
        setting_fft=None,
        setting_mp3_compression=None,
        save_processed_result=False,
    ):

        self.testee = testee
        self.test_name = test_name
        self.test_data_root = test_data_root
        self.save_processed_result = save_processed_result

        self.setting_lowpass_filtering = self._cutoff2sr(setting_lowpass_filtering)
        self.setting_fft = self._cutoff2sr(setting_fft)
        self.setting_subsampling = self._cutoff2sr(setting_subsampling)
        self.setting_mp3_compression = setting_mp3_compression

        self.model_input_sr = input_sr
        self.model_output_sr = output_sr
        self.evaluationset_sr = evaluation_sr

        assert (
            self.evaluationset_sr <= 48000
        ), "Our evaluation set only support up to 48 kHz target sampling rate"

        self.audio_metrics = AudioMetrics(self.evaluationset_sr)
        self.unexpected_symbol_test_folder = "_.*#()_+=!@$%^&~"
        if not os.path.exists(test_data_root):
            os.makedirs(test_data_root, exist_ok=True)

        if "s5" not in os.listdir(test_data_root):
            # Download the testset
            print("vctk 0.92 version is not found. Start downloading...")
            cmd = (
                "wget https://zenodo.org/record/6370601/files/vctk_test_48k.tar?download=1 -O %s"
                % (os.path.join(test_data_root, "vctk_test.tar"))
            )
            cmd2 = "tar -zxf %s -C %s" % (
                os.path.join(test_data_root, "vctk_test.tar"),
                os.path.dirname(test_data_root),
            )
            cmd3 = "rm %s" % (os.path.join(test_data_root, "vctk_test.tar"))
            print(cmd)
            os.system(cmd)
            print(cmd2)
            os.system(cmd2)
            print(cmd3)
            os.system(cmd3)

    def _cutoff2sr(self, dic):
        if dic is None:
            return None
        else:
            dic["cutoff_freq"] = [x * 2 for x in dic["cutoff_freq"]]
        return dic

    def evaluate_single(self, file):
        metrics = {}
        processed_low_res_input = self.preprocess(file, sr=self.model_input_sr)

        # Sox resample method is the best
        os.system("sox %s -r %s %s" % (file, self.evaluationset_sr, "temp.wav"))
        target, _ = librosa.load("temp.wav", sr=None)

        for k in processed_low_res_input.keys():
            result_fname = file + k + "_processed_" + self.test_name + ".wav"
            ret = self.testee.infer(processed_low_res_input[k])
            if type(ret) == tuple:
                processed, addtional_metrics = ret
            else:
                processed = ret
                addtional_metrics = {}
            if self.model_output_sr != self.evaluationset_sr:
                processed = librosa.resample(
                    processed,
                    self.model_output_sr,
                    self.evaluationset_sr,
                    res_type="polyphase",
                )
            metrics[k] = self.audio_metrics.evaluation(processed, target, file)
            metrics[k].update(addtional_metrics)
            if self.save_processed_result:
                sf.write(result_fname, processed, self.evaluationset_sr)
        # print(file, metrics)
        return metrics

    def get_test_file_list(self, path):
        ret = []
        for file in os.listdir(path):
            if file[-4:] != ".wav" and file[-5:] != ".flac":
                continue
            elif "DS_Store" in file:
                continue
            elif "proc" in file:
                continue
            else:
                ret.append(file)
        return ret

    def evaluate(self, limit_test_nums=-1, limit_speaker=-1):
        from tqdm import tqdm
        from datetime import datetime

        final_result = {}
        result_cache = {}
        averaged_result = {}
        os.makedirs("results", exist_ok=True)

        for speaker in sorted(os.listdir(self.test_data_root)):
            if not os.path.isdir(os.path.join(self.test_data_root, speaker)):
                continue  # MacOS files
            if "p" not in speaker and "s" not in speaker:
                continue
            if limit_speaker > 0 and len(final_result.keys()) >= limit_speaker:
                break
            print("Speaker:", speaker)
            final_result[speaker] = {}
            files = sorted(
                self.get_test_file_list(os.path.join(self.test_data_root, speaker))
            )
            assert len(files) != 0, os.path.join(self.test_data_root, speaker)
            for i, file in enumerate(tqdm(files)):
                if limit_test_nums > 0:
                    if i >= limit_test_nums:
                        break
                audio_path = os.path.join(self.test_data_root, speaker, file)
                final_result[speaker][file] = self.evaluate_single(audio_path)
        # import ipdb; ipdb.set_trace()
        for speaker in final_result.keys():
            result_cache[speaker] = {}
            for file in final_result[speaker].keys():
                distortion_type = list(final_result[speaker][file].keys())
                break
            for distortion in distortion_type:
                result_cache[speaker][distortion] = [
                    v[distortion] for k, v in final_result[speaker].items()
                ]
                result_cache[speaker][distortion] = dict_mean(
                    result_cache[speaker][distortion]
                )

        for distortion in distortion_type:
            averaged_result[distortion] = dict_mean(
                [result_cache[speaker][distortion] for speaker in final_result.keys()]
            )
        final_result["each_speaker"] = result_cache
        final_result["averaged"] = averaged_result
        now = datetime.now()
        save_path = (
            str(str(now.date()) + "-" + str(now.time()))
            + "-"
            + self.test_name
            + ".json"
        )
        write_json(final_result, os.path.join("results", save_path))
        return final_result

    def preprocess(self, file, sr):
        # Add various effect on the samples
        """Add various effect on the audio data

        Args:
            file (_type_): str, path to the audio file
            sr (_type_): sample rate

        Returns:
            _type_: dict
        """

        ret_dict = {}
        x, _ = librosa.load(file, sr=sr)
        if (
            self.setting_lowpass_filtering is not None
            and "butter" in self.setting_lowpass_filtering["filter"]
        ):
            ret_dict.update(self.lowpass_butterworth(file, x, sr))
        if (
            self.setting_lowpass_filtering is not None
            and "cheby" in self.setting_lowpass_filtering["filter"]
        ):
            ret_dict.update(self.lowpass_chebyshev(file, x, sr))
        if (
            self.setting_lowpass_filtering is not None
            and "ellip" in self.setting_lowpass_filtering["filter"]
        ):
            ret_dict.update(self.lowpass_ellip(file, x, sr))
        if (
            self.setting_lowpass_filtering is not None
            and "bessel" in self.setting_lowpass_filtering["filter"]
        ):
            ret_dict.update(self.lowpass_bessel(file, x, sr))

        if self.setting_subsampling is not None:
            ret_dict.update(self.lowpass_subsampling(file, x, sr))
        if self.setting_mp3_compression is not None:
            ret_dict.update(self.mp3_encoding(file, x, sr))
        if self.setting_fft is not None:
            ret_dict.update(self.lowpass_stft_hard(file, x, sr))
        return ret_dict

    def shift(self, x, shift):
        ret = np.zeros_like(x)
        if shift >= 0:
            ret[:-shift] = x[shift:]
        elif shift < 0:
            ret[-shift:] = x[:-(-shift)]
        return ret

    def pad(self, x, y):
        if x.shape[0] == y.shape[0]:
            return x, y
        elif x.shape[0] > y.shape[0]:
            cache_y = np.zeros_like(x)
            cache_y[: y.shape[0]] = y
            return x, cache_y
        else:
            cache_x = np.zeros_like(y)
            cache_x[: x.shape[0]] = x
            return cache_x, y

    def unify_length(self, x, target):
        if x.shape[0] == target.shape[0]:
            return x, target
        elif x.shape[0] > target.shape[0]:
            return x[: target.shape[0]], target
        else:
            cache_x = np.zeros_like(target)
            cache_x[: x.shape[0]] = x
            return cache_x, target

    def mp3_encoding(self, file, x, sr):
        ret_dict = {}
        for low_kbps in self.setting_mp3_compression["low_kbps"]:
            key = "proc_mp3_%s_%s" % (low_kbps, sr)
            temp_file = self.cache_file_name("temp", file)
            target_file = self.cache_file_name(key, file)
            target_mp3_file = self.cache_file_name(key, file, suffix=".mp3")
            cmd1 = "sox %s -C %s %s" % (file, low_kbps, target_mp3_file)
            cmd2 = "sox %s %s" % (target_mp3_file, temp_file)
            cmd3 = "rm %s" % (target_mp3_file)
            cmd4 = "rm %s" % (temp_file)
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)
            ret_dict[key], _ = librosa.load(temp_file, sr=sr)
            os.system(cmd4)
            ret_dict[key], x = self.unify_length(ret_dict[key], x)
            shft01 = np.argmax(correlate(ret_dict[key], x)) - x.shape[0]
            shifted = self.shift(ret_dict[key], shft01)
            sf.write(target_file, shifted[..., None], samplerate=sr)
            ret_dict[key] = shifted
            assert ret_dict[key].shape == x.shape, str((ret_dict[key].shape, x.shape))
            assert np.sum(ret_dict[key] - x) != 0.0
        return ret_dict

    def cache_file_name(self, key, file, suffix=".flac"):
        # import ipdb; ipdb.set_trace()
        return os.path.join(
            os.path.dirname(file),
            os.path.splitext(os.path.basename(file))[0] + "_" + key + suffix,
        )

    def lowpass_butterworth(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_lowpass_filtering["cutoff_freq"]:
            for order in self.setting_lowpass_filtering["filter_order"]:
                if low_rate == sr:
                    low_rate -= 1
                key = "proc_bw_%s_%s_%s" % (low_rate, order, sr)
                # target_file = self.cache_file_name(key, file)
                # import ipdb; ipdb.set_trace()
                print(low_rate, sr)
                ret_dict[key] = lowpass(
                    x, low_rate // 2, sr, order=order, _type="butter"
                )
                # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        for k in ret_dict.keys():
            assert ret_dict[k].shape == x.shape, str((ret_dict[k].shape, x.shape))
        return ret_dict

    def lowpass_bessel(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_lowpass_filtering["cutoff_freq"]:
            for order in self.setting_lowpass_filtering["filter_order"]:
                if low_rate == sr:
                    low_rate -= 1
                key = "proc_bessel_%s_%s_%s" % (low_rate, order, sr)
                # target_file = self.cache_file_name(key, file)
                ret_dict[key] = lowpass(
                    x, low_rate // 2, sr, order=order, _type="bessel"
                )
                # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        for k in ret_dict.keys():
            assert ret_dict[k].shape == x.shape, str((ret_dict[k].shape, x.shape))
        return ret_dict

    def lowpass_ellip(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_lowpass_filtering["cutoff_freq"]:
            for order in self.setting_lowpass_filtering["filter_order"]:
                if low_rate == sr:
                    low_rate -= 1
                key = "proc_el_%s_%s_%s" % (low_rate, order, sr)
                # target_file = self.cache_file_name(key, file)
                ret_dict[key] = lowpass(
                    x, low_rate // 2, sr, order=order, _type="ellip"
                )
                # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        for k in ret_dict.keys():
            assert ret_dict[k].shape == x.shape, str((ret_dict[k].shape, x.shape))
        return ret_dict

    def lowpass_chebyshev(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_lowpass_filtering["cutoff_freq"]:
            for order in self.setting_lowpass_filtering["filter_order"]:
                if low_rate == sr:
                    low_rate -= 1
                key = "proc_ch_%s_%s_%s" % (low_rate, order, sr)
                # target_file = self.cache_file_name(key, file)
                # import ipdb; ipdb.set_trace()
                ret_dict[key] = lowpass(
                    x, low_rate // 2, sr, order=order, _type="cheby1"
                )
                # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        for k in ret_dict.keys():
            assert ret_dict[k].shape == x.shape, str((ret_dict[k].shape, x.shape))
        return ret_dict

    def lowpass_stft_hard(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_fft["cutoff_freq"]:
            if low_rate == sr:
                low_rate -= 1
            key = "proc_fft_%s_%s" % (low_rate, sr)
            # target_file = self.cache_file_name(key, file)
            ret_dict[key] = lowpass(x, low_rate // 2, sr, order=1, _type="stft_hard")
            # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        return ret_dict

    def lowpass_subsampling(self, file, x, sr):
        ret_dict = {}
        for low_rate in self.setting_subsampling["cutoff_freq"]:
            if low_rate == sr:
                low_rate -= 1
            key = "proc_subsampling_%s_%s" % (low_rate, sr)
            # target_file = self.cache_file_name(key, file)
            ret_dict[key] = lowpass(x, low_rate // 2, sr, order=1, _type="subsampling")
            # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
        return ret_dict
