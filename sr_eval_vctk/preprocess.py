import librosa
from lowpass import lowpass
import os   
import numpy as np    
import soundfile as sf  

LOW_SAMPLE_RATE=[4000,8000,12000,16000,24000,32000]
LOW_KBPS=[32, 48, 64, 96, 128]
FILTER_ORDERS=[3,6,9]

def preprocess(file, sr):
    # Add various effect on the samples
    """Add various effect on the audio data

    Args:
        file (_type_): str, path to the audio file
        sr (_type_): sample rate

    Returns:
        _type_: dict
    """
    ret_dict={}
    x, _ = librosa.load(file, sr=sr)
    ret_dict.update(lowpass_butterworth(file, x, sr))
    ret_dict.update(lowpass_chebyshev(file,x, sr))
    ret_dict.update(lowpass_stft_hard(file,x, sr))
    ret_dict.update(lowpass_subsampling(file,x, sr))
    ret_dict.update(mp3_encoding(file,x, sr))
    return ret_dict

def mp3_encoding(file, x, sr):
    ret_dict = {}
    for low_kbps in LOW_KBPS:
        key = 'proc_mp3_%s' % (low_kbps)
        target_file = cache_file_name(key, file)
        target_mp3_file = cache_file_name(key, file, suffix=".mp3")
        if(os.path.exists(target_file)): 
            ret_dict[key],_ = librosa.load(target_file, sr=sr)
        else:
            cmd1 = "sox %s -C %s %s" % (file, low_kbps, target_mp3_file)
            cmd2 = "sox %s %s" % (target_mp3_file, target_file)
            cmd3 = "rm %s" % (target_mp3_file)
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)
            ret_dict[key], _ = librosa.load(target_file, sr=sr)
            # if(ret_dict[key].shape[0] > x.shape[0]):
            #     diff = (ret_dict[key].shape[0]-x.shape[0])//2
            #     diff2 = ret_dict[key].shape[0]-x.shape[0]-diff
            #     ret_dict[key] = ret_dict[key][diff: ret_dict[key].shape[0] - diff2] # trim the first few ms
            # assert ret_dict[key].shape == x.shape, str((ret_dict[key].shape, x.shape))
            # sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
    return ret_dict

def cache_file_name(key, file, suffix=".flac"):
    # import ipdb; ipdb.set_trace()
    return os.path.join(os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]+"_"+key+suffix)

def lowpass_butterworth(file, x, sr): 
    ret_dict = {}
    for low_rate in LOW_SAMPLE_RATE:
        for order in FILTER_ORDERS:
            key = 'proc_bw_%s_%s' % (low_rate, order)
            target_file = cache_file_name(key, file)
            if(os.path.exists(target_file)): 
                ret_dict[key],_ = librosa.load(target_file, sr=sr)
            else:
                ret_dict[key] = lowpass(x, low_rate // 2, sr, order=order, _type="butter")
                sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
    return ret_dict

def lowpass_chebyshev(file, x, sr): 
    ret_dict = {}
    for low_rate in LOW_SAMPLE_RATE:
        for order in FILTER_ORDERS:
            key = 'proc_ch_%s_%s' % (low_rate, order)
            target_file = cache_file_name(key, file)
            if(os.path.exists(target_file)): 
                ret_dict[key],_ = librosa.load(target_file, sr=sr)
            else:
                ret_dict[key] = lowpass(x, low_rate // 2, sr, order=order, _type="cheby1")
                sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
    return ret_dict

def lowpass_stft_hard(file, x, sr): 
    ret_dict = {}
    for low_rate in LOW_SAMPLE_RATE:
        key = 'proc_fft_%s' % (low_rate)
        target_file = cache_file_name(key, file)
        if(os.path.exists(target_file)): 
            ret_dict[key],_ = librosa.load(target_file, sr=sr)
        else:
            ret_dict[key] = lowpass(x, low_rate // 2, sr, order=1, _type="stft_hard")
            sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
    return ret_dict

def lowpass_subsampling(file, x, sr): 
    ret_dict = {}
    for low_rate in LOW_SAMPLE_RATE:
        key = 'proc_subsampling_%s' % (low_rate)
        target_file = cache_file_name(key, file)
        if(os.path.exists(target_file)): 
            ret_dict[key],_ = librosa.load(target_file, sr=sr)
        else:
            ret_dict[key] = lowpass(x, low_rate // 2, sr, order=1, _type="subsampling")
            sf.write(target_file, ret_dict[key][...,None], samplerate=sr)
    return ret_dict
