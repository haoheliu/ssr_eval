from nbformat import write
import torch
import numpy as np  
import os    
from metrics import AudioMetrics
from handler import * 
from utils import * 
from preprocess import preprocess

"""_summary_
test
    -p360
    -p361
    ...
"""

# TEST_ROOT = "/Users/liuhaohe/projects/sr_eval_vctk/dataset/wav48/test"
TEST_ROOT = "/blob/v-haoheliu/datasets/vctk_test"
SAMPLE_RATE=44100
audio_metrics = AudioMetrics(SAMPLE_RATE)

def dict_mean(dict_list):
    ret_val = {}
    for k in dict_list[0].keys():
        ret_val[k] = np.mean([v[k] for v in dict_list])
    return ret_val

def evaluate(file):
    metrics = {}
    processed_low_res_input = preprocess(file, sr=SAMPLE_RATE)
    for k in processed_low_res_input.keys():
        original, processed = handler_same(processed_low_res_input[k])
        # processed_low_res_input[k] = (processed_low_res_input[k], processed)
        metrics[k] = audio_metrics.evaluation(original, processed, file)
    return metrics

def main(test_name = "test", test_run=False):
    from tqdm import tqdm
    from datetime import datetime
    final_result = {}
    for speaker in os.listdir(TEST_ROOT):
        print("Speaker:", speaker)
        final_result[speaker] = {}
        for i, file in enumerate(tqdm(os.listdir(os.path.join(TEST_ROOT, speaker)))):
            if(file[-4:]!=".wav" and file[-5:]!=".flac"): # Other files
                continue
            if("DS_Store" in file): # MacOS files
                continue
            if("proc" in file): # Cache files
                continue
            if(test_run): 
                if(i > 10): break
            audio_path = os.path.join(TEST_ROOT, speaker, file)
            final_result[speaker][file] = evaluate(audio_path)
    os.makedirs("outputs", exist_ok=True)
    result_cache = {}
    for speaker in final_result.keys():        
        result_cache[speaker] = {}
        for file in final_result[speaker].keys(): distortion_type = list(final_result[speaker][file].keys()); break  
        for distortion in distortion_type:
            result_cache[speaker][distortion] = [v[distortion] for k,v in final_result[speaker].items()]
            result_cache[speaker][distortion] = dict_mean(result_cache[speaker][distortion])
    averaged_result = {}
    for distortion in distortion_type:
        averaged_result[distortion] = dict_mean([result_cache[speaker][distortion] for speaker in final_result.keys()])
    final_result['each_speaker'] = result_cache
    final_result['averaged'] = averaged_result
    now = datetime.now()
    save_path = str(str(now.date())+"-"+str(now.time()))+"-"+test_name+".json"
    write_json(final_result, os.path.join("outputs", save_path))
    return final_result

if __name__ == "__main__":
    main(test_run=True)
            