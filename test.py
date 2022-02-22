import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")

from sr_eval_vctk import SR_Eval, BasicTestee
    

LOW_SAMPLE_RATE=[4000,8000,12000,16000,24000,32000]
LOW_KBPS=[32, 48, 64, 96, 128]
FILTER_ORDERS=[3,6,9]

class MyTestee(BasicTestee):
    def __init__(self) -> None:
        super().__init__()
        
    def infer(self, x):
        # x: np.array [sample,]
        # return: np.array [sample,]
        return x 
    
if __name__ == "__main__":
    testee = MyTestee()
    handler = SR_Eval(testee, 
                      test_name="basic_test", 
                      test_data_root="./temp_test", 
                      sr=44100,
                      setting_lowpass_filtering = {
                          "filter":["cheby","butter"],
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                          "filter_order": [3,6,9]
                      }, 
                      setting_subsampling = {
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                      }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 16000, 24000, 32000],
                      }, 
                      setting_mp3_compression = {
                          "original_low_kbps": [32, 48, 64, 96, 128],
                      } 
    )
    
    handler.evaluate(limit_test_nums=10, limit_speaker=-1)
    
    
