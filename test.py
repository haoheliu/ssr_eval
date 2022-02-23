import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")

from sr_eval_vctk import SR_Eval, BasicTestee

class MyTestee(BasicTestee):
    def __init__(self) -> None:
        super().__init__()
        
    def infer(self, x, target):
        """A testee that do nothing

        Args:
            x (np.array): [sample,]
            target (np.array): [sample,]

        Returns:
            np.array: [sample,]
        """
        return x, {"additional_metrics":0.9}
    
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
    
    handler.evaluate(limit_test_nums=2, limit_speaker=-1)
    
    
