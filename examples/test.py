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
            x (np.array): [sample,], with original_sr sample rate
            target (np.array): [sample,], with target_sr sample rate

        Returns:
            np.array: [sample,]
        """
        return x, {"additional_metrics_of_this_audio": 1.0}
    
if __name__ == "__main__":
    testee = MyTestee()
    
    handler = SR_Eval(testee, 
                      test_name="unprocess", 
                      test_data_root="/vol/research/dcase2022/sr_eval_vctk/vctk_test", 
                      model_input_sr=44100,
                      model_output_sr=44100,
                      evaluationset_sr=44100,
                        setting_lowpass_filtering = {
                          "filter":["cheby","butter"],
                          "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                          "filter_order": [3,6,9]
                      }, 
                      setting_subsampling = {
                          "original_low_sample_rate": [2000, 4000, 8000, 12000,16000, 24000, 32000],
                      }, 
                      setting_fft = {
                          "original_low_sample_rate": [2000, 4000, 8000, 12000, 16000, 24000, 32000],
                      }, 
                      setting_mp3_compression = {
                          "original_low_kbps": [32, 48, 64, 96, 128],
                      } 
    )
    
    handler.evaluate(limit_test_nums=10, limit_speaker=-1)
    
