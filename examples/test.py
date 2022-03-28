from ssr_eval import SSR_Eval_Helper, BasicTestee

class MyTestee(BasicTestee):
    def __init__(self) -> None:
        super().__init__()
    
    def infer(self, x):
        """A testee that do nothing

        Args:
            x (np.array): [sample,], with original_sr sample rate
            target (np.array): [sample,], with target_sr sample rate

        Returns:
            np.array: [sample,]
        """
        return x
    
if __name__ == "__main__":
    testee = MyTestee()
    helper = SSR_Eval_Helper(testee, 
                      test_name="unprocess", 
                      test_data_root="./your_path/vctk_test", 
                      input_sr=44100,
                      output_sr=44100,
                      evaluation_sr=44100,
                      setting_lowpass_filtering = {
                          "filter":["cheby","butter"],
                          "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000],
                          "filter_order": [3,6,9]
                      }, 
                      setting_subsampling = {
                          "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000],
                      }, 
                      setting_fft = {
                          "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000],
                      }, 
                      setting_mp3_compression = {
                          "low_kbps": [32, 48, 64, 96, 128],
                      },
                      save_processed_result=False,
    )
    
    handler.evaluate(limit_test_nums=10, limit_speaker=-1)
    
