from ssr_eval import SSR_Eval_Helper, BasicTestee

# You need to implement a class for the model to be evaluated.
class MyTestee(BasicTestee):
    def __init__(self) -> None:
        super().__init__()

    # You need to implement this function
    def infer(self, x):
        """A testee that do nothing

        Args:
            x (np.array): [sample,], with model_input_sr sample rate
            target (np.array): [sample,], with model_output_sr sample rate

        Returns:
            np.array: [sample,]
        """
        return x


testee = MyTestee()
# Initialize a evaluation helper
helper = SSR_Eval_Helper(
    testee,
    test_name="unprocessed",  # Test name for storing the result
    input_sr=44100,  # The sampling rate of the input x in the 'infer' function
    output_sr=44100,  # The sampling rate of the output x in the 'infer' function
    evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
    setting_fft={
        "cutoff_freq": [
            12000
        ],  # The cutoff frequency of the input x in the 'infer' function
    },
    save_processed_result=True
)
# Perform evaluation
helper.evaluate(limit_test_nums=1, limit_speaker=1)
