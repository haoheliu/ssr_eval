# Speech Super-resolution Evaluation and Benchmarking
<b>What this repo do:</b>
- A toolbox for the evaluation of speech super-resolution algorithms.
- Unify the evaluation pipline of speech super-resolution algorithms for a easier comparison between different systems.
- Benchmarking speech super-resolution methods (pull request is welcome). Encouraging reproducible research.

I build this repo while I'm writing my paper for INTERSPEECH 2022: <i>Neural Vocoder is All You Need for Speech Super-resolution</i>. The model mentioned in this paper, NVSR, will also be open-sourced here.

## Installation  
Install via pip:
```shell
pip3 install ssr_eval
```
Please make sure you have already installed [sox](http://sox.sourceforge.net/sox.html).

## Quick Example

<b>A basic example:</b> Evaluate on a system that do nothing:

```python
from ssr_eval import test 
test()
```
- The evaluation result json file will be stored in the ./results directory: [Example file](https://github.com/haoheliu/ssr_eval/blob/main/examples/results/2022-03-28-18:07:54.109221-unprocessed.json)
- The code will automatically handle stuffs like downloading test sets.
- You will find a field "averaged" at the bottom of the json file that looks like below. This field mark the performance of the system.
```json
"averaged": {
        "proc_fft_24000_44100": {
            "lsd": 5.152331300436993,
            "log_sispec": 5.8051057146229095,
            "sispec": 30.23394207533686,
            "ssim": 0.8484425044157442
        }
    }
```
Here we report four metrics: 
1. Log spectral distance(LSD).
2. Log scale invariant spectral distance [1] (log-sispec).
3. Scale invariant spectral distance [1] (sispec).
4. Structral similarity (SSIM).

:warning: **LSD is the most widely used metric for super-resolution.** And I include another three metrics just in case you need them. 

<hr>

![main_idea](https://github.com/haoheliu/ssr_eval/blob/main/pics/main.png)

Below is the code of test()

```python
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

def test():
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
    ## Use all eight speakers in the test set for evaluation (limit_test_speaker=-1) 
    ## Evaluate on 10 utterance for each speaker (limit_test_nums=10)
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)
```
The code will automatically handle stuffs like downloading test sets. The evaluation result will be saved in the ./results directory.

## Baselines

We provide several pretrained baselines. For example, to run the NVSR baseline, you can click the link in the following table for more details. 

<hr>

<b>Table.1 Log-spectral distance (LSD) on different input sampling-rate (Evaluated on 44.1kHz).</b>

|  Method | One for all | Params| 2kHz | 4kHz | 8kHz | 12kHz | 16kHz | 24kHz | 32kHz |  AVG |
|:--------------------:|:----:|:----:|:----:|:----:|:----:|:-----:|:-----:|:-----:|:-----:|:----:|
| NVSR [[Pretrained Model](https://github.com/haoheliu/ssr_eval/tree/main/examples/NVSR)] | Yes | 99.0M | 1.04 | 0.98 | 0.91 |  0.85 |  0.79 |  0.70 |  0.60 | 0.84 |
| WSRGlow(24kHz→48kHz) | No | 229.9M | - | - | - |  - |  - |  0.79 |  - | - |
| WSRGlow(12kHz→48kHz) | No | 229.9M | - | - | - |  0.87 |  - |  - |  - | - |
| WSRGlow(8kHz→48kHz) | No | 229.9M | - | - | 0.98 |  - |  - |  - |  - | - |
| WSRGlow(4kHz→48kHz) | No | 229.9M | - | 1.12 | - |  - |  - | - |  - | - |
| Nu-wave(24kHz→48kHz) | No | 3.0M | - | - | - |  - |  - |  1.22 |  - | - |
| Nu-wave(12kHz→48kHz) | No | 3.0M | - | - | - |  1.40 |  - |  - |  - | - |
| Nu-wave(8kHz→48kHz) | No | 3.0M | - | - | 1.42 |  - |  - |  - |  - | - |
| Nu-wave(4kHz→48kHz) | No | 3.0M | - | 1.42 | - |  - |  - |  - |  - | - |
| Unprocessed      | - |  - | 5.69 | 5.50 | 5.15 |  4.85 |  4.54 |  3.84 |  2.95 | 4.65 |

> Click the link of the model for more details.

> Here "one for all" means model can process flexible input sampling rate.

## Features
The following code demonstrate the full options in the SSR_Eval_Helper:

```python
testee = MyTestee()
helper = SSR_Eval_Helper(testee, # Your testsee object with 'infer' function implemented
                        test_name="unprocess",  # The name of this test. Used for saving the log file in the ./results directory
                        test_data_root="./your_path/vctk_test", # The directory to store the test data, which will be automatically downloaded.
                        input_sr=44100, # The sampling rate of the input x in the 'infer' function
                        output_sr=44100, # The sampling rate of the output x in the 'infer' function
                        evaluation_sr=48000, # The sampling rate to calculate evaluation metrics. 
                        save_processed_result=False, # If True, save model output in the dataset directory.
                        # (Recommend/Default) Use fourier method to simulate low-resolution effect
                        setting_fft = {
                            "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000], # The cutoff frequency of the input x in the 'infer' function
                        }, 
                        # Use lowpass filtering to simulate low-resolution effect. All possible combinations will be evaluated. 
                        setting_lowpass_filtering = {
                            "filter":["cheby","butter","bessel","ellip"], # The type of filter 
                            "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000], 
                            "filter_order": [3,6,9] # Filter orders
                        }, 
                        # Use subsampling method to simulate low-resolution effect
                        setting_subsampling = {
                            "cutoff_freq": [1000, 2000, 4000, 6000, 8000, 12000, 16000],
                        }, 
                        # Use mp3 compression method to simulate low-resolution effect
                        setting_mp3_compression = {
                            "low_kbps": [32, 48, 64, 96, 128],
                        },
)

helper.evaluate(limit_test_nums=10, # For each speaker, only evaluate on 10 utterances.
                limit_test_speaker=-1 # Evaluate on all the speakers. 
                )
```
:warning:
<b>
I recommand all the users to use fourier method (setting_fft) to simulate low-resolution effect for the convinence of comparing between different system.
</b>

## Dataset Details
We build the test sets using VCTK (version 0.92), a multi-speaker English corpus that contains 110 speakers with different accents. 
- Speakers used for the test set: p360, p361, p362, p363, p364, p374, p376, s5
- For the remaining 100 speakers, p280 and p315 are omitted for the technical issues.
- Other 98 speakers are used for training.

## Citation

If you find this repo useful for your research, please consider citing:

```bibtex
@misc{liu2022neural,
      title={Neural Vocoder is All You Need for Speech Super-resolution}, 
      author={Haohe Liu and Woosung Choi and Xubo Liu and Qiuqiang Kong and Qiao Tian and DeLiang Wang},
      year={2022},
      eprint={2203.14941},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## Reference 

> [1] Liu, Haohe, et al. "VoiceFixer: Toward General Speech Restoration with Neural Vocoder." arXiv preprint arXiv:2109.13731 (2021).