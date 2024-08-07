# An Efficient Copyright Protection Method for Text-to-Image Diffusion Models

This repository contains the reference code for the report "An Efficient Copyright Protection Method for Text-to-Image Diffusion Models" (semester project at EPFL).

## Table of Contents

- [Installation](#installation)
- [Copyright Metric](#copyright-metric)
- [Copyright Protection](#copyright-protection)
- [FID](#fid)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Conda Environment Setup

1. **Install Conda**: Download and install from the [official Anaconda website](https://www.anaconda.com/products/distribution).

2. **Create a Conda Environment**:
    ```bash
    conda create --name myenv python=3.9
    ```

3. **Activate the Environment**:
    ```bash
    conda activate myenv
    ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Copyright Metric

The following files contain the code for running similarity metrics and plotting results. Provide necessary paths as highlighted in the corresponding files:

- `semantic_similarity.py` computes semantic similarity 
- `elpips/ShiftTolerant-LPIPS/lpips_similarity.py` computes perceptual similarity 
- `copyright_metric.py` combines the metrics above to yield copyright metric (as defined in the report)
- `l2_similarity.py` computes l2 norm based similarity (for comparison with proposed copyright metric)

## Copyright Protection

The directory `copyright_protection` contains reference code for the copyright protection method outlined in the report. The code makes use on the repo [generative-visual-prompt](https://github.com/ChenWu98/generative-visual-prompt), as acknowledged below.

Run the training and evaluation with the following command (modify parameters as needed). Training must be run twice (once per INN, using first semantic then perceptual similarity metric as EBM control - see `copyright_protection/model/lib/celeba/classifier.py`):

```bash
python main.py --seed <SEED> --cfg <CONFIG_PATH> --run_name <RUN_NAME> --logging_strategy <LOG_STRATEGY> --logging_first_step <TRUE_OR_FALSE>  --logging_steps <LOGGING_STEPS> --evaluation_strategy <EVAL_STRATEGY> --eval_steps <EVAL_STEPS> --metric_for_best_model <METRIC> --greater_is_better <TRUE_OR_FALSE> --save_strategy <SAVE_STRATEGY> --save_steps <SAVE_STEPS> --save_total_limit <SAVE_LIMIT> --load_best_model_at_end --gradient_accumulation_steps <GRAD_ACCUM_STEPS> --num_train_epochs <NUM_EPOCHS> --adafactor <TRUE_OR_FALSE> --learning_rate <LEARNING_RATE> --do_train --do_eval --do_predict --output_dir <OUTPUT_DIR> --overwrite_output_dir --per_device_train_batch_size <TRAIN_BATCH_SIZE> --per_device_eval_batch_size <EVAL_BATCH_SIZE> --eval_accumulation_steps <EVAL_ACCUM_STEPS> --ddp_find_unused_parameters <TRUE_OR_FALSE> --verbose <TRUE_OR_FALSE>
```

### Parameters

- **General**:
  - `--seed <SEED>`: Random seed.
  - `--cfg <CONFIG_PATH>`: Path to the configuration file (use `experiments/my_conf.cfg`).
  - `--run_name <RUN_NAME>`: Name of the run for logging purposes.

- **Logging**:
  - `--logging_strategy <LOG_STRATEGY>`: Strategy for logging (e.g., `steps` or `epoch`).
  - `--logging_first_step <TRUE_OR_FALSE>`: Whether to log the first training step.
  - `--logging_steps <LOGGING_STEPS>`: Interval of steps between logging.

- **Evaluation**:
  - `--evaluation_strategy <EVAL_STRATEGY>`: Strategy for evaluation (e.g., `steps` or `epoch`).
  - `--eval_steps <EVAL_STEPS>`: Interval of steps between evaluations.
  - `--metric_for_best_model <METRIC>`: Metric to determine the best model (use `ClassEnergy`).
  - `--greater_is_better <TRUE_OR_FALSE>`: Whether a higher metric value is better (use `false`).

- **Saving**:
  - `--save_strategy <SAVE_STRATEGY>`: Strategy for saving checkpoints (e.g., `steps` or `epoch`).
  - `--save_steps <SAVE_STEPS>`: Interval of steps between saving checkpoints.
  - `--save_total_limit <SAVE_LIMIT>`: Maximum number of checkpoints to keep.
  - `--load_best_model_at_end`: Load the best model at the end of training based on the specified metric.
  - `--overwrite_output_dir`: Overwrites the output directory, useful for starting fresh with each run.

- **Training**:
  - `--gradient_accumulation_steps <GRAD_ACCUM_STEPS>`: Number of steps to accumulate gradients before updating (e.g., `1`).
  - `--num_train_epochs <NUM_EPOCHS>`: Number of training epochs.
  - `--adafactor <TRUE_OR_FALSE>`: Use Adafactor optimizer instead of AdamW (use `false`).
  - `--learning_rate <LEARNING_RATE>`: Learning rate for the optimizer (e.g., `1e-3`).
  - `--per_device_train_batch_size <TRAIN_BATCH_SIZE>`: Training batch size per device (e.g., `10` or `100`).

- **Evaluation**:
  - `--do_train`: Perform training.
  - `--do_eval`: Perform evaluation.
  - `--do_predict`: Perform prediction.
  - `--per_device_eval_batch_size <EVAL_BATCH_SIZE>`: Evaluation batch size per device (e.g., `10` or `100`).
  - `--eval_accumulation_steps <EVAL_ACCUM_STEPS>`: Number of steps to accumulate evaluation results.
  - `--ddp_find_unused_parameters <TRUE_OR_FALSE>`: Whether to find unused parameters in DDP.
  - `--verbose <TRUE_OR_FALSE>`: Verbose output.

## FID

Run the notebook `fid.ipynb` (adjust file paths as needed). For more detailed instructions on dataset preparation and evaluation metrics, refer to the [DCR repository](https://github.com/somepago/DCR.git).

## Disclaimer

This code is provided as reference material and is intended for educational and informational purposes only. While efforts have been made to ensure the accuracy and functionality of the code, there is no guarantee that the experiments or results obtained using this code will be reproducible or free from errors. Users are encouraged to review the code, validate the results independently, and adapt or modify the code as needed for their specific use cases. Reproducibility of experiments may vary due to differences in environments, configurations, dependencies, and other factors.

The authors make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability, or availability of the code with respect to the experiments, results, or related materials. Any reliance you place on such information is therefore strictly at your own risk.

By using this code, you acknowledge that you understand and accept this disclaimer.

## License

The code is released under the following [LICENSE](./LICENSE).

## Acknowledgements

This project makes use of code taken from the following repositories:

- **Semantic Similarity** inspired from paper "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models" ([GitHub Repository](https://github.com/somepago/DCR))
    ```plaintext
    @inproceedings{somepalli2023diffusion,
      title={Diffusion art or digital forgery? investigating data replication in diffusion models},
      author={Somepalli, Gowthami and Singla, Vasu and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={6048--6058},
      year={2023}
    }
    ```

- **Perceptual Similarity** makes use of LPIPS metric as shown in paper "Shift-tolerant Perceptual Similarity Metric" ([GitHub Repository](https://github.com/abhijay9/ShiftTolerant-LPIPS/))
    ```plaintext
    @inproceedings{ghildyal2022stlpips,
      title={Shift-tolerant Perceptual Similarity Metric},
      author={Abhijay Ghildyal and Feng Liu},
      booktitle={European Conference on Computer Vision},
      year={2022}
    }
    ```

- **Copyright Protection Method** builds on paper "Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models" ([GitHub Repository](https://github.com/ChenWu98/generative-visual-prompt))
    ```plaintext
    @inproceedings{promptgen2022,
      title={Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models},
      author={Chen Henry Wu and Saman Motamed and Shaunak Srivastava and Fernando De la Torre},
      booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
      year={2022},
      url={https://openreview.net/forum?id=Gsbnnc--bnw}
    }
    ```



