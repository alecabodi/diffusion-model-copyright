# Created by Chen Henry Wu
import logging
import os
import json
import wandb
import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)
from utils.config_utils import get_config
from utils.program_utils import get_model, get_preprocessor, get_evaluator, get_visualizer
from preprocess.to_model import get_multi_task_dataset_splits
from utils.training_arguments import CustomTrainingArguments
from trainer.trainer import Trainer

logger = logging.getLogger(__name__)

def get_dataset_splits(args):
    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    name2dataset_splits = dict()
    for name, arg_path in args.arg_paths:
        task_args = get_config(arg_path)
        task_raw_data_splits = datasets.load_dataset(
            path=task_args.raw_data.data_program,
            cache_dir=task_args.raw_data.data_cache_dir,
        )
        task_preprocessor = get_preprocessor(task_args.preprocess.preprocess_program)
        task_dataset_splits = task_preprocessor(task_args).preprocess(task_raw_data_splits, cache_root)

        name2dataset_splits[name] = task_dataset_splits

    return get_multi_task_dataset_splits(meta_args=args, name2dataset_splits=name2dataset_splits)


def setup_wandb(training_args, config_path="wandb_config.json"):
    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        # Read configuration from JSON file
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            project_name = config.get("project_name", "default_project")
            entity_name = config.get("entity_name", "default_entity")
        
        wandb.init(
            project=project_name,
            name=training_args.run_name,
            entity=entity_name,
        )
        wandb.config.update(training_args, allow_val_change=True)

        return wandb.run.dir
    else:
        return None



def main():

    # Get training_args and args.
    parser = HfArgumentParser(
        (
            CustomTrainingArguments,
        )
    )
    
    training_args, = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    args = get_config(training_args.cfg)

    # Deterministic behavior of torch.addmm.
    # Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # Ensure that the benchmark mode is disabled to ensure determinism

    # Set up wandb.
    wandb_run_dir = setup_wandb(training_args)
    
    # Setup output directory.
    os.makedirs(training_args.output_dir, exist_ok=True)
    # Build dataset splits.
    dataset_splits = get_dataset_splits(args)
    # Initialize evaluator.
    evaluator = get_evaluator(args.evaluation.evaluator_program)(args)
    print(type(evaluator))
    # Initialize visualizer.
    visualizer = get_visualizer(args.visualization.visualizer_program)(args)
    print(type(visualizer))

    # Initialize model.
    model = get_model(args.model.name)(args)

    # Initialize Trainer.
    trainer = Trainer(
        args=training_args,
        model=model,
        compute_metrics=evaluator.evaluate,
        train_dataset=dataset_splits['train'],
        eval_dataset=dataset_splits['dev'],
        visualizer=visualizer,
        wandb_run_dir=wandb_run_dir,
    )
    print(f'Rank {training_args.local_rank} Trainer build successfully.')

    # Load model from checkpoint.
    print("resume", training_args.resume_from_checkpoint)
    if training_args.resume_from_checkpoint:
        state_dict = torch.load(
            os.path.join(training_args.resume_from_checkpoint, transformers.WEIGHTS_NAME),
            map_location="cpu",
        )
        trainer.model.load_state_dict(state_dict, strict=True)
        # Free memory
        del state_dict

    # Training
    if training_args.do_train:
        metrics = trainer.train()
        trainer.save_model()

        metrics["train_samples"] = len(dataset_splits['train'])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation after training
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(
        metric_key_prefix="eval",
    )
    metrics["eval_samples"] = len(dataset_splits['dev'])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = trainer.predict(
            test_dataset=dataset_splits['test'],
            metric_key_prefix="test",
        )
        metrics["predict_samples"] = len(dataset_splits['test'])

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    main()
