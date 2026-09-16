---
library_name: transformers
language:
- am
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: Whisper Small Amharic - Biniyam Daniel
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Small Amharic - Biniyam Daniel

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3777
- Wer: 51.4327

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Wer     |
|:-------------:|:-------:|:----:|:---------------:|:-------:|
| 0.0866        | 4.1322  | 1000 | 0.2203          | 52.3233 |
| 0.0107        | 8.2645  | 2000 | 0.3035          | 52.2556 |
| 0.0008        | 12.3967 | 3000 | 0.3588          | 51.0552 |
| 0.0006        | 16.5289 | 4000 | 0.3777          | 51.4327 |


### Framework versions

- Transformers 4.57.0
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.22.1
