# Finetune LLaVA on Custom Datasets

## Dataset Format

Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), and `conversations` (the conversation data between human and AI).

A sample JSON for finetuning LLaVA for generating tag-style captions for Stable Diffusion:

```json
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
```

## Command

If you have a limited task-specific data, we recommend finetuning from LLaVA checkpoints with LoRA following this [script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh).

If the amount of the task-specific data is sufficient, you can also finetune from LLaVA checkpoints with full-model finetuning following this [script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task.sh).

You may need to adjust the hyperparameters to fit each specific dataset and your hardware constraint.


