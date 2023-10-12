# LLaVA (LoRA, Preview)

NOTE: This is a technical preview, and is not yet ready for production use. We are still running hyperparameter search for the LoRA model, and will release the final model soon.  If you'd like to contribute to this, please contact us.

You need latest code base for LoRA support (instructions [here](https://github.com/haotian-liu/LLaVA#upgrade-to-latest-code-base))

## Demo (Web UI)

Please execute each of the commands below one by one (after the previous one has finished).  The commands are the same as launching other demos except for an additional `--model-base` flag to specify the base model to use. Please make sure the base model corresponds to the LoRA checkpoint that you are using.  For this technical preview, you need Vicuna v1.1 (7B) checkpoint (if you do not have that already, follow the instructions [here](https://github.com/lm-sys/FastChat#vicuna-weights)).

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```
You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-vicuna-7b-v1.1-lcs_558k-instruct_80k_3e-lora-preview-alpha --model-base /path/to/vicuna-v1.1
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.


## Training

Please see sample training scripts for [LoRA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune_lora.sh) and [QLoRA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune_qlora.sh).

We provide sample DeepSpeed configs, [`zero3.json`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/zero3.json) is more like PyTorch FSDP, and [`zero3_offload.json`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/zero3_offload.json) can further save memory consumption by offloading parameters to CPU. `zero3.json` is usually faster than `zero3_offload.json` but requires more GPU memory, therefore, we recommend trying `zero3.json` first, and if you run out of GPU memory, try `zero3_offload.json`. You can also tweak the `per_device_train_batch_size` and `gradient_accumulation_steps` in the config to save memory, and just to make sure that `per_device_train_batch_size` and `gradient_accumulation_steps` remains the same.

If you are having issues with ZeRO-3 configs, and there are enough VRAM, you may try [`zero2.json`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/zero2.json). This consumes slightly more memory than ZeRO-3, and behaves more similar to PyTorch FSDP, while still supporting parameter-efficient tuning.

## Create Merged Checkpoints

```Shell
python scripts/merge_lora_weights.py \
    --model-path /path/to/lora_model \
    --model-base /path/to/base_model \
    --save-model-path /path/to/merge_model
```
