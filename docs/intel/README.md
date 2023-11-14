# Intel Platforms 

Support [Intel GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)    
Support [Intel CPU Sapphire Rapids](https://ark.intel.com/content/www/us/en/ark/products/codename/126212/products-formerly-sapphire-rapids.html)    
Based on [Intel Extention for Pytorch](https://intel.github.io/intel-extension-for-pytorch)    
Verified Model: https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-gptq     


## Setup
1. Checkout **intel** branch: 
```
$ git clone https://github.com/haotian-liu/LLaVA.git
$ cd LLaVA   
$ git checkout origin/intel
```

2. Prepare Intel Extention for Pytorch environment. Refer to [this page](https://intel.github.io/intel-extension-for-pytorch)    

3. Install python packages, such as "transformers, sentencepiece, einops, ..."

## Run CLI Inference
```
$ python -m llava.serve.cli --model-path YOUR_MODEL_PATH/llava-llama-2-13b-chat-lightning-preview/ --image-file "https://llava-vl.github.io/static/images/view.jpg"
```
> USER: What are the things I should be cautious about when I visit this place?    
> ASSISTANT: When visiting this place, which features a pier extending into a lake surrounded by mountains, there are a few things to be cautious about. Firstly, be aware of the water conditions, as the lake may have strong currents or hidden obstacles beneath the surface. Additionally, be mindful of the weather conditions, as sudden changes in the weather can create hazardous conditions on the water. Lastly, always practice safe boating and swimming habits, and be aware of any posted signs or warnings near the pier or lake.


## Run Gradio Web UI   
* Process 1, launch a controller
```
$ python -m llava.serve.controller --host 0.0.0.0 --port 10000
```
* Process 2, launch a model worker
```
$ python -m llava.serve.model_worker --device xpu  --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path YOUR_MODEL_PATH/llava-llama-2-13b-chat-lightning-preview/
```
* Proess 3, launch a gradio web server
```
$ python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload   --port 7860
```

## DeepSpeed for multi-cards (TODO)
DeepSpeed by multi-cards (PVC, ARC770, Flex170) should also work. (TODO)    
Gaudi-2 (TODO)

## Known issues:
 * bfloat16 and float16 both can work, but float16 output is gibberish.
