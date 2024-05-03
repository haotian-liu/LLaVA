CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
--model-path liuhaotian/llava-v1.5-7b \
--image-file "/node_data/mok/red.png" \
--load-4bit