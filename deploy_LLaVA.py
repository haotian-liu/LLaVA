# 1. 从 LLaVA 项目中导入所有需要的函数
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# 2. 定义你要使用的模型、你的问题和图片
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
# 你可以使用 URL 或者本地文件的路径
# image_file = "/path/to/your/local/image.jpg"
image_file = "/remote-home/shijiajia/LLaVA/LLaVA/images/llava_logo.png"

# 3. 将所有参数打包成一个 'args' 对象
# 这段代码模拟了从命令行传入参数的过程
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

# 4. 调用高级函数来执行所有操作并打印结果
print("Running model evaluation...")
eval_model(args)
print("Done.")

