Lora的时候遇到的问题：

TypeError: Accelerator.__init__() got an unexpected keyword argument 'use_seedable_sampler'

尝试的解决方案：
(1)
在llava-test环境中
pip install accelerate==0.27.2


恢复方法：
在llava-test环境中
pip install --upgrade llava
或者
pip install accelerate==0.21.0

(2)
会不会是得先用module load craype-accel-nvidia80把cuda toolkit激活。