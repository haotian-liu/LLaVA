import sys
import os

# Ensure the project root directory is 
# automatically added to Python module search path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
