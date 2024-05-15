
try:
    from llava.train.train import train
except: # e.g., on colab
    from train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
