from transformers import AutoModel, AutoTokenizer

model_name = "LinkSoul/Chinese-Llama-2-7b"
model = AutoModel.from_pretrained(model_name,use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=True,use_fast=False)

save_path = "/remote-home/ThCheng/weights/linksoul_chinese_llama2_7b"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)