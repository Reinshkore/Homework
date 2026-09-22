import os

# 关闭联网检查，强制只从本地目录加载（避免联网校验 / 超时）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

# 注意：这里指向真正存放 config.json / model.safetensors 的目录
# 如果报错说目录不存在，请把下面这行改成 huggingface-cli 给出的真实路径
model_name = r"C:\Users\xnito\Reishkurs\LLM\models\Qwen--Qwen3-0.6B\snapshots\master"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 准备模型输入
prompt = "请概括一下Qwen3的优点。"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# 执行文本生成
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=4096 
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解码生成的ID
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
print("generate_text:", generate_text)
