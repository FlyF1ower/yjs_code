import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. ==========加载模型和分词器==========
model_name = "meta-llama/Llama-3.3-70B-Instruct"  # 替换为实际的LLaMA 3模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
model.eval()  # 设置为评估模式

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpi")
model.to(device)
print(f"模型已加载至 {device}")


# 2. ==========准备输入文本==========
texts = ["你好，世界", "这是一个测试", "另一个输入"]  # 示例输入，替换为你的m个文本
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)


# 3. ==========定义一个钩子函数来捕获中间值==========
q_values = {}  # 存储Q值
k_values = {}  # 存储K值
ffn_outputs = {}  # 存储FFN输出

def register_hook(module):
    def hook_fn(module, input, output, layer_idx, hook_type):
        if hook_type == "attn_qk":
            # input[0]是query，input[1]是key
            q_values[layer_idx] = input[0].detach().cpu().numpy()   # Q值
            k_values[layer_idx] = input[1].detach().cpu().numpy()   # K值
        elif hook_type == "ffn":
            ffn_outputs[layer_idx] = output.detach().cpu().numpy()  # FFN输出

    # 为每一层的多头注意力模块和FFN模块注册钩子
    for i, layer in enumerate(model.model.layers):
        # 多头注意力模块的Q和K
        layer.self_attn.register_forward_hook(
            lambda module, input, output, idx=i: hook_fn(module, input, output, idx, "attn_qk")
        )
        # FFN模块的输出
        layer.mlp.register_forward_hook(
            lambda module, input, output, idx=i: hook_fn(module, input, output, idx, "ffn")
        )


# 4. ==========注册钩子函数==========
register_hook(model)


# 5. ==========获取模型输出==========
with torch.no_grad():
    outputs = model(**inputs)


# 6. ==========提取FFN的线性层权重==========
ffn_weights = {}
for i, layer in enumerate(model.model.layers):
    # LLaMA中的FFN通常有gate_proj, up_proj, down_proj三个线性层
    ffn_weights[f"layer_{i}_gate_proj"] = layer.mlp.gate_proj.weight.detach().cpu().numpy()
    ffn_weights[f"layer_{i}_up_proj"] = layer.mlp.up_proj.weight.detach().cpu().numpy()
    ffn_weights[f"layer_{i}_down_proj"] = layer.mlp.down_proj.weight.detach().cpu().numpy()


# 7. ==========保存所有数据为NumPy数组==========
output_dir = "./llama3_outputs"
import os
os.makedirs(output_dir, exist_ok=True)

# 保存Q值
for layer_idx, q in q_values.items():
    np.save(f"{output_dir}/q_values_layer_{layer_idx}.npy", q)

# 保存K值
for layer_idx, k in k_values.items():
    np.save(f"{output_dir}/k_values_layer_{layer_idx}.npy", k)

# 保存FFN权重
for weight_name, weight in ffn_weights.items():
    np.save(f"{output_dir}/{weight_name}.npy", weight)

# 保存FFN输出
for layer_idx, output in ffn_outputs.items():
    np.save(f"{output_dir}/ffn_output_layer_{layer_idx}.npy", output)

print(f"所有中间值已保存至 {output_dir}")