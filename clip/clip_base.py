import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ==========加载模型和分词器==========
model_name = "openai/clip-vit-base-patch32"

model = CLIPTextModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)

model.eval()
model.to(device)
print(f"模型已加载至 {device}")


# 2. ==========准备输入文本==========
texts = ["一只猫", "一只狗", "一只猫和一只狗"]  # 示例输入，替换为你的m个文本
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)


# 3. ==========定义一个钩子函数来捕获中间值==========
q_list = [[] for _ in range(len(model.text_model.encoder.layers))]  # 存储每一层的Q
k_list = [[] for _ in range(len(model.text_model.encoder.layers))]  # 存储每一层的K
ffn_output_list = [[] for _ in range(len(model.text_model.encoder.layers))]  # 存储每一层的FFN输出

def get_q_hook(layer_idx):
    def hook(module, input, output):
        q_list[layer_idx].append(output.detach().cpu().numpy())
    return hook

def get_k_hook(layer_idx):
    def hook(module, input, output):
        k_list[layer_idx].append(output.detach().cpu().numpy())
    return hook

def get_ffn_hook(layer_idx):
    def hook(module, input, output):
        ffn_output_list[layer_idx].append(output.detach().cpu().numpy())
    return hook


# 4. ==========注册钩子函数==========
for i, layer in enumerate(model.text_model.encoder.layers):
    layer.self_attn.q_proj.register_forward_hook(get_q_hook(i))
    layer.self_attn.k_proj.register_forward_hook(get_k_hook(i))
    layer.mlp.register_forward_hook(get_ffn_hook(i))


# 5. ==========获取模型输出==========
with torch.no_grad():
    outputs = model(**inputs)

# 提取FFN的权重
state_dict = model.state_dict()
ffn_weights = {}
for i in range(len(model.text_model.encoder.layers)):
    ffn_weights[f'layer_{i}_fc1_weight'] = state_dict[f'text_model.encoder.layers.{i}.mlp.fc1.weight'].cpu().numpy()
    ffn_weights[f'layer_{i}_fc1_bias'] = state_dict[f'text_model.encoder.layers.{i}.mlp.fc1.bias'].cpu().numpy()
    ffn_weights[f'layer_{i}_fc2_weight'] = state_dict[f'text_model.encoder.layers.{i}.mlp.fc2.weight'].cpu().numpy()
    ffn_weights[f'layer_{i}_fc2_bias'] = state_dict[f'text_model.encoder.layers.{i}.mlp.fc2.bias'].cpu().numpy()


# 6. ==========保存所有数据为NumPy数组==========
# 保存Q、K和FFN输出
for i in range(len(model.text_model.encoder.layers)):
    # 保存Q
    np.save(f'q_layer_{i}.npy', np.array(q_list[i]))
    # 保存K
    np.save(f'k_layer_{i}.npy', np.array(k_list[i]))
    # 保存FFN输出
    np.save(f'ffn_output_layer_{i}.npy', np.array(ffn_output_list[i]))

# 保存FFN权重
for key, value in ffn_weights.items():
    np.save(f'{key}.npy', value)

print("所有数据已保存为NumPy数组！")