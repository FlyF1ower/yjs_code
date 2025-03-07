import torch
import numpy as np
from transformers import FlavaTextModel, FlavaProcessor

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ==========加载模型和分词器==========
model_name = "C:/Users/xinlong/Desktop/code/python/flava_use/model/facebook/flava-full"

model = FlavaTextModel.from_pretrained(model_name)
tokenizer = FlavaProcessor.from_pretrained(model_name)

model.eval()
model.to(device)
print(f"模型已加载至 {device}")


# 2. ==========准备输入文本==========
texts = [
    "一只函数的返回大幅改进企鹅瑞华企鹅舞i意见猫", 
    "一只猫和一啊但是发射点发射点只狗", 
    "as阿凡达发hpoerujhiopertfasfa", 
    "放噶撒旦发射覅殴打事件回顾i哦速度返回结果点"
]  # 示例输入，替换为你的m个文本

inputs = tokenizer(
    text=texts, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
).to(device)


# 3. ==========定义一个钩子函数来捕获中间值==========
q_list = [[] for _ in range(len(model.encoder.layer))]  # 存储每一层的Q
k_list = [[] for _ in range(len(model.encoder.layer))]  # 存储每一层的K
ffn_output_list = [[] for _ in range(len(model.encoder.layer))]  # 存储每一层的FFN输出

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
for i, layer in enumerate(model.encoder.layer):
    def q_hook(module, input, output):
        q_list[i].append(output.detach().cpu().numpy())
    layer.attention.attention.query.register_forward_hook(q_hook)

    # def k_hook(module, input, output):
    #     k_list[i] = output.detach().cpu().numpy()
    # layer.attention.attention.key.register_forward_hook(k_hook)

    # layer.attention.attention.query.register_forward_hook(get_q_hook(i))
    # layer.attention.attention.key.register_forward_hook(get_k_hook(i))
    # layer.output.register_forward_hook(get_ffn_hook(i))


# 5. ==========获取模型输出==========
with torch.no_grad():
    outputs = model(**inputs)

# # 提取FFN的权重
# state_dict = model.state_dict()
# ffn_weights = {}
# for i in range(len(model.encoder.layer)):
#     ffn_weights[f'layer_{i}_fc1_weight'] = state_dict[f'encoder.layer.{i}.mlp.fc1.weight'].cpu().numpy()
#     ffn_weights[f'layer_{i}_fc1_bias'] = state_dict[f'encoder.layer.{i}.mlp.fc1.bias'].cpu().numpy()
#     ffn_weights[f'layer_{i}_fc2_weight'] = state_dict[f'encoder.layer.{i}.mlp.fc2.weight'].cpu().numpy()
#     ffn_weights[f'layer_{i}_fc2_bias'] = state_dict[f'encoder.layer.{i}.mlp.fc2.bias'].cpu().numpy()


# 6. ==========保存所有数据为NumPy数组==========
output_dir = "./flava_full_outputs"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存Q、K和FFN输出
for i in range(len(model.encoder.layer)):
    # 保存Q
    np.save(f'{output_dir}/q_layer_{i+1}.npy', np.array(q_list[i]))
    # # 保存K
    # np.save(f'{output_dir}/k_layer_{i+1}.npy', np.array(k_list[i]))
    # 保存FFN输出
    # np.save(f'{output_dir}/ffn_output_layer_{i+1}.npy', np.array(ffn_output_list[i]))
print(f"所有中间值已保存至 {output_dir}")

# # 保存FFN权重
# for key, value in ffn_weights.items():
#     np.save(f'{key}.npy', value)

print("所有数据已保存为NumPy数组！")

# ffn1 = np.load(f"flava_full_outputs/ffn_output_layer_1.npy")
# print(ffn1.shape)
