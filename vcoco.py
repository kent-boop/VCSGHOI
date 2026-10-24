import numpy as np

# 替换为您的文件路径
file_path = '/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/vcoco_lib.txt'

try:
    # 尝试加载数据
    data = np.loadtxt(file_path)
    print(f"该特征文件的维度是: {data.shape}")
    
    # 维度解读示例：
    # 如果 shape 是 (263, 512)
    # 263 可能代表交互类别总数（vcoco_hoi_lib_2 中的条目数）
    # 512 代表每个交互对应的特征向量维度
except Exception as e:
    print(f"读取失败: {e}。如果文件很大或包含非数字字符，请尝试方法 2。")