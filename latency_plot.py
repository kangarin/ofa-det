import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
# from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
# from models.fpn.ofa_supernet_resnet50_fpn import Resnet50Fpn
from models.detection.fasterrcnn import get_faster_rcnn
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
# from inference.detection_inference import DetectionInference

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_faster_rcnn(Mbv3W12Fpn(get_ofa_supernet_mbv3_w12()))
# model = get_faster_rcnn(Resnet50Fpn(get_ofa_supernet_resnet50()))
# model = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_200_mini_extracted.pth', map_location=device)

# checkpoint = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_200_mini.pth', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

def sample_and_measure_latency(model, num_samples, input_size, device, warmup_runs=10, test_runs=20):
    """
    随机采样子网络并测量其延迟
    
    Args:
        model: 模型实例
        num_samples: 要采样的子网络数量
        input_size: 输入尺寸
        device: 运行设备
        warmup_runs: 预热运行次数
        test_runs: 测试运行次数
        
    Returns:
        latencies: 所有子网络的平均延迟列表
        configs: 对应的子网络配置列表
    """
    latencies = []
    configs = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Sampling networks"):
            # 采样子网络配置
            subnet_config = model.backbone.body.sample_active_subnet()
            model.backbone.body.set_active_subnet(**subnet_config)
            
            # 测量时延
            from evaluation.latency_eval import eval_latency
            avg_latency, std_latency = eval_latency(
                model, input_size, device, warmup_runs, test_runs
            )
            
            latencies.append(avg_latency)
            configs.append(subnet_config)
            
    return latencies, configs

def visualize_latency_distribution(latencies, save_path="latency_distribution.png"):
    """
    可视化延迟分布
    
    Args:
        latencies: 延迟数据列表
        save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 创建分布图
    sns.histplot(latencies, kde=True)
    
    # 添加统计信息
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    plt.axvline(mean_latency, color='r', linestyle='--', label=f'Mean: {mean_latency:.2f}ms')
    plt.fill_between([mean_latency-std_latency, mean_latency+std_latency],
                     plt.ylim()[0], plt.ylim()[1],
                     alpha=0.2, color='r',
                     label=f'Std: {std_latency:.2f}ms')
    
    plt.title("Subnet Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to {save_path}")
    print(f"Statistics:\n  Mean: {mean_latency:.2f}ms\n  Std: {std_latency:.2f}ms")
    return mean_latency, std_latency

def analyze_network_latency(model, num_samples=100, input_size=640, device='cuda',
                          save_path="latency_distribution.png"):
    """
    完整的网络延迟分析流程
    """
    print(f"Starting latency analysis for {num_samples} subnet samples...")
    
    # 采样并测量延迟
    latencies, configs = sample_and_measure_latency(
        model, num_samples, input_size, device
    )
    
    # 可视化并保存结果
    mean_latency, std_latency = visualize_latency_distribution(latencies, save_path)
    
    # 找出最快和最慢的网络配置
    min_idx = np.argmin(latencies)
    max_idx = np.argmax(latencies)
    
    print("\nFastest subnet configuration:")
    print(f"Latency: {latencies[min_idx]:.2f}ms")
    print(configs[min_idx])
    
    print("\nSlowest subnet configuration:")
    print(f"Latency: {latencies[max_idx]:.2f}ms")
    print(configs[max_idx])
    
    return latencies, configs

# 使用示例
if __name__ == "__main__":
    # model和device需要在外部定义
    latencies, configs = analyze_network_latency(
        model,
        num_samples=100,  # 采样100个子网络
        input_size=640,
        device='cuda',
        save_path="subnet_latency_distribution.png"
    )