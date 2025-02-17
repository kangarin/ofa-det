import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from evaluation.detection_accuracy_eval import eval_accuracy
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from utils.bn_calibration import set_running_statistics
from datasets.common_transform import common_transform_list
from torchvision import transforms

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
model = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_500_mini_full_net_extracted.pth', map_location=device)

# checkpoint = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_200_mini.pth', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)

def get_accuracy(image_num, model, config, img_size, calib_dataloader, device):
    model.backbone.body.set_active_subnet(**config)
    set_running_statistics(model, calib_dataloader)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        result = eval_accuracy(model, img_size, image_num, device, show_progress=True)
        return result['AP@0.5:0.95']
    except Exception as e:
        print(f"Error during accuracy evaluation: {e}")
        return 0.0
    
def sample_and_measure_accuracy(model, num_samples, image_num, img_size, calib_dataloader, device):
    """
    随机采样子网络并测量其精度
    
    Args:
        model: 模型实例
        num_samples: 要采样的子网络数量
        image_num: 评估使用的图片数量
        img_size: 输入图片尺寸
        calib_dataloader: 用于校准的数据加载器
        device: 运行设备
        
    Returns:
        accuracies: 所有子网络的精度列表
        configs: 对应的子网络配置列表
    """
    accuracies = []
    configs = []
    
    for _ in tqdm(range(num_samples), desc="Sampling networks"):
        # 采样子网络配置
        subnet_config = model.backbone.body.sample_active_subnet()
        
        # 测量精度
        accuracy = get_accuracy(
            image_num=image_num,
            model=model,
            config=subnet_config,
            img_size=img_size,
            calib_dataloader=calib_dataloader,
            device=device
        )
        
        accuracies.append(accuracy)
        configs.append(subnet_config)
            
    return accuracies, configs

def visualize_accuracy_distribution(accuracies, save_path="accuracy_distribution.png"):
    """
    可视化精度分布
    
    Args:
        accuracies: 精度数据列表
        save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 创建分布图
    sns.histplot(accuracies, kde=True)
    
    # 添加统计信息
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    plt.axvline(mean_accuracy, color='r', linestyle='--', 
                label=f'Mean: {mean_accuracy:.2f}')
    plt.fill_between([mean_accuracy-std_accuracy, mean_accuracy+std_accuracy],
                     plt.ylim()[0], plt.ylim()[1],
                     alpha=0.2, color='r',
                     label=f'Std: {std_accuracy:.2f}')
    
    plt.title("Subnet Accuracy Distribution")
    plt.xlabel("mAP@0.5:0.95")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to {save_path}")
    print(f"Statistics:\n  Mean mAP: {mean_accuracy:.4f}\n  Std: {std_accuracy:.4f}")
    return mean_accuracy, std_accuracy

def analyze_network_accuracy(model, num_samples=100, image_num=5000, img_size=640,
                           calib_dataloader=None, device='cuda',
                           save_path="accuracy_distribution.png"):
    """
    完整的网络精度分析流程
    """
    print(f"Starting accuracy analysis for {num_samples} subnet samples...")
    
    # 采样并测量精度
    accuracies, configs = sample_and_measure_accuracy(
        model, num_samples, image_num, img_size, calib_dataloader, device
    )
    
    # 可视化并保存结果
    mean_accuracy, std_accuracy = visualize_accuracy_distribution(accuracies, save_path)
    
    # 找出最高和最低精度的网络配置
    max_idx = np.argmax(accuracies)
    min_idx = np.argmin(accuracies)
    
    print("\nBest performing subnet configuration:")
    print(f"mAP: {accuracies[max_idx]:.4f}")
    print(configs[max_idx])
    
    print("\nWorst performing subnet configuration:")
    print(f"mAP: {accuracies[min_idx]:.4f}")
    print(configs[min_idx])
    
    return accuracies, configs

# 使用示例
if __name__ == "__main__":
    # 需要在外部定义model, calib_dataloader和device
    accuracies, configs = analyze_network_accuracy(
        model,
        num_samples=100,  # 采样100个子网络
        image_num=500,   # 每个网络评估5000张图片
        img_size=640,
        calib_dataloader=calib_dataloader,
        device='cuda',
        save_path="subnet_accuracy_distribution.png"
    )