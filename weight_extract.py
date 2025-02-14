import torch
import os

from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from models.detection.fasterrcnn import get_faster_rcnn
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn

# 提取权重
def extract_model(ckpt_path, model):
    # 加载checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 加载权重到模型
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存完整模型
    save_path = os.path.splitext(ckpt_path)[0] + '_extracted.pth'
    torch.save(model, save_path)
    print(f'模型已保存到: {save_path}')
    
    return save_path

ckpt_path = 'server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_500_mini_full_net.pth'
model = get_faster_rcnn(Mbv3W12Fpn(get_ofa_supernet_mbv3_w12()))
save_path = extract_model(ckpt_path, model)