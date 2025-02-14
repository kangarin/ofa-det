
from train.train_detection_subnet import train_subnet
from models.detection.fasterrcnn import get_faster_rcnn
from models.backbone.ofa_supernet import get_max_net_config, get_min_net_config
from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
import torch
max_net_config = get_max_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
backbone = get_ofa_supernet_mbv3_w12()
backbone_with_fpn = Mbv3W12Fpn(backbone)
model = get_faster_rcnn(backbone_with_fpn)
import os
if os.path.exists('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_500_mini_full_net_extracted.pth'):
    model = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_500_mini_full_net_extracted.pth')
# 打印需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

subnet_config = {'ks': [5, 7, 5, 3, 5, 3, 7, 7, 5, 7, 3, 3, 3, 3, 7, 5, 5, 5, 5, 3], 'e': [6, 6, 6, 4, 3, 6, 3, 3, 4, 3, 4, 4, 6, 6, 4, 6, 4, 4, 3, 3], 'd': [2, 2, 2, 2, 2]}
train_subnet(model, subnet_config, 100, 'single_subnet.pt', 2, 1e-4, 1e-4, 1e-4, 1e-4)