
from train.train_detection_networks_with_kd_checkpoints import train
# from train.train_detection_networks_random_checkpoints import train
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
# 冻结backbone
for param in model.backbone.body.parameters():
    param.requires_grad = False

# 打印需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

import os
# if os.path.exists('ofa_mbv3_w12_fasterrcnn_adam.pth'):
#     model = torch.load('ofa_mbv3_w12_fasterrcnn_adam.pth')
train(model, 10, 'ofa_mbv3_w12_fasterrcnn_kd.pth', max_net_config, min_net_config, batch_size=1,
      backbone_learning_rate=1e-4, head_learning_rate=1e-4, 
          min_backbone_lr=1e-4, min_head_lr=1e-4, resume_from=None)
model = torch.load('ofa_mbv3_w12_fasterrcnn_kd.pth')