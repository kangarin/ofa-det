# 因为旧版本的torchvision的detection模块实现有差异，直接加载会报错，需要做转换。
import torch
from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from models.detection.fasterrcnn import get_faster_rcnn
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
import torch.nn as nn
from torch.nn import Conv2d


if __name__ == '__main__':
    import copy
    model = get_faster_rcnn(Mbv3W12Fpn(get_ofa_supernet_mbv3_w12()))
    # print(model)
    model = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_500_mini_full_net_extracted.pth')
    new_model = copy.deepcopy(model)
    # for k, v in model.state_dict().items():
    #     print(k)
    # old_weights = new_model.rpn.head.conv[0][0].weight.clone()
    new_model.rpn.head.conv = new_model.rpn.head.conv[0][0]
    # print("权重是否相同:", torch.equal(old_weights, new_model.rpn.head.conv.weight))
    # print(new_model)
    torch.save(new_model, 'converted.pth')
