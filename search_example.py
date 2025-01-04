from search.backbone_search import create_study, run_study
from search.common import get_optimal_architecture
from utils.bn_calibration import set_running_statistics
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.imagenet_dataset import get_test_dataset, get_dataloader
from datasets.common_transform import common_transform_with_normalization_list
import random
import torch
import torchvision.transforms as transforms

if __name__ == '__main__':
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    model = get_ofa_supernet_mbv3_w12()
    study_name = "ofa_supernet_mbv3_w12_mac"
    study = create_study(study_name)
    n_trials = 1000
    run_study(model, study, n_trials, 'cpu', [240, 360, 480, 600, 720], 'ofa_supernet_mbv3_w12')

    arch_list = get_optimal_architecture(study, 'ofa_supernet_mbv3_w12', [240, 360, 480, 600, 720])

    print(arch_list)

    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_with_normalization_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)

    test_dataset = get_test_dataset()
    test_dataloader = get_dataloader(test_dataset, 1)

    for i in range(1000):
        one_arch = random.choice(arch_list)
        print(one_arch)
        accuracy = one_arch['accuracy']
        latency = one_arch['latency']
        arch = one_arch['arch']
        resolution = one_arch['resolution']
        model.set_active_subnet(**arch)
        set_running_statistics(model, calib_dataloader)
        model.eval()

        # test
        img, target = next(iter(test_dataloader))

        # 获取真实的类别名称
        print(test_dataset.classes[target.item()])

        # resize to resolution
        img = transforms.Resize((resolution, resolution))(img)

        with torch.no_grad():
            out = model(img)

        # # 获取最大的类别
        # _, pred = out.max(1)
        # # 获取类别名称
        # print(test_dataset.classes[pred])
        
        # 获取top5的类别
        _, pred = out.topk(5, 1)
        # 获取类别名称
        print([test_dataset.classes[i] for i in pred.squeeze(0).tolist()])

        # 显示图片
        import matplotlib.pyplot as plt
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.show()
        


        