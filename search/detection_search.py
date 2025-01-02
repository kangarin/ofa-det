import optuna
from models.backbone.ofa_supernet import get_architecture_dict
from optuna.samplers import NSGAIISampler
from evaluation.detection_accuracy_eval import eval_accuracy
from evaluation.latency_eval import eval_latency
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.bn_calibration import set_running_statistics
from search.custom_sampler import CustomNSGAIISampler
import torch

class ArchSearchOFADetection:
    # backbone_name: 'ofa_supernet_resnet50' or 'ofa_supernet_mbv3_w12' or 'ofa_supernet_mbv3_w10'
    def __init__(self, model, device, resolution_list, backbone_name, n_trials):
        self.model = model
        self.device = device
        self.resolution_list = resolution_list
        self.calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
        self.calib_dataloader = create_fixed_size_dataloader(self.calib_dataset, 10)
        self.backbone_name = backbone_name
        self.n_trials = n_trials

    def objective(self, trial):
        trial_number = trial.number
        arch_dict = get_architecture_dict(self.backbone_name)

        # 动态创建所有架构参数
        config = {}
        trial_params = {}

        for param_name, param_info in arch_dict.items():
            length = param_info['length']
            choices = param_info['choices']
            # 为每个参数创建trial建议值
            param_values = [
                trial.suggest_int(f'{param_name}{i+1}', 0, len(choices)-1) 
                for i in range(length)
            ]
            # 将索引映射到实际值
            mapped_values = [choices[idx] for idx in param_values]
            # 存储映射后的值
            config[param_name] = mapped_values
            trial_params[param_name] = param_values

        # 分辨率参数
        
        r = trial.suggest_int('r', 0, len(self.resolution_list)-1)
        r_values = self.resolution_list
        r_mapped = r_values[r]
        print("Arch: ", config, "resolution: ", r_mapped)

        # 根据trial_num增加逐渐增大eval的样本数
        min_image_num = 20
        max_image_num = 100
        image_num = min_image_num + (max_image_num - min_image_num) * trial_number // self.n_trials

        objective1 = get_accuracy(image_num, self.model, config, r_mapped, self.calib_dataloader, self.device)
        objective2 = get_latency(self.model, config, r_mapped, self.device)

        return objective1, objective2
    
def get_accuracy(image_num, model, config, img_size, calib_dataloader, device):
    model.backbone.body.set_active_subnet(**config)
    set_running_statistics(model, calib_dataloader)
    # 对于精度，可以用gpu加速
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result = eval_accuracy(model, img_size, image_num, device, show_progress=True)
    return result['AP@0.5:0.95']

def get_latency(model, config, img_size, device):
    model.backbone.body.set_active_subnet(**config)
    result = eval_latency(model, img_size, device)
    return result[0]
    
def create_study(study_name):
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, 
                                storage=storage_name, 
                                directions=["maximize", "minimize"],
                                load_if_exists=True,
                                # sampler=NSGAIISampler())
                                sampler=CustomNSGAIISampler())
    return study

def run_study(model, study, n_trials, device, resolution_list, backbone_name):
    arch_searcher = ArchSearchOFADetection(model, device, resolution_list, backbone_name, n_trials)
    objective = arch_searcher.objective
    study.optimize(objective, n_trials=n_trials)