# 把项目根目录添加到Python路径中
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import asyncio
import websockets
import json
import optuna
from models.backbone.ofa_supernet import get_architecture_dict
from models.detection.fasterrcnn import get_faster_rcnn
from optuna.samplers import NSGAIISampler
from evaluation.detection_accuracy_eval import eval_accuracy
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.bn_calibration import set_running_statistics
import torch
from search.custom_sampler import CustomNSGAIISampler
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12

# 全局变量存储WebSocket连接
client_websocket = None

class ArchSearchOFADetection:
    def __init__(self, model, device, resolution_list, backbone_name, n_trials):
        self.model = model
        self.device = device
        self.resolution_list = resolution_list
        self.calib_dataset = get_calib_dataset(
            custom_transform=transforms.Compose(common_transform_list)
        )
        self.calib_dataloader = create_fixed_size_dataloader(self.calib_dataset, 10)
        self.backbone_name = backbone_name
        self.n_trials = n_trials

    async def get_latency(self, config, img_size):
        """通过WebSocket获取延迟测量结果"""
        if client_websocket is None:
            raise RuntimeError("No client connected")
        
        request_data = {
            "config": config,
            "resolution": img_size,
            "warmup_times": 10,
            "repeat_times": 50
        }
        
        # 发送请求
        await client_websocket.send(json.dumps(request_data))
        
        # 等待响应
        response = await client_websocket.recv()
        result = json.loads(response)
        return result["latency"]

    async def objective(self, trial):
        trial_number = trial.number
        arch_dict = get_architecture_dict(self.backbone_name)

        # 动态创建所有架构参数
        config = {}
        trial_params = {}

        for param_name, param_info in arch_dict.items():
            length = param_info['length']
            choices = param_info['choices']
            param_values = [
                trial.suggest_int(f'{param_name}{i+1}', 0, len(choices)-1) 
                for i in range(length)
            ]
            mapped_values = [choices[idx] for idx in param_values]
            config[param_name] = mapped_values
            trial_params[param_name] = param_values

        r = trial.suggest_int('r', 0, len(self.resolution_list)-1)
        r_mapped = self.resolution_list[r]
        print("Arch: ", config, "resolution: ", r_mapped)

        min_image_num = 100
        max_image_num = 200
        image_num = min_image_num + (max_image_num - min_image_num) * trial_number // self.n_trials

        objective1 = get_accuracy(
            image_num, 
            self.model, 
            config, 
            r_mapped, 
            self.calib_dataloader, 
            self.device
        )
        objective2 = await self.get_latency(config, r_mapped)

        return objective1, objective2

async def handle_client(websocket, path=None):
    """处理客户端连接"""
    global client_websocket
    client_websocket = websocket
    print("Client connected")
    try:
        # 保持连接直到客户端断开
        await websocket.wait_closed()
    finally:
        if client_websocket == websocket:
            client_websocket = None
        print("Client disconnected")

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

async def run_study(model, study, n_trials, device, resolution_list, backbone_name):
    """异步运行搜索实验"""
    arch_searcher = ArchSearchOFADetection(model, device, resolution_list, backbone_name, n_trials)
    
    async def optimize():
        for trial_idx in range(n_trials):
            trial = study.ask()
            values = await arch_searcher.objective(trial)
            print(f"Trial {trial_idx+1}/{n_trials}: {values}")
            study.tell(trial, values)
    
    await optimize()

async def main():
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, "localhost", 8768, ping_timeout=300)
    print("WebSocket server started")
    
    # 等待客户端连接
    while client_websocket is None:
        print("Waiting for client connection...")
        await asyncio.sleep(5)

    print("Starting experiment...")
    
    # 创建和运行实验
    study = optuna.create_study(
        study_name="search_mbv3_w12_fasterrcnn_remote_tx2_eval_on_det_net",
        storage="sqlite:///search_mbv3_w12_fasterrcnn_remote_tx2_eval_on_det_net.db",
        directions=["maximize", "minimize"],
        load_if_exists=True,
        sampler=CustomNSGAIISampler()
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_faster_rcnn(Mbv3W12Fpn(get_ofa_supernet_mbv3_w12()))
    model = torch.load('ofa_mbv3_w12_fasterrcnn_kd4.pth', map_location=device)
    await run_study(model, study, 5000, 'cpu', [240, 360, 480, 600, 720], 'ofa_supernet_mbv3_w12')
    
    # 保持服务器运行直到实验完成
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())