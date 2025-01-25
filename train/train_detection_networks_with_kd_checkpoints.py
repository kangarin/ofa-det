from utils.bn_calibration import set_running_statistics
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.coco_dataset import get_train_dataset, get_dataloader
import torch
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.logger import setup_logger
from evaluation.detection_accuracy_eval import eval_accuracy
from torch.utils.tensorboard import SummaryWriter
import os

logger = setup_logger('train')

def fpn_distill_loss(teacher_fpn, student_fpn):
    """计算FPN特征图的蒸馏损失,带有特征归一化与注意力机制"""
    # 保持原有的fpn_distill_loss函数不变
    dist_loss = 0
    for level in teacher_fpn.keys():
        t_feat = teacher_fpn[level]
        s_feat = student_fpn[level]
        
        t_feat = torch.nn.functional.normalize(t_feat, p=2, dim=1)
        s_feat = torch.nn.functional.normalize(s_feat, p=2, dim=1)
        
        t_attention = torch.sum(torch.pow(t_feat, 2), dim=1, keepdim=True)
        s_attention = torch.sum(torch.pow(s_feat, 2), dim=1, keepdim=True)
        
        t_feat = t_feat * t_attention
        s_feat = s_feat * s_attention
        
        level_loss = torch.nn.functional.mse_loss(s_feat, t_feat)
        
        if torch.isnan(level_loss) or torch.isinf(level_loss):
            continue
            
        dist_loss += level_loss
    
    return dist_loss

def train(model, num_epochs, save_path, max_net_config, min_net_config,
          batch_size=1, 
          backbone_learning_rate=1e-3, head_learning_rate=1e-3, 
          min_backbone_lr=1e-4, min_head_lr=1e-4,  
          subnet_sample_interval=5,
          distill_alpha=5.0,
          resume_from=None,
          load_optimizer_scheduler=True):
    """训练检测网络，使用sandwich rule采样并添加FPN蒸馏"""
    # 设置TensorBoard
    writer = SummaryWriter('runs/detection_train_kd')
    
    # 设置优化器和学习率调度器
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if hasattr(model, 'head'):
        params_head = [p for p in model.head.parameters() if p.requires_grad]
    elif hasattr(model, 'roi_heads'):
        params_head = [p for p in model.roi_heads.parameters() if p.requires_grad] + \
                     [p for p in model.rpn.parameters() if p.requires_grad]

    params_backbone = [{'params': params_backbone, 'lr': backbone_learning_rate}]
    params_head = [{'params': params_head, 'lr': head_learning_rate}]
    
    optimizer_backbone = torch.optim.SGD(params_backbone, lr=backbone_learning_rate, momentum=0.9)
    optimizer_head = torch.optim.SGD(params_head, lr=head_learning_rate, momentum=0.9)
    
    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_backbone, T_max=num_epochs, eta_min=min_backbone_lr, verbose=True
    )
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_head, T_max=num_epochs, eta_min=min_head_lr, verbose=True
    )

    # 初始化开始轮次
    start_epoch = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Start training, using device: {device}")
    model.to(device)

    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        
        # 1. 先将模型移到目标设备
        model = model.to(device)
        
        # 2. 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 3. 重新初始化优化器(这样可以确保优化器状态在正确的设备上)
        params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        if hasattr(model, 'head'):
            params_head = [p for p in model.head.parameters() if p.requires_grad]
        elif hasattr(model, 'roi_heads'):
            params_head = [p for p in model.roi_heads.parameters() if p.requires_grad] + \
                        [p for p in model.rpn.parameters() if p.requires_grad]

        params_backbone = [{'params': params_backbone, 'lr': backbone_learning_rate}]
        params_head = [{'params': params_head, 'lr': head_learning_rate}]
        
        optimizer_backbone = torch.optim.SGD(params_backbone, lr=backbone_learning_rate, momentum=0.9)
        optimizer_head = torch.optim.SGD(params_head, lr=head_learning_rate, momentum=0.9)

        if load_optimizer_scheduler:
        
            # 4. 然后加载优化器状态
            optimizer_backbone.load_state_dict(checkpoint['optimizer_backbone_state_dict'])
            optimizer_head.load_state_dict(checkpoint['optimizer_head_state_dict'])
            
            # 5. 加载调度器状态
            scheduler_backbone.load_state_dict(checkpoint['scheduler_backbone_state_dict'])
            scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])

    start_epoch = checkpoint['epoch']
    
    # 准备数据集和数据加载器
    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    set_running_statistics(model, calib_dataloader, 10)

    train_dataloader = get_dataloader(get_train_dataset(), batch_size)
    sandwich_counter = 0
    current_subnet = max_net_config  # 初始化为最大网络配置

    for epoch in range(start_epoch, num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        det_loss_sum = 0
        distill_loss_sum = 0
        
        for data in train_dataloader:
            if not data:
                continue
                
            if i > 1000:  # 临时用于本地测试
                break
                
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            ofa_network = model.backbone.body
            
            # Sandwich采样逻辑保持不变
            if epoch == -1:
                ofa_network.set_max_net()
                current_subnet = max_net_config
            else:
                if i % subnet_sample_interval == 0:
                    sandwich_counter += 1
                    if sandwich_counter % 10 == 0:
                        current_subnet = max_net_config
                        ofa_network.set_active_subnet(**current_subnet)
                        logger.info("Using max network")
                    elif sandwich_counter % 10 == 1:
                        current_subnet = min_net_config
                        ofa_network.set_active_subnet(**current_subnet)
                        logger.info("Using min network")
                    else:
                        current_subnet = ofa_network.sample_active_subnet()
                        ofa_network.set_active_subnet(**current_subnet)
                        logger.info(f"Using random network: {current_subnet}")
                    
                    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
                    set_running_statistics(model, calib_dataloader, 10)

            # 知识蒸馏相关代码保持不变
            if sandwich_counter % 10 != 0:
                with torch.no_grad():
                    ofa_network.set_active_subnet(**max_net_config)
                    set_running_statistics(model, calib_dataloader, 10)
                    
                    teacher_fpn_features = []
                    for img in images:
                        feat = model.backbone(img.unsqueeze(0))
                        teacher_fpn_features.append(feat)
                    
                    ofa_network.set_active_subnet(**current_subnet)
                    set_running_statistics(model, calib_dataloader, 10)

            student_fpn_features = []
            for img in images:
                feat = model.backbone(img.unsqueeze(0))
                student_fpn_features.append(feat)

            det_loss_dict = model(images, targets)
            det_loss = sum(loss for loss in det_loss_dict.values())
            
            # 记录检测损失的各个组件
            for loss_name, loss_value in det_loss_dict.items():
                writer.add_scalar(f'DetLoss/{loss_name}', loss_value.item(), epoch * 1000 + i)
            
            if sandwich_counter % 10 != 0:
                batch_distill_loss = 0
                for t_fpn, s_fpn in zip(teacher_fpn_features, student_fpn_features):
                    batch_distill_loss += fpn_distill_loss(t_fpn, s_fpn)
                distillation_loss = batch_distill_loss / len(images)
                
                total_loss = det_loss + distill_alpha * distillation_loss
                writer.add_scalar('Loss/distillation', distillation_loss.item(), epoch * 1000 + i)
                distill_loss_sum += distillation_loss.item()
                
                if i % subnet_sample_interval == 0:
                    logger.info(f"Det loss: {det_loss.item():.4f}, Distill loss: {distillation_loss.item():.4f}")
            else:
                total_loss = det_loss
                distillation_loss = torch.tensor(0.0, device=device)

            optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            total_loss.backward()
            optimizer_backbone.step()
            optimizer_head.step()

            # 记录损失
            loss_sum += total_loss.item()
            det_loss_sum += det_loss.item()
            writer.add_scalar('Loss/total', total_loss.item(), epoch * 1000 + i)
            writer.add_scalar('Loss/detection', det_loss.item(), epoch * 1000 + i)

            if i > 0 and i % subnet_sample_interval == 0:
                avg_loss = loss_sum / subnet_sample_interval
                avg_det_loss = det_loss_sum / subnet_sample_interval
                avg_distill_loss = distill_loss_sum / subnet_sample_interval
                logger.info(f"Iteration #{i} loss: {avg_loss:.4f}")
                writer.add_scalar('Loss/avg_total', avg_loss, epoch * 1000 + i)
                writer.add_scalar('Loss/avg_detection', avg_det_loss, epoch * 1000 + i)
                writer.add_scalar('Loss/avg_distillation', avg_distill_loss, epoch * 1000 + i)
                torch.cuda.empty_cache()
                loss_sum = 0
                det_loss_sum = 0
                distill_loss_sum = 0
            i += 1
        
        logger.info(f"Epoch {epoch+1} finished.")
        scheduler_backbone.step()
        scheduler_head.step()
        writer.add_scalar('LR/backbone', scheduler_backbone.get_last_lr()[0], epoch)
        writer.add_scalar('LR/head', scheduler_head.get_last_lr()[0], epoch)
        logger.info(f"Current learning rates - backbone: {scheduler_backbone.get_last_lr()}, head: {scheduler_head.get_last_lr()}")
        
        try:
            # 评估不同网络配置的精度
            ofa_network.set_active_subnet(**max_net_config)
            set_running_statistics(model, calib_dataloader, 10)
            max_acc = eval_accuracy(model, None, 100, device, show_progress=False)['AP@0.5:0.95']
            writer.add_scalar('Accuracy/max_AP', max_acc, epoch)
            logger.info(f"Max network accuracy: {max_acc}")

            ofa_network.set_active_subnet(**min_net_config)
            set_running_statistics(model, calib_dataloader, 10)
            min_acc = eval_accuracy(model, None, 100, device, show_progress=False)['AP@0.5:0.95']
            writer.add_scalar('Accuracy/min_AP', min_acc, epoch)
            logger.info(f"Min network accuracy: {min_acc}")

            random_subnet = ofa_network.sample_active_subnet()
            ofa_network.set_active_subnet(**random_subnet)
            set_running_statistics(model, calib_dataloader, 10)
            random_acc = eval_accuracy(model, None, 100, device, show_progress=False)['AP@0.5:0.95']
            writer.add_scalar('Accuracy/random_AP', random_acc, epoch)
            logger.info(f"Random network config: {random_subnet}, accuracy: {random_acc}")
        except Exception as e:
            print(f"Error occured: {e}")

        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_backbone_state_dict': optimizer_backbone.state_dict(),
            'optimizer_head_state_dict': optimizer_head.state_dict(),
            'scheduler_backbone_state_dict': scheduler_backbone.state_dict(),
            'scheduler_head_state_dict': scheduler_head.state_dict()
        }, save_path)

    writer.close()
    logger.info("Training complete.")