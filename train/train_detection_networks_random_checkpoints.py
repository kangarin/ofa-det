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

def train(model, num_epochs, save_path, max_net_config, min_net_config,
          batch_size=1, 
          backbone_learning_rate=1e-3, head_learning_rate=1e-3, 
          min_backbone_lr=1e-4, min_head_lr=1e-4,  
          subnet_sample_interval=5,
          resume_from=None):
    """
    Train detection network using sandwich rule sampling.
    """
    # 设置优化器
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if hasattr(model, 'head'):
        params_head = [p for p in model.head.parameters() if p.requires_grad]
    elif hasattr(model, 'roi_heads'):
        params_head = [p for p in model.roi_heads.parameters() if p.requires_grad] + \
                     [p for p in model.rpn.parameters() if p.requires_grad]

    params_backbone = [{'params': params_backbone, 'lr': backbone_learning_rate}]
    params_head = [{'params': params_head, 'lr': head_learning_rate}]
    
    optimizer_backbone = torch.optim.SGD(params_backbone, 
                                       lr=backbone_learning_rate, 
                                       momentum=0.9, 
                                       weight_decay=1e-4)
    optimizer_head = torch.optim.SGD(params_head, 
                                    lr=head_learning_rate, 
                                    momentum=0.9, 
                                    weight_decay=1e-4)
    
    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_backbone,
        T_max=num_epochs,
        eta_min=min_backbone_lr
    )
    
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_head,
        T_max=num_epochs,
        eta_min=min_head_lr
    )

    # 设置Tensorboard
    writer = SummaryWriter('runs/detection_train')

    # 设置起始epoch
    start_epoch = 0
    
    # 加载检查点（如果存在）
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_backbone.load_state_dict(checkpoint['optimizer_backbone_state_dict'])
        optimizer_head.load_state_dict(checkpoint['optimizer_head_state_dict'])
        scheduler_backbone.load_state_dict(checkpoint['scheduler_backbone_state_dict'])
        scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    model.to(device)

    # 准备校准数据集
    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    set_running_statistics(model, calib_dataloader, 10)

    # 准备训练数据
    train_dataloader = get_dataloader(get_train_dataset(), batch_size)

    # 开始训练
    for epoch in range(start_epoch, num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        
        for data in train_dataloader:
            if not data:  # 跳过空数据
                continue
                
            # 临时用于本地测试，正式训练应该删除
            if i > 1000:
                break
                
            # 准备数据
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 获取OFA网络
            ofa_network = model.backbone.body

            subnet_config = ofa_network.sample_active_subnet()
            ofa_network.set_active_subnet(**subnet_config)
            logger.info("Using random network")
        
            # 重新校准BN统计量
            calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
            set_running_statistics(model, calib_dataloader, 10)

            # 前向传播和损失计算
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播和优化
            optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            losses.backward()
            optimizer_backbone.step()
            optimizer_head.step()

            # 记录损失
            loss_sum += losses.item()
            
            # Tensorboard记录每个mini-batch的损失
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), epoch * 1000 + i)
            writer.add_scalar('Loss/total', losses.item(), epoch * 1000 + i)

            # 打印训练信息
            if i > 0 and i % subnet_sample_interval == 0:
                avg_loss = loss_sum / subnet_sample_interval
                logger.info(f"Iteration #{i} loss: {avg_loss}")
                writer.add_scalar('Loss/avg_per_interval', avg_loss, epoch * 1000 + i)
                torch.cuda.empty_cache()
                loss_sum = 0
            i += 1
        
        # 每个epoch结束
        logger.info(f"Epoch {epoch+1} finished.")
        scheduler_backbone.step()
        scheduler_head.step()
        
        # 记录学习率
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
            'scheduler_head_state_dict': scheduler_head.state_dict(),
        }, save_path)

    writer.close()
    logger.info("Training complete.")