from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
from models.detection.fasterrcnn import get_faster_rcnn
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from inference.detection_inference import DetectionInference

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_faster_rcnn(Mbv3W12Fpn(get_ofa_supernet_mbv3_w12()))
model = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_200_mini_extracted.pth', map_location=device)

# checkpoint = torch.load('server_ofa_mbv3_w12_fasterrcnn_kd_ckpt_200_mini.pth', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
detection_inference = DetectionInference(model, device)

with torch.no_grad():

    # image_path1 = "D:\\Projects\\coco2017\\val2017\\000000512564.jpg"
    # image_path2 = "D:\\Projects\\coco2017\\val2017\\000000153229.jpg"
    image_path1 = "/Users/wenyidai/Development/datasets/coco2017/val2017/000000124798.jpg"
    image_path2 = "/Users/wenyidai/Development/datasets/coco2017/val2017/000000142238.jpg"

    for i in range(10):
        subnet_config = model.backbone.body.sample_active_subnet()
        detection_inference.set_active_subnet(**subnet_config)
        print(subnet_config)

        # from evaluation.detection_accuracy_eval import eval_accuracy
        # eval_accuracy(model, 640, 100, device)

        # 保存原始图像用于显示
        pil_img1 = Image.open(image_path1).convert("RGB")
        pil_img2 = Image.open(image_path2).convert("RGB")
        original_images = [pil_img1, pil_img2]

        # 转换图像为tensor
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        img1 = transform(pil_img1)
        img2 = transform(pil_img2)

        from utils.common import resize_images
        img = resize_images([img1, img2])

        # 推理
        batch_boxes, batch_labels, batch_scores = detection_inference.detect(img, 0.3)

        # 显示原始图像和检测结果
        for orig_img, boxes, labels, scores in zip(original_images, batch_boxes, batch_labels, batch_scores):
            plt.figure(figsize=(10, 8))
            # 显示原始PIL图像
            plt.imshow(orig_img)
            ax = plt.gca()
            
            # 确保边界框坐标与原始图像尺寸匹配
            img_width, img_height = orig_img.size
            for box, label, score in zip(boxes, labels, scores):
                # 绘制边界框
                x_min, y_min, x_max, y_max = box
                # 确保坐标不超出图像边界
                x_min = max(0, min(x_min, img_width))
                x_max = max(0, min(x_max, img_width))
                y_min = max(0, min(y_min, img_height))
                y_max = max(0, min(y_max, img_height))
                
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                rect = plt.Rectangle((x_min, y_min), box_width, box_height,
                                   fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # 添加类别和置信度标签
                from datasets.coco_dataset import coco_labels
                class_name = coco_labels[label]
                plt.text(x_min, y_min - 5, f'{class_name} {score:.2f}',
                        color='red', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))
            
            plt.axis('off')
            plt.show()

with torch.no_grad():
    # 采样十个网络，各推理10次，看时延
    for i in range(10):
        subnet_config = model.backbone.body.sample_active_subnet()
        detection_inference.set_active_subnet(**subnet_config)
        print(subnet_config)

        from evaluation.latency_eval import eval_latency
        avg, std = eval_latency(model, 640, device, 10, 10)
        print(avg, std)
        