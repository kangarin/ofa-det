# # 测试bn校准对精度的影响
# from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
# from evaluation.classification_accuracy_eval import eval_accuracy
# from datasets.imagenet_dataset import get_dataloader, get_test_dataset
# from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
# from datasets.common_transform import common_transform_with_normalization_list
# from torchvision import transforms
# from utils.bn_calibration import set_running_statistics

# model = get_ofa_supernet_mbv3_w12()

# eval_dataset = get_test_dataset()
# eval_dataloader = get_dataloader(eval_dataset, 4)

# calib_image_num = 10
# calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_with_normalization_list))

# for i in range(5):
#     subnet_config = model.sample_active_subnet()
#     model.set_active_subnet(**subnet_config)
#     top1, top5 = eval_accuracy(model, 224, eval_dataloader, 100, 'cpu',(1,5))
#     print(top1, top5)

#     calib_dataloader = create_fixed_size_dataloader(calib_dataset, calib_image_num)
#     set_running_statistics(model, calib_dataloader, calib_image_num)
#     top1, top5 = eval_accuracy(model, 224, eval_dataloader, 100, 'cpu',(1,5))
#     print(top1, top5)


import numpy as np
import matplotlib.pyplot as plt
from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from evaluation.classification_accuracy_eval import eval_accuracy
from datasets.imagenet_dataset import get_dataloader, get_test_dataset
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.common_transform import common_transform_with_normalization_list
from torchvision import transforms
from utils.bn_calibration import set_running_statistics

def run_calibration_experiment(num_subnets=10, calib_nums=[10,20,30,40,50], eval_samples=100):
    # Initialize model and datasets
    model = get_ofa_supernet_mbv3_w12()
    eval_dataset = get_test_dataset()
    eval_dataloader = get_dataloader(eval_dataset, 4)
    calib_dataset = get_calib_dataset(
        custom_transform=transforms.Compose(common_transform_with_normalization_list)
    )
    
    # Store results
    results = {
        'subnet_configs': [],
        'baseline_top1': [],
        'baseline_top5': [],
        'calib_results': {n: {'top1': [], 'top5': []} for n in calib_nums}
    }
    
    # Run experiment for each subnet
    for i in range(num_subnets):
        print(f"\nTesting subnet {i+1}/{num_subnets}")
        
        # Sample and activate subnet
        subnet_config = model.sample_active_subnet()
        model.set_active_subnet(**subnet_config)
        results['subnet_configs'].append(subnet_config)
        
        # Get baseline accuracy
        top1, top5 = eval_accuracy(model, 224, eval_dataloader, eval_samples, 'cpu', (1,5))
        results['baseline_top1'].append(top1)
        results['baseline_top5'].append(top5)
        print(f"Baseline accuracy - Top1: {top1:.2f}, Top5: {top5:.2f}")
        
        # Test different calibration data sizes
        for calib_num in calib_nums:
            print(f"Testing with {calib_num} calibration samples...")
            calib_dataloader = create_fixed_size_dataloader(calib_dataset, calib_num)
            set_running_statistics(model, calib_dataloader, calib_num)
            
            top1, top5 = eval_accuracy(model, 224, eval_dataloader, eval_samples, 'cpu', (1,5))
            results['calib_results'][calib_num]['top1'].append(top1)
            results['calib_results'][calib_num]['top5'].append(top5)
            print(f"Calibrated accuracy - Top1: {top1:.2f}, Top5: {top5:.2f}")
    
    return results

def plot_results(results):
    plt.figure(figsize=(12, 6))
    
    # Calculate means and std devs
    baseline_mean_top1 = np.mean(results['baseline_top1'])
    baseline_std_top1 = np.std(results['baseline_top1'])
    
    calib_nums = sorted(results['calib_results'].keys())
    calib_means_top1 = [np.mean(results['calib_results'][n]['top1']) for n in calib_nums]
    calib_stds_top1 = [np.std(results['calib_results'][n]['top1']) for n in calib_nums]
    
    # Plot baseline as horizontal line
    plt.axhline(y=baseline_mean_top1, color='r', linestyle='--', label='Baseline (No Calibration)')
    plt.fill_between(calib_nums, 
                    [baseline_mean_top1 - baseline_std_top1] * len(calib_nums),
                    [baseline_mean_top1 + baseline_std_top1] * len(calib_nums),
                    color='r', alpha=0.1)
    
    # Plot calibration results
    plt.errorbar(calib_nums, calib_means_top1, yerr=calib_stds_top1, 
                marker='o', label='With Calibration', color='b')
    
    plt.xlabel('Number of Calibration Samples')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Impact of BN Calibration Data Size on Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('bn_calibration_results.png')
    plt.close()

def main():
    # Run experiment
    print("Starting BN calibration experiment...")
    results = run_calibration_experiment()
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Baseline Top-1 Accuracy: {np.mean(results['baseline_top1']):.2f} ± {np.std(results['baseline_top1']):.2f}")
    for calib_num in sorted(results['calib_results'].keys()):
        mean_acc = np.mean(results['calib_results'][calib_num]['top1'])
        std_acc = np.std(results['calib_results'][calib_num]['top1'])
        print(f"Calibration with {calib_num} samples: {mean_acc:.2f} ± {std_acc:.2f}")

if __name__ == "__main__":
    main()