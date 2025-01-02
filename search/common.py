from models.backbone.ofa_supernet import get_architecture_dict

def plot_pareto_front(study):
    from optuna.visualization import plot_pareto_front
    plot_pareto_front(study)

def get_optimal_architecture(study, arch_name, resolution_list):
    print("Best trials(Accuracy, Latency):")
    arch_list = []
    for t in study.best_trials:
        # print(t.values)
        # print(convert_to_architecture_dict(t, arch_name, resolution_list))
        one_arch_dict = {}
        one_arch_dict['accuracy'] = t.values[0]
        one_arch_dict['latency'] = t.values[1]
        one_arch_dict['arch'], one_arch_dict['resolution'] = convert_to_architecture_dict(t, arch_name, resolution_list)
        arch_list.append(one_arch_dict)
    return arch_list


def convert_to_architecture_dict(trial, arch_name, resolution_list):
    arch_dict = get_architecture_dict(arch_name)
    config = {}
    for param_name, param_info in arch_dict.items():
        length = param_info['length']
        choices = param_info['choices']
        param_values = [trial.params[f'{param_name}{i+1}'] for i in range(length)]
        mapped_values = [choices[idx] for idx in param_values]
        config[param_name] = mapped_values

    r = trial.params['r']
    r_values = resolution_list
    r_mapped = r_values[r]
    return config, r_mapped