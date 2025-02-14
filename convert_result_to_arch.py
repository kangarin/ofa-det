def convert_dict(input_dict):
    # 定义分辨率列表
    resolutions = [240, 360, 480, 600, 720]
    resolution = resolutions[input_dict['r']]
    print(f"Resolution: {resolution}")

    # 定义映射规则
    mapping_rules = {
        "ks": {
            "length": 20,
            "choices": [3, 5, 7],
        },
        "e": {
            "length": 20,
            "choices": [3, 4, 6],
        },
        "d": {
            "length": 5,
            "choices": [2, 3, 4],
        }
    }
    
    # 初始化结果字典
    result = {
        'ks': [],
        'e': [],
        'd': []
    }
    
    # 转换函数：将索引值转换为实际值
    def convert_value(index, choices):
        return choices[index]
    
    # 处理每种类型的值
    for key_type in ['ks', 'e', 'd']:
        length = mapping_rules[key_type]['length']
        choices = mapping_rules[key_type]['choices']
        
        # 处理每个索引
        for i in range(1, length + 1):
            input_key = f'{key_type}{i}'
            if input_key in input_dict:
                result[key_type].append(convert_value(input_dict[input_key], choices))
    
    return result

# 测试代码
input_dict = {'ks1': 1, 'ks2': 2, 'ks3': 1, 'ks4': 0, 'ks5': 1, 'ks6': 0, 'ks7': 2, 'ks8': 2, 'ks9': 1, 'ks10': 2, 'ks11': 0, 'ks12': 0, 'ks13': 0, 'ks14': 0, 'ks15': 2, 'ks16': 1, 'ks17': 1, 'ks18': 1, 'ks19': 1, 'ks20': 0, 'e1': 2, 'e2': 2, 'e3': 2, 'e4': 1, 'e5': 0, 'e6': 2, 'e7': 0, 'e8': 0, 'e9': 1, 'e10': 0, 'e11': 1, 'e12': 1, 'e13': 2, 'e14': 2, 'e15': 1, 'e16': 2, 'e17': 1, 'e18': 1, 'e19': 0, 'e20': 0, 'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd5': 0, 'r': 3}

result = convert_dict(input_dict)
print(result)