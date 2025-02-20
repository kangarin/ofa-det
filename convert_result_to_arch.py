def convert_dict(input_dict):
    # 定义分辨率列表
    # resolutions = [240, 360, 480, 600, 720]
    resolutions = [640]
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
input_dict = {'ks1': 0, 'ks2': 1, 'ks3': 2, 'ks4': 0, 'ks5': 0, 'ks6': 2, 'ks7': 0, 'ks8': 2, 'ks9': 2, 'ks10': 0, 'ks11': 0, 'ks12': 2, 'ks13': 1, 'ks14': 2, 'ks15': 0, 'ks16': 0, 'ks17': 1, 'ks18': 0, 'ks19': 1, 'ks20': 2, 'e1': 0, 'e2': 1, 'e3': 0, 'e4': 2, 'e5': 1, 'e6': 2, 'e7': 0, 'e8': 2, 'e9': 1, 'e10': 0, 'e11': 1, 'e12': 2, 'e13': 2, 'e14': 2, 'e15': 1, 'e16': 0, 'e17': 1, 'e18': 1, 'e19': 2, 'e20': 1, 'd1': 1, 'd2': 2, 'd3': 1, 'd4': 0, 'd5': 0, 'r': 0}
result = convert_dict(input_dict)
print(result)