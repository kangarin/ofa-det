import torch

if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w10
    model = get_ofa_supernet_mbv3_w10()
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    model = get_ofa_supernet_mbv3_w12()

    random_subnet = model.sample_active_subnet()
    model.set_active_subnet(**random_subnet)

    random_input = torch.randn(1, 3, 224, 224)
    out = model(random_input)
    print(out.shape)

    from models.fpn.ofa_supernet_mbv3_w12_fpn import Mbv3W12Fpn
    model = Mbv3W12Fpn(model)
    out = model(random_input)
    print(out.keys())

    from models.detection.fasterrcnn import get_faster_rcnn
    model = get_faster_rcnn(model)
    model.eval()
    out = model(random_input)
    print(out)

    # from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
    # model = get_ofa_supernet_resnet50()

    # random_subnet = model.sample_active_subnet()
    # model.set_active_subnet(**random_subnet)

    # random_input = torch.randn(1, 3, 224, 224)
    # out = model(random_input)
    # print(out.shape)

    # from models.fpn.ofa_supernet_resnet50_fpn import Resnet50Fpn
    # model = Resnet50Fpn(model)
    # out = model(random_input)
    # print(out.keys())

    # from models.detection.fasterrcnn import get_faster_rcnn
    # model = get_faster_rcnn(model)
    # model.eval()
    # out = model(random_input)
    # print(out)

    model.to(device)
    random_input = torch.randn(1, 3, 224, 224).to(device)

    # 耗时测试
    import time
    with torch.no_grad():
        for i in range(20):
            random_subnet = model.backbone.body.sample_active_subnet()
            print(random_subnet)
            model.backbone.body.set_active_subnet(**random_subnet)
            start = time.time()
            for i in range(20):
                out = model(random_input)
            print("Average time: ", (time.time() - start) / 20)