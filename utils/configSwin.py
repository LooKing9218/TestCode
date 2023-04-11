# -*- coding: utf-8 -*-
class DefaultConfig(object):
    # 网络训练设置
    net_work = 'SwinTransformer'
    num_classes = 9
    input_size = (224,224)
    num_epochs = 100
    batch_size = 64
    validation_step = 1
    save_model_path = './Model_Saved_Adam_224/{}_bs_64_LR_1e-4_TempSoftV1Swin'.format(net_work)
    log_dirs = './Logs_Adam_224/Log_{}_loss_bs_64_LR_1e-4_TempSoftV1Swin'.format(net_work)


    # 数据集与日志
    root = "/raid/wangmeng/Project/IdeaTest/LinT/DTS/DatasetResize_224"  # 数据存放的根目录
    train_file = "Datasets/pred_train.csv"
    val_file = "Datasets/pred_val.csv"
    test_file = "Datasets/pred_test.csv"


    # 优化器设置
    lr = 1e-4
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4  # L2正则化系数

    # 预训练模型
    pretrained = False
    pretrained_model_path = None

    # GPU使用
    cuda = '0'
    num_workers = 4
    use_gpu = True


    # 网络预测
    trained_model_path = ''  # test的时候模型文件的选择（当mode='test'的时候用）
    predict_fold = 'predict_mask'
    snapshot_fname = '/raid/wangmeng/Project/IdeaTest/CodePython/MICCAI2023/FedDR_FuServer/Pretrained/upernet_swin_base_patch4_window7_512x512.pth'