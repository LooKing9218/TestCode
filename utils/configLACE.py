# -*- coding: utf-8 -*-
class DefaultConfig(object):
    # 网络训练设置
    net_work = 'DenseUnNet'
    num_classes = 9
    num_epochs = 300
    batch_size = 64
    validation_step = 1



    # 数据集与日志
    root = "/raid/wangmeng/Project/IdeaTest/LinT/DTS/DC_ClassicalResized"  # 数据存放的根目录
    train_file = "Datasets/DC_pred_train.csv"
    val_file = "Datasets/DC_pred_val.csv"
    test_file = "Datasets/DC_pred_test.csv"


    # 优化器设置
    lr = 1e-3
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4  # L2正则化系数


    save_model_path = './Model_Saved_Adam/NoDropPretrained_{}_bs_64_LR_{}_train_un_TempSoftV1_FC1_TotalEpoch300'.format(net_work,lr)
    log_dirs = './Logs_Adam/Log_NoDropPretrained_{}_loss_bs_64_LR_{}_train_un_TempSoftV1_FC1_TotalEpoch300'.format(net_work,lr)
    # 预训练模型
    pretrained = False
    pretrained_model_path = None

    # GPU使用
    cuda = 2
    num_workers = 4
    use_gpu = True


    # 网络预测
    trained_model_path = ''  # test的时候模型文件的选择（当mode='test'的时候用）
    predict_fold = 'predict_mask'
