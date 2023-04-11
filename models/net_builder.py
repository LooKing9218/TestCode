def net_builder(name,num_classes=9):
    if name == 'ResNet50':
        from models.ResNet.ResNet import ResNet
        net= ResNet(num_classes=num_classes)
    elif name == 'Densenet':
        from models.Densenet.DenseNet import DenseNet
        net= DenseNet(num_classes=num_classes)
    elif name == 'VGG':
        from models.Densenet.DenseNet import DenseNet
        net= DenseNet(num_classes=num_classes)


    elif name == 'DenseUnNet':
        from models.Densenet.DenseUnNet import DenseUnNet
        net= DenseUnNet(num_classes=num_classes)
    elif name == 'ResUnNet50':
        from models.ResNet.ResUnNet import ResUnNet
        net= ResUnNet(num_classes=num_classes)

    else:
        raise NameError("Unknow Model Name!")
    return net
