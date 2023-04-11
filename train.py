import torch
import os
import tqdm
import socket
import numpy as np
import torch.nn as nn
from sklearn import metrics
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import utils.utils as u
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP
from torch import optim
from torch.nn import functional as F

# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    # beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)




def val(val_dataloader, model, epoch, args, mode, device):
    criterion_B = nn.BCELoss()

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()
    labels = []
    outputs = []

    predictions = []
    gts = []
    correct = 0.0
    # critern = nn.MSELoss()
    # MSE_ERROR = 0.0
    num_total = 0
    tbar = tqdm.tqdm(val_dataloader, desc='\r')  # 添加一个进度提示信息，只需要封装任意的迭代器 tqdm(iterator)，desc进度条前缀

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S


            pred = torch.softmax(b,dim=1)

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(cls_label.cpu().detach().float().numpy()[idx])
            # correct += torch.eq(pred,cls_label).sum().float().item()
    epoch_auc = metrics.roc_auc_score(labels, outputs)
    Acc = metrics.accuracy_score(gts, predictions)
    # Acc = correct/num_total
    # Error_Reg = MSE_ERROR/num_total
    if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
        os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))

    with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
        Txt.write("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode, round(Acc,6),round(epoch_auc,6)
        ))
    print("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode,round(Acc,6),round(epoch_auc,6)
        ))
    torch.cuda.empty_cache()
    return epoch_auc,Acc
import numpy as np
def train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device):
    criterion_B = nn.BCELoss()

    step = 0
    best_auc = 0.0
    best_auc_Test = 0.0
    for epoch in range(1,args.num_epochs+1):
        model.train()
        labels = []
        outputs = []
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)  # 进度条总迭代次数 len*batchsize
        tq.set_description('Epoch %d, lr %f' % (epoch, args.lr))  # 设置修改进度条的前缀内容
        loss_record = []
        train_loss = 0.0
        for i, img_data_list in enumerate(train_loader):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            optimizer.zero_grad()  # 根据backward()函数的计算，当网络参量反馈时梯度是被积累的而不是被替换掉；因此在每一个batch时设置一遍zero_grad将梯度清零
            pretict = model(Fundus_img)
            evidences = [F.softplus(pretict)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            # un_output = args.num_classes / S

            # Tem_Coef = ((epoch+1e-8) / args.num_epochs)*0.01
            Tem_Coef = (epoch/args.num_epochs)

            loss_CE = criterion(b/Tem_Coef, cls_label)


            loss_un += ce_loss(cls_label, alpha[0], args.num_classes, epoch, args.num_epochs, device)
            loss_ACE = torch.mean(loss_un)
            loss = loss_CE+loss_ACE#
            loss.backward()  # 反向传播回传损失，计算梯度（loss为一个零维的标量）
            optimizer.step()  # optimizer基于反向梯度更新网络参数空间，因此当调用optimizer.step()的时候应当是loss.backward()后
            tq.update(args.batch_size)  # 进度条每次更新batch_size个进度
            train_loss += loss.item()  # 使用.item()可以从标量中获取Python数字
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))  # 设置进度条后缀，显示之前所有loss的平均
            step += 1
            one_hot = torch.zeros(pretict.size(0), args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pretict = torch.softmax(pretict, dim=1)
            for idx_data in range(pretict.size(0)):
                outputs.append(pretict.cpu().detach().float().numpy()[idx_data])
                labels.append(one_hot.cpu().detach().float().numpy()[idx_data])

            if step%10==0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())  # loss_record包含所有loss的列表
        tq.close()
        torch.cuda.empty_cache()
        loss_train_mean = np.mean(loss_record)  # 此次训练epoch的448个Loss的平均值
        epoch_train_auc = metrics.roc_auc_score(labels, outputs)

        del labels,outputs

        writer.add_scalar('Train/loss_epoch', float(loss_train_mean),
                          epoch)  # 每一折的每个epoch记录一次平均loss和epoch到日志
        writer.add_scalar('Train/train_auc', float(epoch_train_auc),
                          epoch)  # 每一折的每个epoch记录一次平均loss和epoch到日志

        print('loss for train : {}, {}'.format(loss_train_mean,round(epoch_train_auc,6)))
        if epoch % args.validation_step == 0:  # 每训练validation_step个epoch进行一次验证，此时为1
            if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
                os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
            with open(os.path.join(args.save_model_path, "{}/{}_Metric.txt".format(args.net_work,args.net_work)), 'a+') as f:
                f.write('EPOCH:' + str(epoch) + ',')
            mean_AUC, mean_ACC = val(val_loader, model, epoch,args,mode="val",device=device)  # 验证集结果
            writer.add_scalar('Valid/Mean_val_AUC', mean_AUC, epoch)
            is_best = mean_AUC > best_auc  # is_best为bool，以验证集平均dice为指标判断是否为best
            best_auc = max(best_auc, mean_AUC)  # 更新当前best dice
            checkpoint_dir = os.path.join(args.save_model_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            mean_AUC_Test, mean_ACC_Test = val(test_loader, model, epoch, args, mode="Test",device=device) # 验证集结果
            writer.add_scalar('Test/Mean_Test_AUC', mean_AUC_Test, epoch)
            print('===> Saving models...')


            u.save_checkpoint_epoch({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_AUC': mean_AUC,
                'mean_ACC': mean_ACC,
                'mean_AUC_Test': mean_AUC_Test,
                'mean_ACC_Test': mean_ACC_Test,
            }, mean_AUC, mean_ACC, mean_AUC_Test, mean_ACC_Test, epoch, True, checkpoint_dir, stage="Test",
                filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))


def main(args=None,writer=None):
    train_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='train',
        data_file=args.train_file,
    ),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='val',
        data_file=args.val_file,
    ),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='test',
        data_file=args.test_file,
    ),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # bulid model
    device = torch.device('cuda:{}'.format(args.cuda))

    model = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # if torch.cuda.is_available() and args.use_gpu:
    #     model = torch.nn.DataParallel(model).cuda()  # torch.nn.DataParallel是支持并行GPU使用的模型包装器，并将模型放到cuda上
    # load trained model for test
    if args.trained_model_path:  # 测试时加载训练好的模型
        print("=> loading trained model '{}'".format(args.trained_model_path))
        checkpoint = torch.load(
            args.trained_model_path)  # torch.load：使用pickle unpickle工具将pickle的对象文件反序列化为内存，包括参数、优化器、epoch等
        model.load_state_dict(checkpoint['state_dict'])  # torch.nn.Module.load_state_dict:使用反序列化状态字典加载model’s参数字典
        print('Done!')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # set the loss criterion
    criterion = nn.CrossEntropyLoss().to(device)
    train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device)

if __name__ == '__main__':
    seed_list =[i_x for i_x in range(0,1000)]
    torch.set_num_threads(1)
    import random
    random.shuffle(seed_list)

    for seed in seed_list:
        u.setup_seed(seed)
        args = DefaultConfig()  # 配置设置
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # 指定gpu

        log_dir = os.path.join(args.log_dirs,"Seed_{}_Lr_{}".format(seed,args.lr))  # 获取当前计算节点名称cu01
        writer = SummaryWriter(log_dir=log_dir)  # 创建writer object，log会被存入指定文件夹，writer.add_scalar保存标量值到log
        args.save_model_path = os.path.join(args.save_model_path,"Seed_{}_Lr_{}".format(seed,args.lr))

        main(args=args, writer=writer)