import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np


def test(dataloader, model, args, viz, device):
    # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭／线程中锁的自动获取和释放等。
    # with torch.no_grad的作用
    # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
    # 在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
    # https://blog.csdn.net/sazass/article/details/116668755
    with torch.no_grad():
        # model.eval() 不启用 BatchNormalization（对网络中间的每层进行归一化处理） 和 Dropout （减少过拟合现象）
        # https://blog.csdn.net/sazass/article/details/116616664
        model.eval()

        pred = torch.zeros(0, device=device)

        # https://blog.csdn.net/zouxiaolv/article/details/109530461
        # enumerate:返回值有两个：一个是序号，也就是在这里的batch地址，一个是数据train_ids
        #
        # for i, data in enumerate(train_loader,1):此代码中1，是batch从batch=1开始，也就是batch的地址是从1开始算起，不是0开始算起。batch仍然是3个。就算batch从8开始，他也是三份，分别是8,9,10

        # enumerate(dataloader).__sizeof__() = 48
        # https://blog.csdn.net/qq_32938525/article/details/115588656
        for i, input in enumerate(dataloader):
            input = input.to(device)
            # permute permute()可以对任意高维矩阵进行转置。
            # https://cloud.tencent.com/developer/article/1914024#:~:text=%E7%AD%89%E4%BB%B7%E4%BA%8E%20torch.permute%20%28input%2C%20%28dim0%2C%20dim1%2C%20dim2%2C%20%E2%80%A6%29%29%E6%96%B9%E6%B3%95%EF%BC%8Cpermute%20%28%29%E5%8F%AF%E4%BB%A5%E5%AF%B9%E4%BB%BB%E6%84%8F%E9%AB%98%E7%BB%B4%E7%9F%A9%E9%98%B5%E8%BF%9B%E8%A1%8C%E8%BD%AC%E7%BD%AE%E3%80%82,%E6%9C%AA%E8%BF%9B%E8%A1%8C%E5%8F%98%E6%8D%A2%E5%89%8D%E7%9A%84dim%E6%98%AF%20%5B0%2C%201%2C%202%5D%E7%9A%84%E6%96%B9%E5%BC%8F%EF%BC%8C%20%E8%BD%AC%E6%8D%A2%E5%90%8E%E8%A1%A8%E7%A4%BA%E5%B0%86%E7%AC%AC0%E7%BB%B4%E5%BA%A6%E5%92%8C%E7%AC%AC2%E7%BB%B4%E5%BA%A6%E8%B0%83%E6%8D%A2%E3%80%82%202%E3%80%81torch.Tensor.transpose%20%28dim0%2C%20dim1%29
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            # squeeze()函数的功能是维度压缩。返回一个tensor（张量），其中 input 中大小为1的所有维都已删除。
            # https://blog.csdn.net/qq_40305043/article/details/107767652
            logits = torch.squeeze(logits, 1)
            # mean() 函数的参数：dim = 0, 按列求平均值，返回的形状是（1，列数）；dim = 1, 按行求平均值，返回的形状是（行数，1）, 默认不设置dim的时候，返回的是所有元素的平均值。
            logits = torch.mean(logits, 0)
            sig = logits
            #cat() 函数目的： 在给定维度上对输入的张量序列seq 进行连接操作
            # https://blog.csdn.net/xinjieyuan/article/details/105208352
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        # 来源于 sklearn 库的 metrics.roc_curve 主要用来计算ROC曲线面积
        # https://blog.csdn.net/qq_42479987/article/details/124157449
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        # auc https://blog.csdn.net/liweibin1994/article/details/79462554
        # FPR表示，在所有的异常包中，被预测成正的比例。称为伪阳性率
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        ## 保存模型
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        # visdom 划线
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc

