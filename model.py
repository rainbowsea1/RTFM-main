import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out

class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)
        # nn.linear()是用来设置网络中的全连接层的，而在全连接层中的输入与输出都是二维张量
        # 初始节点为 2048
        # 三个全连接层FC的节点分别设置为512、128，以及1
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        #1.Dropout是为了防止过拟合而设置的
        #2.Dropout顾名思义有丢掉的意思
        #3.nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活
        #4.Dropout只能用在训练部分而不能用在测试部分
        #5.Dropout一般用在全连接神经网络映射层之后，如代码的nn.Linear(20, 30)之后
        # https://blog.csdn.net/weixin_47050107/article/details/122722516
        #Dropout字面意思就是“丢掉”，是为了防止神经网络出现过拟合，让隐藏层的节点在每次迭代时(包括正向和反向传播)有一定几率(keep-prob)失效。以提高模型的泛化能力，减少过拟合。

        #Dropout属于一种正则化技术。Dropout的probability，实践中最常用的是0.5(或0.1，0.05)
        #Dropout不能用于测试集，只能用于训练集！！！

        self.drop_out = nn.Dropout(0.7)
        # nn.ReLU()激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    # foword是基本模板https://blog.csdn.net/qq_42178122/article/details/116669379
    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        # 重构(或者调整)张量维度
        # torch.view(参数a，参数b，...)
        # 在上面例子中参数a=3和参数b=2决定了将一维的tt1重构成3x2维的张量。
        # 如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。
        # https://zhuanlan.zhihu.com/p/87856193
        # 表示一个有x个向量，则参数a=x/t/f
        out = out.view(-1, t, f)

        # 聚合函数
        # https://blog.csdn.net/Jormungand_V/article/details/110007057

        # aggregate(self, func=None, *args, **kwargs)
        # func：表示用于汇总数据的函数，可以为单个函数或函数列表
        # axis：表示函数作用于轴的方向，0或1，index或columns
        out = self.Aggregate(out)

        # 得模型具有更好的泛化性，并防止过拟合
        # 而dropout的实质则是以一定概率使得输入网络的数据某些维度上变为0
        # https://www.cnblogs.com/CircleWang/p/16025723.html
        out = self.drop_out(out)

        features = out
        # https://zhuanlan.zhihu.com/p/360980567
        # 线性整流函数，即变为f(x) = max(0,X)
        # self.fc1(features)表示全连接层的第一层输入
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        # Sigmoid，是常用的连续、平滑的s型激活函数
        scores = self.sigmoid(self.fc3(scores))

        # 解决维度不对等的问题
        # .mean(1)求平均数
        # https://zhuanlan.zhihu.com/p/433165698
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)

        normal_features = features[0:self.batch_size*10]
        normal_scores = scores[0:self.batch_size]

        # 神经网络对应的异常分数 abnormal_scores
        abnormal_features = features[self.batch_size*10:]
        abnormal_scores = scores[self.batch_size:]

        # norm求范数
        # 特征量级 = 缩减到2维的2范数
        # dim (int) – 缩减的维度
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        # 正常视频特征量级
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        # 异常视频特征量级
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes

        #  shape[0]为矩阵的行数
        #  同理shape[1]输出列数
        n_size = nfea_magnitudes.shape[0]

        # 行数为1,包大小为1
        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        # 1张量
        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        # topk沿给定dim维度返回输入张量input中 k 个最大值
        # idx_abn为前k给特征量级
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            # 从原tensor中获取指定dim和指定index的数据
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        # top 3 scores in abnormal bag based on the top-3 magnitude
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag


        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes