import argparse
# argparse模块是命令行选项、参数和子命令解析器
# https://blog.csdn.net/RudeTomatoes/article/details/117003291
parser = argparse.ArgumentParser(description='RTFM')
# -feat-extractor 特征提取器
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
# feature-size 特征大小 从预训练I3D（下载得到）提取2048维
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
# 输入形式 视频/音频/混合
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
# rgb特征列表训练集
parser.add_argument('--rgb-list', default='list/shanghai-i3d-train-10crop.list', help='list of rgb features ')
# rgb特征列表测试集
parser.add_argument('--test-rgb-list', default='list/shanghai-i3d-test-10crop.list', help='list of test rgb features ')
# 实际情况
parser.add_argument('--gt', default='list/gt-sh.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
# 每步学习的比率
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
# 包的大小/一个视频分为几帧
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
#
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='shanghai', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
# 训练的最大迭代次数
parser.add_argument('--max-epoch', type=int, default=15000, help='maximum iteration to train (default: 100)')
