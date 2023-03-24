from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    # 加载配置和参数
    args = option.parser.parse_args()
    config = Config(args)

    #加载数据
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    #优化器 为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
    # 要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
    # 然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。
    # https://blog.csdn.net/kgzhang/article/details/77479737
    # params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
    # lr (float, optional) ：学习率(默认: 1e-3)
    # betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
    # eps (float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
    # weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)
    #  lr=config.lr[0]0.001
    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = 'C:\\Users\\rainb\\Desktop\\bysj\\output\\'  # put your own path here
    # AUC就是衡量学习器优劣的一种性能指标。从定义可知，AUC可通过对ROC曲线下各部分的面积求和而得。
    auc = test(test_loader, model, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if step % 5 == 0 and step > 200:

            auc = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

