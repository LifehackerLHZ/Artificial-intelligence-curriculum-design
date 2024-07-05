

# 导入必要的库
import os  # 操作系统接口模块，用于文件和目录操作
import mindspore as ms  # 导入MindSpore，主要用于机器学习和深度学习任务
import mindspore.context as context  # MindSpore的上下文管理模块，用于设置执行环境
import mindspore.dataset.transforms.c_transforms as C  # MindSpore数据集转换模块，用于数据处理
import mindspore.dataset.vision.c_transforms as CV  # MindSpore视觉数据转换模块，用于图像处理
from mindspore import nn  # MindSpore的神经网络模块
from mindspore.train import Model  # MindSpore的模型训练模块
from mindspore.train.callback import LossMonitor  # MindSpore的损失监控回调模块

# 设置MindSpore的执行上下文，选择计算模式和设备
# context.set_context函数用于设置MindSpore的执行上下文，选择计算模式和设备
# GRAPH_MODE表示图模式，代码会被编译成静态图执行，提高性能
# device_target指定计算设备，这里选择CPU
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# 创建数据集函数
def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=64):
    """
    创建数据集并进行预处理

    参数:
    data_dir (str): 数据集路径，包含训练和测试数据集的目录
    training (bool): 是否为训练集，默认为True，表示创建训练数据集
    batch_size (int): 批处理大小，默认为32
    resize (tuple): 调整后的图像大小，默认为(32, 32)
    rescale (float): 图像缩放比例，默认为1/(255*0.3081)
    shift (float): 图像偏移量，默认为-0.1307/0.3081
    buffer_size (int): 缓存大小，用于数据混洗，默认为64

    返回:
    ds (Dataset): 处理后的数据集
    """
    data_path = os.path.join(data_dir, 'train' if training else 'test')  # 根据是否是训练集选择数据路径
    ds = ms.dataset.MnistDataset(data_path)  # 加载MNIST数据集

    # 数据预处理：
    # Resize调整图像大小
    # Rescale重新缩放像素值
    # HWC2CHW将图像格式从Height x Width x Channel转换为Channel x Height x Width
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))  # 将标签转换为int32类型

    # 数据混洗和批处理：
    # shuffle用于打乱数据顺序
    # batch将数据分成批次，每批次包含batch_size个样本
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return ds  # 返回预处理后的数据集

# 定义LeNet5神经网络模型
class LeNet5(nn.Cell):
    """
    LeNet5模型定义

    LeNet5是一个经典的卷积神经网络，结构包括两层卷积层，三层全连接层，和池化及激活函数。
    """
    def __init__(self):
        super(LeNet5, self).__init__()  # 调用父类的初始化方法
        # 定义第一层卷积，输入通道为1（灰度图像），输出通道为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        # 定义第二层卷积，输入通道为6，输出通道为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()  # ReLU激活函数，用于引入非线性
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，核大小为2x2，步幅为2
        self.flatten = nn.Flatten()  # 展平层，将多维张量展平为一维
        # 定义第一层全连接层，输入大小为400，输出大小为120
        self.fc1 = nn.Dense(400, 120)
        # 定义第二层全连接层，输入大小为120，输出大小为84
        self.fc2 = nn.Dense(120, 84)
        # 定义第三层全连接层，输入大小为84，输出大小为10，对应10个类别
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        """
        前向传播定义

        参数:
        x (Tensor): 输入张量，表示输入的图像

        返回:
        x (Tensor): 输出张量，表示分类的概率分布
        """
        x = self.relu(self.conv1(x))  # 第一层卷积，接ReLU激活函数
        x = self.pool(x)  # 第一层池化
        x = self.relu(self.conv2(x))  # 第二层卷积，接ReLU激活函数
        x = self.pool(x)  # 第二层池化
        x = self.flatten(x)  # 展平张量
        x = self.fc1(x)  # 第一层全连接
        x = self.fc2(x)  # 第二层全连接
        x = self.fc3(x)  # 第三层全连接，输出分类结果

        return x  # 返回输出

# 训练函数
def train(data_dir, lr=0.01, momentum=0.9, num_epochs=10):
    """
    训练函数

    参数:
    data_dir (str): 数据集路径，包含训练和测试数据集的目录
    lr (float): 学习率，控制模型参数更新的速度
    momentum (float): 动量，用于加速梯度下降，默认为0.9
    num_epochs (int): 训练轮数，表示模型训练的次数，默认为10
    """
    ds_train = create_dataset(data_dir)  # 创建训练数据集
    ds_eval = create_dataset(data_dir, training=False)  # 创建测试数据集

    net = LeNet5()  # 实例化LeNet5模型
    # 定义损失函数，使用Softmax交叉熵损失函数，sparse=True表示使用稀疏标签，reduction='mean'表示对损失求平均
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 定义优化器，使用动量优化算法
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    # 定义损失监控回调，每次训练打印损失
    loss_cb = LossMonitor(per_print_times=ds_train.get_dataset_size())

    # 创建模型，指定网络、损失函数、优化器和评估指标
    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # 训练模型，指定训练轮数、数据集和回调
    model.train(num_epochs, ds_train, callbacks=[loss_cb], dataset_sink_mode=False)
    # 评估模型，使用测试数据集，dataset_sink_mode=False表示不使用数据集下沉模式
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)  # 输出评估指标

# 主程序入口
if __name__ == "__main__":
    import argparse  # 导入argparse库，用于解析命令行参数
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    # 添加命令行参数，data_url表示数据集路径，train_url表示训练输出位置
    parser.add_argument('--data_url', required=False, default='MNIST/', help='数据位置')
    parser.add_argument('--train_url', required=False, default=None, help='训练输出位置')
    args, unknown = parser.parse_known_args()  # 解析命令行参数

    # 如果数据位置是S3存储，使用moxing库从OBS复制数据
    if args.data_url.startswith('s3'):
        import moxing  # 导入moxing库，用于操作OBS
        # 从公开OBS桶复制数据集
        moxing.file.copy_parallel(src_url="s3://share-course/dataset/MNIST/", dst_url='MNIST/')
        data_path = 'MNIST/'
    else:
        data_path = os.path.abspath(args.data_url)  # 获取数据路径的绝对路径

    train(data_path)  # 调用训练函数训练模型

# 运行代码前需激活MindSpore环境
# conda activate mindspore
# 进入代码目录并运行代码
# cd C:\Users\F\Desktop\course-master\course-master\07_cloud_base\lenet5
# python main.py --data_url=C:\Users\F\Desktop\course-master\course-master\07_cloud_base\lenet5\MNIST
