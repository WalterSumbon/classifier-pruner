'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
    
class PrunableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PrunableBasicBlock, self).__init__()
        
        self.dependency_list = []
        
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        
        self.dependency_list.append({'conv':self.conv2, 'bn':self.bn2})

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes[1])
            )
            self.dependency_list.append({'conv':self.shortcut[0], 'bn':self.shortcut[1]})
            

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PrunableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PrunableBottleneck, self).__init__()
        
        self.dependency_list = []
        
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])
        
        self.dependency_list.append({'conv':self.conv3, 'bn':self.bn3})

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes[2]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes[2])
            )
            self.dependency_list.append({'conv':self.shortcut[0], 'bn':self.shortcut[1]})

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


_num_classes = {
    'cifar10' : 10,
    'cifar100' : 100,
}

class PrunableResNet_imagenet(nn.Module):
    def __init__(self, block, num_blocks, cfg=None):
        print("[dataset = imagenet]")
        super(PrunableResNet_imagenet, self).__init__()
        assert not(cfg is None)

        self.num_classes = 1000

        self.cfg = cfg
        
        self.dependency = []
        self.dependency_list = []
            
        self.in_planes = cfg[0][0]

        self.conv1 = nn.Conv2d(3, cfg[0][0], kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(cfg[0][0])
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dependency_list.extend([{'conv':self.conv1, 'bn':self.bn1}])
        
        if cfg[0][0] != cfg[1][-1]: # 当第一个卷积层的filter数目不同于第一个block的最后一个卷积层的filter数目时
            tmp = []
            for i in range(len(self.dependency_list)):
                tmp.append(self.dependency_list[i])
            self.dependency.append(tmp)
            self.dependency_list.clear()
        
        start = 1
        end = start + num_blocks[0]
        self.layer1 = self._make_layer(block, cfg[start:end], num_blocks[0], stride=1)
        
        start = end
        end = start + num_blocks[1]
        self.layer2 = self._make_layer(block, cfg[start:end], num_blocks[1], stride=2)
        
        start = end
        end = start + num_blocks[2]
        self.layer3 = self._make_layer(block, cfg[start:end], num_blocks[2], stride=2)
        
        start = end
        end = start + num_blocks[3]
        self.layer4 = self._make_layer(block, cfg[start:end], num_blocks[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        
        self.linear = nn.Linear(cfg[-1][-1], self.num_classes)
        
        # for i in range(len(self.dependency)):
        #     for j in range(len(self.dependency[i])):
        #         print(self.dependency[i][j]['conv'])
        #         print(self.dependency[i][j]['bn'])

    def _make_layer(self, block, blocks_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for i,stride in enumerate(strides):
            # print(blocks_planes[i])
            b = block(self.in_planes, blocks_planes[i],stride)
            layers.append(b)
            self.in_planes = blocks_planes[i][-1]
            
            self.dependency_list.extend(b.dependency_list)
            
        tmp = []
        for i in range(len(self.dependency_list)):
            tmp.append(self.dependency_list[i])
        self.dependency.append(tmp)
        self.dependency_list.clear()
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    # module_list: 不包含shortcutCB的所有剩余的<Conv, BN>
    # shortcutCB: 只包含shortcut中的<Conv, BN>
    def get_module_list(self): # get the module_list that only contain CONV and BN, but not contain CONV and BN which is in shortcut
        self.module_list = []
        self.shortcutCB_list = []
        CB = []
        for name,module in self.named_modules():
            if 'conv' in name:
                CB.append(module)
            elif isinstance(module, nn.Conv2d): # shortcut-CONV
                CB.append(module)
                
            if 'bn' in name: 
                CB.append(module)
                self.module_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
            elif isinstance(module, nn.BatchNorm2d): # shortcut-BN
                CB.append(module)
                self.shortcutCB_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
                
        return self.module_list, self.shortcutCB_list
        

class ResNetCFG_imagenet:   #包含各种结构信息的类
    def __init__(self):
        self.cfg18 = [[64], [64, 64], [64, 64], [128, 128], [128, 128], [256, 256], [256, 256], [512, 512], [512, 512]]
        self.cfg34 = [[64], [64, 64], [64, 64], [64, 64], [128, 128], [128, 128], [128, 128], [128, 128], [256, 256], \
                        [256, 256], [256, 256], [256, 256], [256, 256], [256, 256], [512, 512], [512, 512], [512, 512]]
        self.cfg50 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                      [128, 128, 512], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                      [256, 256, 1024], [256, 256, 1024], [512, 512, 2048], [512, 512, 2048], [512, 512, 2048]] 
        self.cfg101 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [128, 128, 512], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [512, 512, 2048], \
                       [512, 512, 2048], [512, 512, 2048]]
        self.cfg152 = [[64], [64, 64, 256], [64, 64, 256], [64, 64, 256], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [128, 128, 512], [128, 128, 512], [128, 128, 512], [128, 128, 512], [128, 128, 512], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024], \
                       [256, 256, 1024], [512, 512, 2048], [512, 512, 2048], [512, 512, 2048]]
        
        self.prune_idx18 = [1,3,5,7,9,11,13,15]
        self.prune_idx34 = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]

        # self.prune_idx50 = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44,46,47]
        self.prune_idx50 = []
        for idx in range(50 - 1): # 不算全连接层，有49个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx50.append(idx)

        self.prune_idx101 = []
        for idx in range(101 - 1): # 不算全连接层，有100个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx101.append(idx)

        self.prune_idx152 = []
        for idx in range(152 - 1): # 不算全连接层，有151个卷积层(也不包括shortcut支路上的卷积层)，能被3整除的代表第0个卷积层或block中的最后一个卷积层
            if idx % 3 != 0:
                self.prune_idx152.append(idx)
    
    # 根据原始模型获取该模型对应的cfg
    def get_cfg(self, raw_model): # raw_model is generated by class ResNet
        layers = 1 # '1' is linear layer
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
        
        if layers in [18, 34]:
            block_convs = 2
        else:
            block_convs = 3
        
        first_conv_flag = True
        cfg = []
        block = []
        count = 0
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                if first_conv_flag: # first conv's output channels
                    block = [params.shape[0]] 
                    cfg.append(deepcopy(block))
                    block.clear()
                    first_conv_flag = False
                else: # all conv's output channels in a block, but shortcut's conv is not included  
                    block.append(params.shape[0])
                    count = count + 1
                    if count == block_convs:
                        cfg.append(deepcopy(block))
                        block.clear()
                        count = 0
        return cfg
    
    # 获取prune = 0下的可剪枝的module的idx
    def get_prune_idx(self, model): 
        layers = 1 # '1' is linear layer
        for name,params in model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
                
        if layers == 18:
            return self.prune_idx18

        if layers == 34:
            return self.prune_idx34

        if layers == 50:
            return self.prune_idx50

        if layers == 101:
            return self.prune_idx101

        if layers == 152:
            return self.prune_idx152

        assert False,"Error: model's layers=%g is illegal!!!"%layers
                
    # 将一个1D的list变为2D的cfg
    def convert_list_to_cfg(self, li): 
        layers = len(li) + 1 # '1' is linear layer
        if layers in [18, 34]:
            block_convs = 2
        else:
            block_convs = 3
            
        first_conv_flag = True
        cfg = []
        block = []
        count = 0
        for i in range(len(li)):
            if first_conv_flag: # first conv's output channels
                block = [li[i]] 
                cfg.append(deepcopy(block))
                block.clear()
                first_conv_flag = False
                
            else: # all conv's output channels in a block, but shortcut's conv is not included  
                block.append(li[i])
                count = count + 1
                if count == block_convs:
                    cfg.append(deepcopy(block))
                    block.clear()
                    count = 0
        return cfg

def ResNet18_imagenet(cfg=None):
    if cfg is None:
        # return ResNet(BasicBlock, [2, 2, 2, 2])
        return PrunableResNet_imagenet(PrunableBasicBlock, [2, 2, 2, 2], cfg=ResNetCFG_imagenet().cfg18)
    else:
        return PrunableResNet_imagenet(PrunableBasicBlock, [2, 2, 2, 2], cfg=cfg)

def ResNet34_imagenet(cfg=None):
    if cfg is None:
        # return ResNet(BasicBlock, [3, 4, 6, 3])
        return PrunableResNet_imagenet(PrunableBasicBlock, [3, 4, 6, 3], cfg=ResNetCFG_imagenet().cfg34)
    else:
        return PrunableResNet_imagenet(PrunableBasicBlock, [3, 4, 6, 3], cfg=cfg)
    
def ResNet50_imagenet(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 4, 6, 3])
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 4, 6, 3], cfg=ResNetCFG_imagenet().cfg50)
    else:
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 4, 6, 3], cfg=cfg)
    
def ResNet101_imagenet(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 4, 23, 3])
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 4, 23, 3], cfg=ResNetCFG_imagenet().cfg101)
    else:
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 4, 23, 3], cfg=cfg)

def ResNet152_imagenet(cfg=None):
    if cfg is None:
        # return ResNet(Bottleneck, [3, 8, 36, 3])
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 8, 36, 3], cfg=ResNetCFG_imagenet().cfg152)
    else:
        return PrunableResNet_imagenet(PrunableBottleneck, [3, 8, 36, 3], cfg=cfg)

class ResNet_cifar(nn.Module):
    def __init__(self, num_layers, num_classes, cfg=None):
        print('[num_classes: %d, num_layers: %d]', num_classes, num_layers)
        super(ResNet_cifar, self).__init__()
        assert (num_layers-2)%6 == 0
        _n = (num_layers-2)//6
        if cfg is not None:
            self.cfg = cfg
        else:
            cfg = self.cfg = [[16]] + [[16,16]]*_n + [[32,32]]*_n + [[64,64]]*_n  #这里假设self.cfg不会被原位修改(指例如self.cfg[1][0]=100)
        self.num_classes = num_classes
        self.dependency = []
        self.dependency_list = []
        self.in_planes = self.cfg[0][0]

        self.conv1 = nn.Conv2d(3, self.cfg[0][0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cfg[0][0])

        self.dependency_list.extend([{'conv':self.conv1, 'bn':self.bn1}])

        self.layer1 = self._make_layer(self.cfg[1:1+_n],stride = 1)
        self.layer2 = self._make_layer(self.cfg[1+_n:1+_n*2],stride = 2)
        self.layer3 = self._make_layer(self.cfg[1+_n*2:1+_n*3],stride = 2)

        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(self.cfg[-1][-1], self.num_classes)

    def _make_layer(self, cfg, stride = 1):
        strides = [stride] + [1]*len(cfg)
        layers = []
        for i,planes in enumerate(cfg):
            b = PrunableBasicBlock(self.in_planes, planes, strides[i])
            layers.append(b)
            self.in_planes = planes[-1]
            self.dependency_list.extend(b.dependency_list)
        # the following code is directly copied from the `PrunableResNet_imagenet`
        tmp = []
        for i in range(len(self.dependency_list)):
            tmp.append(self.dependency_list[i])
        self.dependency.append(tmp)
        self.dependency_list.clear()
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # the following code is directly copied from the `PrunableResNet_imagenet`
    # module_list: 不包含shortcutCB的所有剩余的<Conv, BN>
    # shortcutCB: 只包含shortcut中的<Conv, BN>
    def get_module_list(self): # get the module_list that only contain CONV and BN, but not contain CONV and BN which is in shortcut
        self.module_list = []
        self.shortcutCB_list = []
        CB = []
        for name,module in self.named_modules():
            if 'conv' in name:
                CB.append(module)
            elif isinstance(module, nn.Conv2d): # shortcut-CONV
                CB.append(module)
                
            if 'bn' in name: 
                CB.append(module)
                self.module_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
            elif isinstance(module, nn.BatchNorm2d): # shortcut-BN
                CB.append(module)
                self.shortcutCB_list.append({"conv":CB[0],"bn":CB[1]})
                CB.clear()
                
        return self.module_list, self.shortcutCB_list

class ResNetCFG_cifar:   #包含各种结构信息的类
    def __init__(self,dataset):
        self.dataset = dataset

    def get_cfg(self,raw_model):
        layers = 1 # '1' is linear layer
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
        
        if layers in [18, 34]:
            block_convs = 2
        else:
            block_convs = 3
        
        first_conv_flag = True
        cfg = []
        block = []
        count = 0
        for name,params in raw_model.named_parameters():
            if 'conv' in name:
                if first_conv_flag: # first conv's output channels
                    block = [params.shape[0]] 
                    cfg.append(deepcopy(block))
                    block.clear()
                    first_conv_flag = False
                else: # all conv's output channels in a block, but shortcut's conv is not included  
                    block.append(params.shape[0])
                    count = count + 1
                    if count == block_convs:
                        cfg.append(deepcopy(block))
                        block.clear()
                        count = 0
        return cfg
        
    def get_prune_idx(self, model): 
        layers = 1 # '1' is linear layer
        for name,params in model.named_parameters():
            if 'conv' in name:
                layers = layers + 1
        assert (layers-2)%6 == 0

def PrunableResNet(num_layers, dataset, cfg=None):
    '总的可剪枝残差网络生成入口'
    assert dataset in ['imagenet','cifar10','cifar100']
    if dataset == 'imagenet':
        assert num_layers in [18,34,50,101,152]
        return globals()['ResNet%d_imagenet'%num_layers](cfg)
    else:
        num_classes = _num_classes[dataset]
        return ResNet_cifar(num_layers, num_classes, cfg)
