import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch.nn.functional as F
import torch

from time import time

def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):

        b, c, h, w = x.size()

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        x = x.unfold(2, t + self.pad*2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad*2, (w+self.pad*2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y


class RPN(nn.Module):

    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.s_time_list = []
        self.m_time_list = []
        self.l_time_list = []
        
        self.base = base

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        self.num_rows = int(min(conf.bins, calc_output_size(conf.test_scale, conf.feat_stride)))
        
        self.base = torch.nn.Sequential(*(list(self.base.children())[:-2]))
        # self.base[7][0].downsample[0].stride = (1,1)
        self.upconv = torch.nn.ConvTranspose2d(2048, 1024, 2, stride=2, padding=0)
        self.prop_feats = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.prop_feats_loc = nn.Sequential(
            LocalConv2d(self.num_rows, 1024, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # outputs
        self.cls_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

        # bbox 2d
        self.bbox_x_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.cls_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.bbox_x_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_y_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_w_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_h_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.bbox_x3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_y3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_z3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_w3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_h3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_l3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_rY3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds

    def forward(self, x):
        s_time = time()

        batch_size = x.size(0)

        # resnet
        x = self.base(x)
        x = self.upconv(x)
        m_time = time()
        x = x.detach()
        prop_feats = self.prop_feats(x)
        prop_feats_loc = self.prop_feats_loc(x)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        cls_loc = self.cls_loc(prop_feats_loc)

        # bbox 2d
        bbox_x_loc = self.bbox_x_loc(prop_feats_loc)
        bbox_y_loc = self.bbox_y_loc(prop_feats_loc)
        bbox_w_loc = self.bbox_w_loc(prop_feats_loc)
        bbox_h_loc = self.bbox_h_loc(prop_feats_loc)

        # bbox 3d
        bbox_x3d_loc = self.bbox_x3d_loc(prop_feats_loc)
        bbox_y3d_loc = self.bbox_y3d_loc(prop_feats_loc)
        bbox_z3d_loc = self.bbox_z3d_loc(prop_feats_loc)
        bbox_w3d_loc = self.bbox_w3d_loc(prop_feats_loc)
        bbox_h3d_loc = self.bbox_h3d_loc(prop_feats_loc)
        bbox_l3d_loc = self.bbox_l3d_loc(prop_feats_loc)
        bbox_rY3d_loc = self.bbox_rY3d_loc(prop_feats_loc)

        cls_ble = self.sigmoid(self.cls_ble)

        # bbox 2d
        bbox_x_ble = self.sigmoid(self.bbox_x_ble)
        bbox_y_ble = self.sigmoid(self.bbox_y_ble)
        bbox_w_ble = self.sigmoid(self.bbox_w_ble)
        bbox_h_ble = self.sigmoid(self.bbox_h_ble)

        # bbox 3d
        bbox_x3d_ble = self.sigmoid(self.bbox_x3d_ble)
        bbox_y3d_ble = self.sigmoid(self.bbox_y3d_ble)
        bbox_z3d_ble = self.sigmoid(self.bbox_z3d_ble)
        bbox_w3d_ble = self.sigmoid(self.bbox_w3d_ble)
        bbox_h3d_ble = self.sigmoid(self.bbox_h3d_ble)
        bbox_l3d_ble = self.sigmoid(self.bbox_l3d_ble)
        bbox_rY3d_ble = self.sigmoid(self.bbox_rY3d_ble)

        # blend
        cls = (cls * cls_ble) + (cls_loc * (1 - cls_ble))

        bbox_x = (bbox_x * bbox_x_ble) + (bbox_x_loc * (1 - bbox_x_ble))
        bbox_y = (bbox_y * bbox_y_ble) + (bbox_y_loc * (1 - bbox_y_ble))
        bbox_w = (bbox_w * bbox_w_ble) + (bbox_w_loc * (1 - bbox_w_ble))
        bbox_h = (bbox_h * bbox_h_ble) + (bbox_h_loc * (1 - bbox_h_ble))

        bbox_x3d = (bbox_x3d * bbox_x3d_ble) + (bbox_x3d_loc * (1 - bbox_x3d_ble))
        bbox_y3d = (bbox_y3d * bbox_y3d_ble) + (bbox_y3d_loc * (1 - bbox_y3d_ble))
        bbox_z3d = (bbox_z3d * bbox_z3d_ble) + (bbox_z3d_loc * (1 - bbox_z3d_ble))
        bbox_h3d = (bbox_h3d * bbox_h3d_ble) + (bbox_h3d_loc * (1 - bbox_h3d_ble))
        bbox_w3d = (bbox_w3d * bbox_w3d_ble) + (bbox_w3d_loc * (1 - bbox_w3d_ble))
        bbox_l3d = (bbox_l3d * bbox_l3d_ble) + (bbox_l3d_loc * (1 - bbox_l3d_ble))
        bbox_rY3d = (bbox_rY3d * bbox_rY3d_ble) + (bbox_rY3d_loc * (1 - bbox_rY3d_ble))

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.reshape(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.reshape(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)

        l_time = time()
        self.s_time_list.append(s_time)
        self.m_time_list.append(m_time)
        self.l_time_list.append(l_time)
        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):
    train = phase.lower() == 'train'

    densenet121 = models.resnet50(pretrained=train)

    num_cls = len(conf['lbls']) + 1
    num_anchors = conf['anchors'].shape[0]

    # make network
    rpn_net = RPN(phase, densenet121, conf)

    dst_weights = rpn_net.state_dict()
    dst_keylist = list(dst_weights.keys())

    src_weights = torch.load('/media/hdd/jhkim/git/M3D-RPN/output/kitti_3d_multi_main_resnet50/weights/model_100000_pkl')
    
    
    
    dst_keys = list(dst_weights.keys())
    src_keys = list(src_weights.keys())
    # copy keys without module
    for key in src_keys:
        src_weights[key.replace('module.', '')] = src_weights[key]
        del src_weights[key]
    src_keys = list(src_weights.keys())

    # remove keys not in dst
    for key in src_keys:
        if key not in dst_keys: del src_weights[key]
        
    rpn_net.load_state_dict(src_weights)
    import sys
    sys.path.append('/media/hdd/jhkim/git/M3D-RPN/')

    import quan
    import util
    from pathlib import Path
    import logging
    quan_logger = logging.getLogger('/media/hdd/jhkim/git/M3D-RPN/output/moudule_info')
    quan_stream_handler = logging.StreamHandler()
    quan_log_handler = logging.FileHandler('/media/hdd/jhkim/git/M3D-RPN/output/moudule_info.log')
    quan_logger.addHandler(quan_stream_handler)
    quan_logger.addHandler(quan_log_handler)
    
    script_dir = Path('/media/hdd/jhkim/git/M3D-RPN/')
    args_q = util.get_config(default_file=script_dir / 'config_16x16.yaml')
    modules_to_replace = quan.find_modules_to_quantize(rpn_net, args_q.quan, logger=quan_logger)
    rpn_net = quan.replace_module_by_names(rpn_net, modules_to_replace)

    if train:
        rpn_net.train()
    else:
        rpn_net.eval()

    return rpn_net
