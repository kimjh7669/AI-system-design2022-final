import logging

from .func import *
from .quantizer import *


def quantizer(default_cfg, this_cfg=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)


except_name = [
'prop_feats.0',
'cls',
'bbox_x',
'bbox_y',
'bbox_w',
'bbox_h',
'bbox_x3d',
'bbox_y3d',
'bbox_z3d',
'bbox_w3d',
'bbox_h3d',
'bbox_l3d',
'bbox_rY3d',
'prop_feats_loc.0.group_conv',
'cls_loc.group_conv',
'bbox_x_loc.group_conv',
'bbox_y_loc.group_conv',
'bbox_w_loc.group_conv',
'bbox_h_loc.group_conv',
'bbox_x3d_loc.group_conv',
'bbox_y3d_loc.group_conv',
'bbox_z3d_loc.group_conv',
'bbox_w3d_loc.group_conv',
'bbox_h3d_loc.group_conv',
'bbox_l3d_loc.group_conv',
'bbox_rY3d_loc.group_conv',]

def find_modules_to_quantize(model, quan_scheduler, logger=None):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if name in except_name:
                continue
            if name in quan_scheduler.excepts:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight,
                                        quan_scheduler.excepts[name].weight),
                    quan_a_fn=quantizer(quan_scheduler.act,
                                        quan_scheduler.excepts[name].act)
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight),
                    quan_a_fn=quantizer(quan_scheduler.act)
                )
        elif name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        if full_name in except_name:
                            continue
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
