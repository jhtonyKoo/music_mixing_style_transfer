""" Utility file for trainers """
import os
import shutil
from glob import glob

import torch
import torch.distributed as dist



''' checkpoint functions '''
# saves checkpoint
def save_checkpoint(model, \
                        optimizer, \
                        scheduler, \
                        epoch, \
                        checkpoint_dir, \
                        name, \
                        model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_state = {    
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
        }
    checkpoint_path = os.path.join(checkpoint_dir,'{}_{}_{}.pt'.format(name, model_name, epoch))
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


# reload model weights from checkpoint file
def reload_ckpt(args, \
                network, \
                optimizer, \
                scheduler, \
                gpu, \
                model_name, \
                manual_reload_name=None, \
                manual_reload=False, \
                manual_reload_dir=None, \
                epoch=None, \
                fit_sefa=False):
    if manual_reload:
        reload_name = manual_reload_name
    else:
        reload_name = args.name
    if manual_reload_dir:
        ckpt_dir = manual_reload_dir + reload_name + "/ckpt/"
    else:
        ckpt_dir = args.output_dir + reload_name + "/ckpt/"
    temp_ckpt_dir = f'{args.output_dir}{reload_name}/ckpt_temp/'
    reload_epoch = epoch
    # find best or latest epoch
    if epoch==None:
        reload_epoch_temp = 0
        reload_epoch_ckpt = 0
        if len(os.listdir(temp_ckpt_dir))!=0:
            reload_epoch_temp = find_best_epoch(temp_ckpt_dir)
        if len(os.listdir(ckpt_dir))!=0:
            reload_epoch_ckpt = find_best_epoch(ckpt_dir)
        if reload_epoch_ckpt >= reload_epoch_temp:
            reload_epoch = reload_epoch_ckpt
        else:
            reload_epoch = reload_epoch_temp
            ckpt_dir = temp_ckpt_dir
    else:
        if os.path.isfile(f"{temp_ckpt_dir}{reload_epoch}/{reload_name}_{model_name}_{reload_epoch}.pt"):
            ckpt_dir = temp_ckpt_dir
    # reloading weight
    if model_name==None:
        resuming_path = f"{ckpt_dir}{reload_epoch}/{reload_name}_{reload_epoch}.pt"
    else:
        resuming_path = f"{ckpt_dir}{reload_epoch}/{reload_name}_{model_name}_{reload_epoch}.pt"
    if gpu==0:
        print("===Resume checkpoint from: {}===".format(resuming_path))
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(resuming_path, map_location=loc)
    start_epoch = 0 if manual_reload and not fit_sefa else checkpoint["epoch"]

    if manual_reload_dir is not None and 'parameter_estimation' in manual_reload_dir:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = 'module.' + k # add `module.`
            new_state_dict[name] = v
        network.load_state_dict(new_state_dict)
    else:
        network.load_state_dict(checkpoint["model"])
    if not manual_reload:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    if gpu==0:
        # print("=> loaded checkpoint '{}' (epoch {})".format(resuming_path, checkpoint['epoch']))
        print("=> loaded checkpoint '{}' (epoch {})".format(resuming_path, epoch))
    return start_epoch


# find best epoch for reloading current model
def find_best_epoch(input_dir):
    cur_epochs = glob("{}*".format(input_dir))
    return find_by_name(cur_epochs)


# sort string epoch names by integers
def find_by_name(epochs):
    int_epochs = []
    for e in epochs:
        int_epochs.append(int(os.path.basename(e)))
    int_epochs.sort()
    return (int_epochs[-1])


# remove ckpt files
def remove_ckpt(cur_ckpt_path_dir, leave=2):
    ckpt_nums = [int(i) for i in os.listdir(cur_ckpt_path_dir)]
    ckpt_nums.sort()
    del_num = len(ckpt_nums) - leave
    cur_del_num = 0
    while del_num > 0:
        shutil.rmtree("{}{}".format(cur_ckpt_path_dir, ckpt_nums[cur_del_num]))
        del_num -= 1
        cur_del_num += 1



''' multi-GPU functions '''

# gather function implemented from DirectCLR
class GatherLayer_Direct(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer_Direct.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


