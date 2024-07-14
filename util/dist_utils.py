import torch
import torch.distributed as dist


# utils
@torch.no_grad()
def get_rank():
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    else:
        return dist.get_rank()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output