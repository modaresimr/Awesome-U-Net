from fb import *
import torch
import torch.nn.functional as F
def bases_list(ks, num_bases):
    len_list = ks // 2
    b_list = []
    for i in range(len_list):
        kernel_size = (i+1)*2+1
        normed_bases, _, _ = calculate_FB_bases(i+1)
        normed_bases = normed_bases.transpose().reshape(-1, kernel_size, kernel_size).astype(np.float32)[:num_bases, ...]

        pad = len_list - (i+1)
        bases = torch.Tensor(normed_bases)
        bases = F.pad(bases, (pad, pad, pad, pad, 0, 0)).view(num_bases, ks*ks)
        b_list.append(bases)
        print(kernel_size,bases.shape)
    return torch.cat(b_list, 0)

x=bases_list(9,6)
print(x.shape)
