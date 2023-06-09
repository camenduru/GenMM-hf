import torch    
import torch.nn as nn

from .utils import extract_patches, combine_patches, efficient_cdist, get_NNs_Dists

def make_criteria(conf):
    if conf['type'] == 'PatchCoherentLoss':
        return PatchCoherentLoss(conf['patch_size'], stride=conf['stride'], loop=conf['loop'], coherent_alpha=conf['coherent_alpha'])
    elif conf['type'] == 'SWDLoss':
        raise NotImplementedError('SWDLoss is not implemented')
    else:
        raise ValueError('Invalid criteria: {}'.format(conf['criteria']))

class PatchCoherentLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, loop=False, coherent_alpha=None, cache=False):
        super(PatchCoherentLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.loop = loop
        self.coherent_alpha = coherent_alpha
        assert self.stride == 1, "Only support stride of 1"
        # assert self.patch_size % 2 == 1, "Only support odd patch size"
        self.cache = cache
        if cache:
            self.cached_data = None

    def forward(self, X, Ys, dist_wrapper=None, ext=None, return_blended_results=False):
        """For each patch in input X find its NN in target Y and sum the their distances"""
        assert X.shape[0] == 1, "Only support batch size of 1"
        dist_fn = lambda X, Y: dist_wrapper(efficient_cdist, X, Y) if dist_wrapper is not None else efficient_cdist(X, Y)

        x_patches = extract_patches(X, self.patch_size, self.stride, loop=self.loop)

        if not self.cache or self.cached_data is None:
            y_patches = []
            for y in Ys:
                y_patches += [extract_patches(y, self.patch_size, self.stride, loop=False)]
            y_patches = torch.cat(y_patches, dim=1)
            self.cached_data = y_patches
        else:
            y_patches = self.cached_data
        
        nnf, dist = get_NNs_Dists(dist_fn, x_patches.squeeze(0), y_patches.squeeze(0), self.coherent_alpha)

        if return_blended_results:
            return combine_patches(X.shape, y_patches[:, nnf, :], self.patch_size, self.stride, loop=self.loop), dist.mean()
        else:
            return dist.mean()
    
    def clean_cache(self):
        self.cached_data = None