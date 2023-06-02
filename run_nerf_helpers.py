import torch, math
# torch.autograd.set_detect_anomaly(True)
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np


# Misc
img2mse  = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x    : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b     = lambda x    : (255*np.clip(x,0,1)).astype(np.uint8)


"""Positional Encoding (section 5.1)"""
class Embedder:
    def __init__(self, **kwargs):
        # self.kwargs = {
        #                'include_input' : True,
        #                'input_dims'    : 3,
        #                'max_freq_log2' : multires-1,
        #                'num_freqs'     : multires,
        #                'log_sampling'  : True,
        #                'periodic_fns'  : [torch.sin, torch.cos]
        #               }.
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        input_dims = self.kwargs['input_dims']
        out_dim = 0
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Consider the original input in the final encoding result."""
        if self.kwargs['include_input']:
            # 'embed_fns[0]' = lambda x : x.
            embed_fns.append(lambda x : x)
            out_dim += input_dims
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Calculate the frequence band."""
        # 'self.kwargs['max_freq_log2']' = multires-1 = 'L'-1.
        # 'self.kwargs['num_freqs']'     = multires   = 'L'.
        max_freq = self.kwargs['max_freq_log2']
        N_freqs  = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 'torch.linspace(0., max_freq, steps=N_freqs)' = tensor([0., 1., ..., L-1.]).
            # 'freq_bands' = tensor([2**0., 2**1., ..., 2**L-1.]).
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        """Positional Encoding."""
        for freq in freq_bands:
            # 'self.kwargs['periodic_fns']': [torch.sin, torch.cos].
            for p_fn in self.kwargs['periodic_fns']:
                ## Where is phi!!!
                ## embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * math.pi * freq))
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += input_dims
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        # 'self.embed_fns': A list of lambda functions.
        # 'self.embed_fns[0]' = lambda x : x.
        # 'self.embed_fns[1:]' = lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq).
        self.embed_fns = embed_fns
        # 'self.out_dim' = 3 + 'L'*2*3 = 63 (for L = 10) / 27 (for L = 4).
        self.out_dim = out_dim


    """The 'Positional Encoding' function."""
    def embed(self, inputs):
        # Input : 'inputs'.shape = batchsize * 3.
        # Output: 'torch.cat([fn(inputs) for fn in self.embed_fns], -1)'.shape = batchsize * 63 / batchsize * 27.
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """
    Args:
        multires: 'L'=10 for the three coordinate values for position in "Positional Encoding".
                  'L'= 4 for the three unit vector components for viewing direction in "Positional Encoding".
        i: Set 0 for activating positional encoding, -1 for none.

    Returns:
        embed [Lambda function]: A lambda function that takes a numpy array / tensor of shape ['batchsize', 'input_dims'] as input
                                 and outputs a numpy array / tensor of shape ['batchsize', 'embedder_obj.out_dim'] as encoding results.
        embedder_obj.out_dim = 63 (for L = 10) / 27 (for L = 4).
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    # Deactivate 'Positional Encoding' for 'i == -1'.
    if i == -1:
        return nn.Identity(), 3
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    embed_kwargs = {
                    'include_input' : True,
                    'input_dims'    : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs'     : multires,
                    'log_sampling'  : True,
                    'periodic_fns'  : [torch.sin, torch.cos]
                   }
    embedder_obj = Embedder(**embed_kwargs)

    # 'x': Inputs of shape = batchsize * 3.
    # The lambda function 'embed' returns an encoding result
    # with shape = batchsize * embedder_obj.out_dim (3+60) / batchsize * embedder_obj.out_dim (3+24).
    embed = lambda x, eo=embedder_obj : eo.embed(x)

    return embed, embedder_obj.out_dim


"""Hierarchical Sampling (section 5.2)"""
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Args:
        bins: 'N_partitions' Partitions along each distance / line of the same length in one batch.
              Each distance / line of the same length in one batch is not necessarily evenly-spaced.
              Shape = batchsize * (number of partitions N_partitions + 1).
              Example: bins = tensor([[55.0000, 57.5000, 60.0000, 62.5000, 65.0000],
                                      [55.0000, 57.5000, 60.0000, 62.5000, 65.0000],
                                      [55.0000, 57.5000, 60.0000, 62.5000, 65.0000]]) with batchsize = 3 and N_partitions = 4.
                       "
                         N_partitions = 4
                         near = 55
                         far = 65
                         t_vals = torch.linspace(0., 1., steps=N_partitions+1)
                         bins = near * (1.-t_vals) + far * (t_vals)
                         bins = bins.expand([batchsize, N_partitions+1])
                       "
        weights: 'N_partitions' Weights for each partition along each  distance / line.
                 Shape = batchsize * number of partitions N_partitions (also number of weights).
                 Example: weights = tensor([[10., 15., 25., 15.],
                                            [20., 25., 18., 16.],
                                            [13., 11., 11., 16.]]) with batchsize = 3 and N_partitions = 4.
        N_samples: Number of additional fine samples per ray (Nf),
                   which are only passed to the fine network.
                   Example: N_samples = 8.

    Returns:
        samples: Relative positions of samples along the distance / line axis.
                 Shape = batchsize * 'N_samples'.
                 Example: samples = tensor([[55.0000, 57.3214, 58.9286, 60.2857, 61.2143, 62.1429, 63.4524, 65.0000],
                                            [55.0000, 56.4107, 57.7571, 58.8857, 60.0198, 61.5873, 63.2366, 65.0000],
                                            [55.0000, 56.4011, 57.8571, 59.5130, 61.1688, 62.7232, 63.8616, 65.0000]]),
                          where 'det' = True.
                 Example: samples = tensor([[59.6072, 60.3283, 62.3404, 61.2213, 60.5227, 62.2906, 56.1814, 59.8515],
                                            [57.6756, 64.5658, 57.4879, 64.6510, 58.8377, 61.1371, 63.8077, 58.9304],
                                            [55.0990, 59.9843, 64.9838, 64.5204, 58.3890, 62.9700, 60.9688, 58.8586]]),
                          where 'det' = False.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Get PDF."""
    weights = weights + 1e-5 # To prevent nans.
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # 'cdf'.shape = batchsize * (number of partitions N_partitions + 1).
    # Example for the first distance / line: tensor([0.0000, 0.1538, 0.3846, 0.7692, 1.0000]).
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Take samples with values between 0 and 1."""
    if det:
        # 'perturb' = 0, then 'det = (perturb==0)' = 1.
        # Uniform partitions through points 'u' between 0 and 1.
        # Example for 'u': tensor([[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0000],
        #                          [0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0000],
        #                          [0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0000]]),
        # where 'N_samples' (number of additional fine samples per ray: Nf) = 8.
        # 'u'.shape = batchsize *  N_samples (number of additional fine samples per ray: Nf).
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 'perturb' = 1, then 'det = (perturb==0)' = 0.
        # Stratified random points 'u' in time between 0 and 1. → Each ray is sampled at stratified random points in time.
        # The function 'torch.rand(size)' outputs a set of random numbers with defined size extracted from the uniform distribution of interval [0, 1).
        # Example for 'u': tensor([[0.0090, 0.2698, 0.3854, 0.7626, 0.6135, 0.0376, 0.2137, 0.2583],
        #                          [0.0924, 0.3453, 0.8902, 0.5376, 0.6115, 0.0242, 0.8310, 0.7446],
        #                          [0.2363, 0.8845, 0.2242, 0.9936, 0.2180, 0.6289, 0.8530, 0.7781]]),
        # where 'N_samples' (number of additional fine samples per ray: Nf) = 8.
        # 'u'.shape = batchsize *  N_samples (number of additional fine samples per ray: Nf).
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        ## u = torch.sort(u, dim=-1).values

    """Pytest, overwrite u with numpy's fixed random numbers."""
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 'torch.Tensor.contiguous()' returns a tensor with identical data but contiguous in memory.
    # It is called before 'torch.searchsorted()' for performance concern.
    u = u.contiguous()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Find the indice of the partitions along the 'CDF' axis,
       where sampled points are located.
    """
    # Example for 'cdf': tensor([[0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                            [0.0000, 0.1453, 0.2991, 0.6325, 0.8034, 0.8974, 1.0000],
    #                            [0.0000, 0.2746, 0.3990, 0.5803, 0.7668, 0.9067, 1.0000]]).
    # Example for 'u': tensor([[0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
    #                          [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
    #                          [0.0000, 0.2500, 0.5000, 0.7500, 1.0000]]).
    # For the first ray:                | For the second ray:               | For the third ray:
    # below:                 above:     | below:                 above:     | below:                 above:
    # 0.0000 (0) <= 0.0000 < 0.0756 (1) | 0.0000 (0) <= 0.0000 < 0.1453 (1) | 0.0000 (0) <= 0.0000 < 0.2746 (1)
    # 0.1977 (2) <= 0.2500 < 0.4477 (3) | 0.1453 (1) <= 0.2500 < 0.2991 (2) | 0.0000 (0) <= 0.2500 < 0.2746 (1)
    # 0.4477 (3) <= 0.5000 < 0.6453 (4) | 0.2991 (2) <= 0.5000 < 0.6325 (3) | 0.3990 (2) <= 0.5000 < 0.5803 (3)
    # 0.6453 (4) <= 0.7500 < 0.9070 (5) | 0.6325 (3) <= 0.7500 < 0.8034 (4) | 0.5803 (3) <= 0.7500 < 0.7668 (4)
    # 1.0000 (6) <= 1.0000 < X.XXXX (7) | 1.0000 (6) <= 1.0000 < X.XXXX (7) | 1.0000 (6) <= 1.0000 < X.XXXX (7)
    # Example for 'inds': tensor([[1, 3, 4, 5, 7],
    #                             [1, 2, 3, 4, 7],
    #                             [1, 1, 3, 4, 7]]),
    # where 'N_samples' (number of additional fine samples per ray: Nf) = 5.
    # 'inds'.shape = batchsize *  N_samples (number of additional fine samples per ray: Nf).
    inds = torch.searchsorted(cdf, u, right=True)
    # Example for 'below': tensor([[0, 2, 3, 4, 6],
    #                              [0, 1, 2, 3, 6],
    #                              [0, 0, 2, 3, 6]]),
    # where 'N_samples' (number of additional fine samples per ray: Nf) = 5.
    # 'below'.shape = batchsize *  N_samples (number of additional fine samples per ray: Nf).
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    # Example for 'above': tensor([[1, 3, 4, 5, 6],
    #                              [1, 2, 3, 4, 6],
    #                              [1, 1, 3, 4, 6]]),
    # where 'N_samples' (number of additional fine samples per ray: Nf) = 5.
    # 'above'.shape = batchsize *  N_samples (number of additional fine samples per ray: Nf).
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    # Example for 'inds_g': tensor([[[0, 1],
    #                                [2, 3],
    #                                [3, 4],
    #                                [4, 5],
    #                                [6, 6]],
    #
    #                               [[0, 1],
    #                                [1, 2],
    #                                [2, 3],
    #                                [3, 4],
    #                                [6, 6]],
    #
    #                               [[0, 1],
    #                                [0, 1],
    #                                [2, 3],
    #                                [3, 4],
    #                                [6, 6]]]),
    # where 'N_samples' (number of additional fine samples per ray: Nf) = 5.
    # 'inds_g' with shape = num_rays (number of rays: batchsize) *  N_samples (number of additional fine samples per ray: Nf) * 2.
    inds_g = torch.stack([below, above], -1)

    """Find the partitions along the 'CDF' and the diatance / line axis respectively
       according to the found indice above, where sampled points are located.
    """
    # 'matched_shape' = [batchsize, Nf, (number of partitions N_partitions + 1)].
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    # Find the partitions along the 'CDF' axis according to the found indice 'inds_g' above.
    # 'cdf.unsqueeze(0)'.shape = 1 * batchsize * (Nc - 1).
    # 'cdf.unsqueeze(2)'.shape = batchsize * (Nc - 1) * 1.
    # 'cdf.unsqueeze(1)'.shape = batchsize * 1 * (Nc - 1).
    # 'cdf.unsqueeze(1).expand([batchsize, Nf, Nc - 1])'.shape = batchsize * Nf * (Nc - 1).
    # Example for 'cdf.unsqueeze(1).expand([batchsize, Nf, Nc - 1])': tensor([[[0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                                                                          [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                                                                          [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                                                                          [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                                                                          [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000]],
    #
    #                                                                         [[0.0000, 0.1453, 0.2991, 0.6325, 0.8034, 0.8974, 1.0000],
    #                                                                          [0.0000, 0.1453, 0.2991, 0.6325, 0.8034, 0.8974, 1.0000],
    #                                                                          ......,
    # where 'Nf' = 5, 'Nc' = 8.
    # Example for 'cdf_g': from [[0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                            [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                            [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                            [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000],
    #                            [0.0000, 0.0756, 0.1977, 0.4477, 0.6453, 0.9070, 1.0000]]
    #    along dim=2-1=1 sample [[0, 1],  → 00, 01 → 00, 01 → 0.0000, 0.0756
    #                            [2, 3],  → 10, 11 → 12, 13 → 0.1977, 0.4477
    #                            [3, 4],  → 20, 21 → 23, 24 → 0.4477, 0.6453
    #                            [4, 5],  → 30, 31 → 34, 35 → 0.6453, 0.9070
    #                            [6, 6]]  → 40, 41 → 46, 46 → 1.0000, 1.0000,
    # 'cdf_g': tensor([[[0.0000, 0.0756],
    #                   [0.1977, 0.4477],
    #                   [0.4477, 0.6453],
    #                   [0.6453, 0.9070],
    #                   [1.0000, 1.0000]],
    #                  ......). 
    # 'cdf_g'.shape = 'inds_g'.shape = batchsize * N_samples (Nf) * 2.
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Find the partitions along the distance / line axis according to the found indice 'inds_g' above.
    # 'bins'.shape = batchsize * (number of partitions N_partitions + 1).
    # 'bins.unsqueeze(1)'.shape = batchsize * 1 * (number of partitions N_partitions + 1).
    # 'bins.unsqueeze(1).expand(matched_shape)'.shape = batchsize * Nf * (number of partitions N_partitions + 1).
    # Example for 'bins.unsqueeze(1).expand(matched_shape)': tensor([[[30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857]],
    #                                                                [[30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                                                                 ......,
    # where 'Nf' = 5.
    # Example for 'bins_g': from [[30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                             [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                             [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                             [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857],
    #                             [30.7143, 32.1429, 33.5714, 35.0000, 36.4286, 37.8571, 39.2857]]
    #     along dim=2-1=1 sample [[0, 1],   → 00, 01 → 00, 01 → 30.7143, 32.1429
    #                             [2, 3],   → 10, 11 → 12, 13 → 33.5714, 35.0000
    #                             [3, 4],   → 21, 22 → 23, 24 → 35.0000, 36.4286
    #                             [4, 5],   → 31, 32 → 34, 35 → 36.4286, 37.8571
    #                             [6, 6]]   → 41, 42 → 46, 46 → 39.2857, 39.2857,
    # 'bins_g': tensor([[[30.7143, 32.1429],
    #                    [33.5714, 35.0000],
    #                    [35.0000, 36.4286],
    #                    [36.4286, 37.8571],
    #                    [39.2857, 39.2857]],
    #                   ......).
    # 'bins_g'.shape = 'inds_g'.shape = batchsize * N_samples (Nf) * 2.
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Calculate new sample locations along the distance / line axis."""
    # 'denom' = tensor([[0.0756, 0.2500, 0.1976, 0.2617, 0.0000],
    #                  ......)
    denom = (cdf_g[...,1]-cdf_g[...,0])
    # 'torch.ones_like(denom)': tensor([[1., 1., 1., 1., 1.],
    #                                  ......)
    # 'denom' = tensor([[0.0756, 0.2500, 0.1976, 0.2617, 1.0000],
    #                  ......)
    # 'denom'.shape = batchsize * N_samples (Nf).
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) # To prevent nans.
    # 'u'.shape = batchsize * N_samples (Nf).
    # 'cdf_g'.shape = batchsize * N_samples (Nf) * 2.
    # 'cdf_g[...,0]'.shape = batchsize * N_samples (Nf).
    # 't'.shape = batchsize * N_samples (Nf).
    t = (u-cdf_g[...,0])/denom
    # 'bins_g[...,0]'.shape = batchsize * N_samples (Nf).
    # 't * (bins_g[...,1]-bins_g[...,0])'.shape = batchsize * N_samples (Nf).
    # 'samples'.shape = batchsize * N_samples (Nf).
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


"""Generate rays."""
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """
    First generate rays from the camera origin to each pixel on the image plane ('H', 'W' and 'K')
    and then describe rays and their origin in the world coordinate system.
    (according to different camera poses 'c2w': 'c2w[:3,:3]' and 'c2w[:3,-1]').

    Args:
        H   (Int)        : Height of the image plane in pixels.
        W   (Int)        : Width of the image plane in pixels.
        K   (Numpy array): Camera intrinsic matrix.
        c2w (Numpy array): A camera pose according to the world coordinate system / Camera-to-world transformation matrix.
            Shape = 3 * 4.
            c2w[:3,:3]: Rotation matrix from the camera coordinate system to world coordinate system.
            c2w[:3,-1]: Translation matrix from the camera coordinate system to world coordinate system,
                        which points from the origin of the world coordinate system to the origin of the camera coordinate system.

    Returns:
        A tuple of two numpy arrays.
        rays_o (Numpy array of shape: 'H' ('y') * 'W' ('x') * 3): Ray origin positions of all the rays in the world coordinate system.
        rays_d (Numpy array of shape: 'H' ('y') * 'W' ('x') * 3): Ray directions in the world coordinate system,
                                                                  which is 'z-axis normalized' in the camera coordinate system.
    """

    # Create a 2D grid of pixels / points [0, H) * [0, W) of the image plane,
    # where numpy arrays 'i' and 'j' are respectively the x- and y pixel coordinates of the grid points.
    # Example: 'i' = numpy.array([[0., 1., 2.],
    #                             [0., 1., 2.],
    #                             [0., 1., 2.],
    #                             [0., 1., 2.],
    #                             [0., 1., 2.]], dtype=float32), 'i.shape' = 5 * 3,
    #          'j' = numpy.array([[0., 0., 0.],
    #                             [1., 1., 1.],
    #                             [2., 2., 2.],
    #                             [3., 3., 3.],
    #                             [4., 4., 4.]], dtype=float32), 'j.shape' = 5 * 3.
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # A ray is defined both by an origin, which lies at the origin of the camera coordinate system (principle point),
    # and its direction 'dir' that connects the origin to a pixel on the image plane.
    # The coordinates of the pixels on the image plane are [i - W/2, -(j - H/2), -focal].T in the camera coordinate system,
    # which can also be described as [i-K[0][2], -(j-K[1][2]), -K[0][0]].T with the camera intrinsic matrix 'K'.
    # The 'z-axis normalized' ray direction per 'i' and 'j' is: [(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -1].T in the camera coordinate system.
    # 'dirs.shape' = 'H' ('y') * 'W' ('x') * 3 (3 components to describe a 'normalized' vector in the camera coordinate system).
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate 'z-axis normalized' ray directions from camera frame to the world frame.
    # 'dirs[..., np.newaxis, :].shape' = 5 ('y') * 3 ('x') * 1 * 3 ('3 components to describe a 'normalized' vector).
    # 'c2w[:3,:3]': Rotation matrix from the camera coordinate system to world coordinate system.
    # '(dirs[..., np.newaxis, :] * c2w[:3,:3]).shape' = 5 ('y') * 3 ('x') * 3 * 3.
    # 'rays_d.shape' = 'H' ('y') * 'W' ('x') * 3 (3 components to describe a 'normalized' vector in the world coordinate system).
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) ## Trick!
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 'rays_o.shape' = 'H' ('y') * 'W' ('x') * 3 (3 components to describe a 'normalized' vector in the world coordinate system).
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d


"""Instantiate a NeRF's MLP network."""
class NeRF(nn.Module):

    """Construct MLP network structure."""
    def __init__(self, D=8,
                       W=256,
                       input_ch=63,
                       input_ch_views=27,
                       output_ch=4,
                       skips=[4],
                       use_viewdirs=True):
        super(NeRF, self).__init__()

        """Read inputs."""
        # Number of (the first) fully-connected layers for processing the input positions. 
        self.D = D
        # Number of channels per fully-connected layer for processing the input positions.
        self.W = W
        # Number of input position channels.
        self.input_ch = input_ch
        # Number of input direction channels.
        self.input_ch_views = input_ch_views
        # To which layer's activation we concatenate the input position
        # by including a skip connection.
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        """Construct fully-connected layers for processing the input positions.
           Create 'self.D' fully-connected layers and
           include a skip connection to concatenate the input position.
        """
        # 'nn.ModuleList()' takes in a list of layer structures as input-parameter.
        # 'enumerate()' + 'nn.ModuleList()'.
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
            )
        # ModuleList(
        #     (0): Linear(in_features=63,       out_features=256, bias=True)
        #     (1): Linear(in_features=256,      out_features=256, bias=True)
        #     (2): Linear(in_features=256,      out_features=256, bias=True)
        #     (3): Linear(in_features=256,      out_features=256, bias=True)
        #     (4): Linear(in_features=256,      out_features=256, bias=True)
        #     (5): Linear(in_features=(63+256), out_features=256, bias=True)
        #     (6): Linear(in_features=256,      out_features=256, bias=True)
        #     (7): Linear(in_features=256,      out_features=256, bias=True)
        #   )

        """Generate one single fully-connected layer for processing the input directions."""
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        # ModuleList(
        #     (0): Linear(in_features=(256+27), out_features=128, bias=True)
        #   )

        if use_viewdirs:

            """Generate one single fully-connected layer to output the volume density 'sigma'
               and a 256-dimensional festure vector respectively.
            """
            # "An additional layer outputs the volume density 'sigma'
            #  (which is rectified using a ReLU to ensure that the output volume density is nonnegative)
            #  and a 256-dimensional feature vector.
            # "
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)

            """Generate one single fully-connected layer to output three color values
               depending on input position and viewing direction.
            """
            # "A final layer (with a sigmoid activation) outputs
            #  the emitted RGB radiance at position 'x',
            #  as viewed by a ray with direction 'd'.
            # "
            self.rgb_linear = nn.Linear(W//2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)


    def forward(self, x):
        """
        Forward-Propagation through the network.

        Args:
            x [Tensor of shape: Number_of_points * (3+60+3+24)]: Combined results of the positional encoding of
                                                                 the input points' positions and their corresponding viewing directions.

        Returns:
            outputs [Tensor of shape: Number_of_points * (3+1)]: Network's prediction for each input point
                                                                 consisting of color values depending on point position and its corresponding viewing direction
                                                                 and volume density depending on point position.
        """

        """Split the inputs into the input positions and viewing directions."""
        # 'input_pts.shape'   = Number_of_points * (3+60).
        # 'input_views.shape' = Number_of_points * (3+24).
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        """Forward-Propagate the input positions
           through the first 'self.D' fully-connected layers.
        """
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # Include a skip connection to concatenate the input position
            # with the 'self.skips'-th layer's output.
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:

            """Get the output of the volume density 'sigma'
               after being rectified using a ReLU to ensure its nonnegativity."""
            alpha = self.alpha_linear(h)
            alpha = F.relu(alpha)

            """Get the output of a 256-dimensional festure vector.
               This feature vector is first concatenated with the positional encoding of
               the input viewing direction in sequence and then fed into a single fully-connected layer.
            """
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            """Get the output of three color values ranging from 0 to 1 
               after using the 'Sigmoid' activation function."""
            rgb = self.rgb_linear(h)
            rgb = torch.sigmoid(rgb)

            """Combine the results of color values and volume density."""
            outputs = torch.cat([rgb, alpha], -1)

        else:
            outputs = self.output_linear(h)

        return outputs


    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from the world coordinate system to the NDC space.
    Normalized device coordinates (NDC): A cube with sides [-1, 1] in each axis.

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    (See https://github.com/bmild/nerf/issues/18.)

    Args:
        H      [Int]  : Height in pixels.
        W      [Int]  : Width in pixels.
        focal  [Float]: Focal length of pinhole camera.
        near   [Float]: The depths of the near plane.
                        The near plane is always at 1.0.
                        'near' and 'far' in NDC are always 0.0 and 1.0.
                        (See https://github.com/bmild/nerf/issues/34.)
        rays_o [Tensor of shape: batch_size * 3]: Camera origin in the world coordinate system.
        rays_d [Tensor of shape: batch_size * 3]: Ray direction in the world coordinate system.

    Returns:
        rays_o [Tensor of shape: batch_size * 3]: Camera origin in the NDC space.
        rays_d [Tensor of shape: batch_size * 3]: Ray direction in the NDC space.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Shift ray origins to the near clipping plane."""
    # 'rays_o[...,2]': Absolute z-axis position of ray origins in the world coordinate system.
    # 'rays_d[...,2]': Absolute z-axis component of ray directions in the world coordinate system,
    #                  which are previously normalized to -1. in the camera coordinate system.
    # 't'            : Relative distances to the ray origins along the z-axis in the camera coordinate system.
    # '(-1) * near'  : Absolute z-axis position of the near clipping plane in the world coordinate system.
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Projection."""
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


if __name__=='__main__':

    model = NeRF()
    input = torch.randn(30, 90)
    output = model(input)
    print(output.shape)
