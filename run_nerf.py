import os, sys, imageio, json, random, time, torch, configargparse
import numpy               as np
import torch.nn            as nn
import torch.nn.functional as F
import matplotlib.pyplot   as plt
from tqdm                  import tqdm, trange

from run_nerf_helpers      import *

# Introduce datasets for training and testing.
from load_llff       import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender    import load_blender_data
from load_LINEMOD    import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.

    Args:
        fn    [Object]: The NeRF-network (MLP) for predicting RGB and density at each point in space.
        chunk [Int]   : Number of points sent through the network in parallel,
                        decrease if running out of memory.

    Returns:
        ret [Function]: First partition the total number of input points into minibatches of points,
                        then feed these minibatches of points in sequence into the NeRF's MLP network
                        by using the network's 'forward()' function to avoid OOM.
                        (instead of feeding the total number of points into the network)
                        At last, combine the results of each minibatches together
                        and get the final network's prediction for color values and volume density
                        for all the input points of shape [Number_of_points, (rgb: 3 + volume density: 1)].
    """

    if chunk is None:
        return fn


    def ret(inputs):
        """
        First partition the total number of input points into minibatches of points,
        then feed these minibatches of points in sequence into the NeRF's MLP network
        by using the network's 'forward()' function to avoid OOM.
        (instead of feeding the total number of points into the network)
        At last, combine the results of each minibatches together
        and get the final network's prediction for color values and volume density
        for all the input points of shape [Number_of_points, (rgb: 3 + volume density: 1)].

        Args:
            inputs [Tensor of shape: Number_of_points * (3+60+3+24)]: The results of positions and directions in a higher dimensional space
                                                                      after applying the 'Positional Encoding' functions,
                                                                      where 'Number_of_points' = N_rays * N_samples in practice.

        Returns:
            [Tensor of shape: Number_of_points * (rgb: 3 + volume density: 1)]: The network's prediction
                                                                                for color values and volume density
                                                                                for each input point.
        """
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    Prepare inputs and apply network 'fn'.

    Args:
        inputs       [Tensor of shape: N_rays * N_samples * 3]: Absolute sample positions along each ray in a set of 'N_rays' rays
                                                                described in the world coordinate system / NDC space.
        viewdirs     [Tensor of shape: N_rays * 3]            : Normalized viewing directions per ray in a set of 'N_rays' rays
                                                                described in the world coordinate system / NDC space.
        fn           [Object]                                 : The NeRF's MLP network for predicting RGB and density at each point in space.
        embed_fn     [Function]                               : A lambda function.
                                                                Input: Absolute sample positions along each ray of shape [batchsize, 3]
                                                                       described in the world coordinate system / NDC space.
                                                                Output: An encoding result of shape [batchsize, embedder_obj.out_dim (3+60)].
        embeddirs_fn [Function]                               : A lambda function.
                                                                Input: Normalized viewing directions per ray of shape [batchsize, 3]
                                                                       described in the world coordinate system / NDC space.
                                                                Output: An encoding result of shape [batchsize, embedder_obj.out_dim (3+24)].
        netchunk     [Int]                                    : Number of points sent through the network in parallel,
                                                                decrease if running out of memory.

    Returns:
        outputs [Tensor of shape: N_rays * N_samples * 4]: The network's prediction for color values and volume density
                                                           for each input point per ray in a set of 'N_rays' rays.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Apply 'Positional Encoding' functions to map the inputs
       of positions and directions into a higher dimensional space."""
    # 'inputs_flat.shape' = (N_rays * N_samples) * 3.
    # 'N_rays * N_samples': The number of points in a set of 'N_rays' rays.
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # 'embedded.shape' = (N_rays * N_samples) * (3+60).
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        # 'viewdirs[:,None].shape' = N_rays * 1 * 3.
        # 'input_dirs.shape' = N_rays * N_samples * 3.
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        # 'input_dirs_flat.shape' = (N_rays * N_samples) * 3.
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 'embedded_dirs.shape' = (N_rays * N_samples) * (3+24).
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 'embedded.shape' = (N_rays * N_samples) * (3+60+3+24).
        embedded = torch.cat([embedded, embedded_dirs], -1)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Feed the encoded inputs of each point's position and direction
       into the NeRF's MLP network and get predictions for color values and volume density."""
    # 'outputs_flat' [Tensor of shape: Number_of_points * (rgb: 3 + volume density: 1)]:
    # The network's prediction for color values and volume density for each input point.
    outputs_flat = batchify(fn, netchunk)(embedded)
    # 'list(inputs.shape[:-1])' = [N_rays, N_samples].
    # '[outputs_flat.shape[-1]]' = [3+1].
    # 'outputs.shape' = N_rays * N_samples * (3+1).
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def create_nerf(args):
    """
    Instantiate NeRF's MLP model.

    Args:
        args.multires       [Int]  : 'L'=10 for the three coordinate values for position in "Positional Encoding".
        args.multires_views [Int]  : 'L'= 4 for the three unit vector components for viewing direction in "Positional Encoding".
        args.use_viewdirs   [Bool] : If True, feed viewing directions to the network for view-dependent apprearance
                                     (use full 5D input instead of 3D).
        args.i_embed        [Int]  : Set 0 for activating positional encoding, -1 for none.
        args.N_importance   [Int]  : Number of additional fine samples per ray.
        args.netdepth       [Int]  : Number of layers in the coarse network.
        args.netwidth       [Int]  : Number of channels per layer in the coarse network.
        args.netdepth_fine  [Int]  : Number of layers in the fine network.
        args.netwidth_fine  [Int]  : Number of channels per layer in the fine network.
        args.netchunk       [Int]  : Number of points sent through the network in parallel, decrease if running out of memory.
        args.lrate          [Float]: Initial learning rate.
        args.ft_path        [Str]  : Numpy file path of specific weights to reload for the coarse network.
        args.no_reload      [Bool] : If True, do not reload weights from saved checkpoints.
        args.lindisp        [Bool] : If True, sample linearly in inverse depth (disparity) rather than in depth.

    Returns:
        render_kwargs_train [Dictionary]: Save necessary arguments for training.
        render_kwargs_test  [Dictionary]: Save necessary arguments for testing.
        start               [Int]       : The current iteration step.
        grad_vars           [List]      : A list of network parameters from both the NeRF's MLP coarse and fine networks.
        optimizer           [Object]    : Optimizer.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Positional Encoding for positions."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    """Positional Encoding for directions."""
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    else:
        embeddirs_fn = None
        input_ch_views = 0

    """Instantiate a NeRF's MLP coarse model."""
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth,
                 W=args.netwidth,
                 input_ch=input_ch,
                 input_ch_views=input_ch_views,
                 output_ch=output_ch,
                 skips=skips,
                 use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    """Instantiate a NeRF's MLP fine model."""
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine,
                          W=args.netwidth_fine,
                          input_ch=input_ch,
                          input_ch_views=input_ch_views,
                          output_ch=output_ch,
                          skips=skips,
                          use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    """Create a 'black box' function, which takes data and a NeRF's MLP model as input to perform query.
       Data includes absolute sample positions along each ray and normalized viewing directions.
       Both are described in the world coordinate system / NDC space.
       This function itself comes with two sets of 'Positional Encoding' functions
       for positions and directions respectively.
    """
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                         embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
                                                                         netchunk=args.netchunk)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Define optimizer."""
    # 'params' [iterable]: Iterable of parameters to optimize or dicts defining parameter groups.
    # Attention: We optimize the model parameters of both the coarse and fine NeRF's MLP networks
    #            together during the optimization process.
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), eps=1e-7)
    optimizer.param_groups[0]['capturable'] = True ## ???
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Initialize the iteration step."""
    start = 0
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Load saved checkpoints."""
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, 'Checkpoints', f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname, 'Checkpoints'))) if 'tar' in f]
    print('Found ckpts', ckpts)

    # If the found checkpoints are valid and loading them is permitted.
    if len(ckpts) > 0 and not args.no_reload:
        # Load the latest saved checkpoints from its file path.
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load the current iteration step from checkpoints.
        start = ckpt['global_step']

        # Load the current optimizer parameters from checkpoints.
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load trained model parameters from checkpoints
        # for both the instantiated coarse and fine models.
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if args.N_importance > 0 and model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Create dictionary to save necessary arguments for training."""
    render_kwargs_train = {'network_query_fn' : network_query_fn,
                           'perturb'          : args.perturb,
                           'N_importance'     : args.N_importance,
                           'network_fine'     : model_fine,
                           'N_samples'        : args.N_samples,
                           'network_fn'       : model,
                           'use_viewdirs'     : args.use_viewdirs,
                           'white_bkgd'       : args.white_bkgd,
                           'raw_noise_std'    : args.raw_noise_std}

    # The transformation in the NDC (normalized device coordinate) space
    # is only good for forward-facing scenes, e.g. the LLFF dataset,
    # which is comprised of front-facing scenes with camera poses.
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        # Inject two more keys 'ndc' and 'lindisp' into the dictionary.
        # Because the argument 'ndc'     in the function 'render()'      is always set True  as default.
        # Because the argument 'lindisp' in the function 'render_rays()' is always set False as default.
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    """Create dictionary to save necessary arguments for testing."""
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    # Inject two more keys 'perturb' and 'raw_noise_std' into the dictionary.
    # During testing, do not activate 'Stratified Sampling'.
    # 'Stratified Sampling' is only designed for the training process.
    render_kwargs_test['perturb'] = False
    # During testing, no noise is added to the predicted volume density values
    # before passing them through the 'ReLU' activation function.
    render_kwargs_test['raw_noise_std'] = 0.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    Transforms model's predictions to semantically meaningful values,
    e.g. estimated radiance per ray and weights assigned to each sample's color per ray.

    Args:
        raw    [Tensor with shape: N_rays * N_samples along each ray * rendered rgb + volume density = 4]: Network's prediction of color values and volume density for each sample along a ray.
        z_vals [Tensor with shape: N_rays * N_samples along each ray]                                    : The 'z-axis' positions of samples ('marching depth' along each ray) relative to the ray origin in the camera coordinate system.
        rays_d [Tensor with shape: N_rays * 3]                                                           : Direction of each ray described in the NDC space / the world coordinate system.

    Returns:
        rgb_map    [Tensor with shape: N_rays * rgb=3]                   : Estimated RGB color / radiance per ray.
        weights    [Tensor with shape: N_rays * N_samples along each ray]: Weights assigned to each sample's color per ray.
        acc_map    [1D Tensor with 'N_rays' elements]                    : Sum of weights along each ray.
        disp_map   [1D Tensor with 'N_rays' elements]                    : Disparity map. Inverse of depth map.
        depth_map  [1D Tensor with 'N_rays' elements]                    : Estimated distance to object.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Calculate the distance between adjacent samples
       along the 'z-axis' in the camera coordinate system.
       Attention: distance not along the ray direction in the camera coordinate system!!!
    """
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 1e10 is appended to the last column of 'dists'.
    # Advantages: 1. To maintain the shape of 'dists' as N_rays * N_samples.
    #             2. To force the last column of "opacticy" to be 1, such that the classic alpha compositing holds. 
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    """Calculate the vector length of the ('z-axis' normalized in the camera coordinate system) ray direction.
       Vector length does not change after coordinate system transformation.
    """
    # rays_d     [Tensor with shape: N_rays * 3]: Ray direction described in the NDC space / the world coordinate system.
    #                                             It has been normalized along the 'z-axis' in the camera coordinate system
    #                                             (The absolute value of the 'z-axis' component equals to 1.)
    #                                             before transformed to the world coordinate system (and then to the NDC space).
    # rays_d_len [Tensor with shape: N_rays * 1]: Vector length of the ('z-axis' normalized in the camera coordinate system) ray direction above.
    #                                             This length does not change after coordinate system transformation.
    #                                             (→ The absolute value of the 'z-axis' component / vector length = 1 / 'rays_d_len'.)
    # 'rays_d[...,None,:].shape' = N_rays * 1 * 3.
    # 'torch.norm(rays_d[...,None,:], dim=-1).shape' = N_rays * 1.
    rays_d_len = torch.norm(rays_d[...,None,:], dim=-1)

    """Calculate the distance between adjacent samples along the ray direction
       in the NDC / world coordinate system according to:
       1. the distance between adjacent samples along the 'z-axis' in the camera coordinate system and
       2. the vector length of the ray direction, which is 'z-axis' normalized
          in the camera coordinate system, even though it is then transformed
          in the world coordinate system / NDC space.
          (Coordinate system transformation does not change vector length.)
    """
    dists = dists * rays_d_len
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Force the network's predicted RGB values in the range of (0,1)
       by using the 'sigmoid' activation function.
    """
    rgb = torch.sigmoid(raw[...,:3]) # [N_rays, N_samples, 3].
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Network regularization for performance improvement:
       Inject random Gaussian noise with zero mean and defined variance (here: unit variance with 'raw_noise_std' = 1.)
       to the predicted volume density values before passing them through the 'ReLU' activation function.
    """
    noise = 0.
    if raw_noise_std > 0.:
        # 'raw.shape'         = [N_rays, N_samples, 4].
        # 'raw[...,:3].shape' = [N_rays, N_samples, 3].
        # 'raw[...,3].shape'  = [N_rays, N_samples].
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """The alpha values for traditional alpha compositing."""
    # "The volume density 'sigma' is rectified using a ReLU to ensure that the output volume density is nonnegative."
    raw2alpha = lambda raw, dists, act_fn=F.relu : 1.-torch.exp(-act_fn(raw)*dists)
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples].

    """The accumulated transmittance along the ray."""
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples].

    """The weights assigned to each sampled color for 'Hierarchical Sampling'."""
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * transmittance # [N_rays, N_samples].
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Estimated RGB color / radiance per ray."""
    # 'weights[...,None].shape'       = [N_rays, N_samples, 1]
    # 'rgb.shape'                     = [N_rays, N_samples, rgb=3] (RGB: rendered und '0-1 normalized' per sample)
    # 'weights[...,None] * rgb.shape' = [N_rays, N_samples, rgb=3] (RGB: rendered, '0-1 normalized' and weighted per sample)
    # 'rgb_map.shape'                 = [N_rays, rgb=3].           (RGB: rendered, '0-1 normalized', weighted and summed per ray)
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    """Sum of weights along each ray."""
    acc_map = torch.sum(weights, -1) # [N_rays].

# ------------------------------------------------------------------------???
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
# ------------------------------------------------------------------------???

# ------------------------------------------------------------------------???
    depth_map = torch.sum(weights * z_vals, -1) # [N_rays].
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # [N_rays].
# ------------------------------------------------------------------------???

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """
    Volumetric rendering.

    Args:
        ray_batch [Tensor of shape: batch_size * 11]: A batch of ray origins, (z-axis normalized) directions, near, far and viewing directions,
                                                      when 'batch_size' <= 'chunk'.
                  [Tensor of shape:    'chunk' * 11]: A 'chunk' of ray origins, (z-axis normalized) directions, near, far and viewing directions,
                                                      when 'batch_size' > 'chunk'.
        network_fn       [Object]  : The coarse NeRF's MLP network for predicting RGB and density at each point in space.
        network_query_fn [Function]: A 'black box' function, which takes data and a NeRF's MLP model as input to perform query.
                                     Data includes absolute sample positions along each ray and normalized viewing directions.
                                     Both are described in the world coordinate system / NDC space.
                                     This function itself comes with two sets of 'Positional Encoding' functions
                                     for positions and directions respectively.
        N_samples     [Int]            : Number of coarse samples along each ray.
        retraw        [Bool]           : If True, include model's raw, unprocessed predictions.
        lindisp       [Bool]           : If True, sample linearly in inverse depth (disparity) rather than in depth.
        perturb       [Float: 0. or 1.]: If 1., adopt 'Stratified Sampling' and each ray is sampled at stratified random points in time.
        N_importance  [Int]            : Number of additional fine samples along each ray. These samples are only passed to the fine network.
        network_fine  [Object]         : The fine NeRF's MLP network for predicting RGB and density at each point in space.
        white_bkgd    [Bool]           : If True, assume a white background for rendering.
                                         This applies to the synthetic dataset only, which contains images with transparent background.
        raw_noise_std [Bool]           : Magnitude of noise to inject into volume density.
        verbose       [Bool]           : If True, print more debugging info.

    Returns:
        Volumetric rendering for a batch / 'chunk' of rays saved in one dictionary.
            For the fine network:
                z_std    [1D Tensor with 'N_rays' elements]: Standard deviation (Std) of distances along ray for fine samples.
            From the fine network:
                rgb_map  [Tensor of shape: N_rays * 3]:                              Estimated RGB color / radiance of a ray.
                disp_map [1D Tensor with 'N_rays' elements]:                         Disparity map. Inverse of depth map.
                acc_map  [1D Tensor with 'N_rays' elements]:                         Sum of weights along each ray.
                raw      [Tensor of shape: N_rays * (N_samples + N_importance) * 4]: Raw, unprocessed predictions from the fine network.
            From the coarse network:
                rgb0     [Tensor of shape: N_rays * 3]:      Estimated RGB color / radiance of a ray.
                disp0    [1D Tensor with 'N_rays' elements]: Disparity map. Inverse of depth map.
                acc0     [1D Tensor with 'N_rays' elements]: Sum of weights along each ray.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Unpack each minibatch to separate physical values."""
    # 'N_rays': Number of rays to process (volumetric rendering) simultaneously / concurrently.
    N_rays = ray_batch.shape[0]
    # Unpack rays origins, (z-axis normalized) directions, unit viewing directions, min distance and max distance (integration boundaries).
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3].
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # [N_rays, 3].
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) # [N_rays, 1, 2].
    near, far = bounds[...,0], bounds[...,1] # [N_rays, 1].
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """First partition the distance '[near, far]' along the z-axis
       of the camera coordinate system into ('N_samples'-1) evenly-spaced bins.
    """
    # Create a sequence of Nc points evenly scattered along unit length.
    # 1th______2th______3th______......______(Nc-1)th______(Nc)th
    #  0.__________________________________________________1.
    t_vals = torch.linspace(0., 1., steps=N_samples)
    # Recall that rays are previously projected to the NDC space,
    # where 'near' = 0. and 'far' = 1.!
    if not lindisp:
        # 'lindisp = False': Sample linearly in depth.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # 'lindisp = True': Sample linearly in inverse depth (disparity) rather than in depth.
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # Modulate all points per ray in the batch.
    z_vals = z_vals.expand([N_rays, N_samples])

    """Adopt 'Stratified Sampling'."""
    # The implementation of 'Stratified Sampling' here is inconsistent with what is described in the paper.
    # The first and last bins, in practice, are half the size of the others.
    # This does not harm the correctness of the algorithm.

    # 'perturb' [Float: 0. or 1.]: If 1., adopt 'Stratified Sampling'.
    if perturb > 0.:

        """Generate 'N_samples' bins for sampling per ray."""
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # Get intervals between samples.
        # The first and last bins, in practice, are half the size of the others.
        # This does not harm the correctness of the algorithm.
        # 'intervals.shape' = [N_rays, N_samples].
        intervals = upper - lower

        """Generate 'N_samples' random proportiones for the bins above per ray."""
        # The random proportiones 't_rand' decide where exactly
        # the coarse stratified samples lie in the intervals above.
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        """Calculate the 'z-axis' positions of the coarse stratified random points
           ('marching depth' along each ray) relative to the ray origin
           in the camera coordinate system.
        """
        # Relative positions of coarse stratified random points along each ray.
        # 'z_vals.shape' = [N_rays, N_samples].
        z_vals = lower + intervals * t_rand

    """Calculate the absolute positions of coarse stratified random points
       per ray in the NDC space / the world coordinate system.
    """
    # 'rays_o[...,None,:].shape' = [N_rays, 1, 3].
    # 'rays_d[...,None,:].shape' = [N_rays, 1, 3].
    # 'z_vals[...,:,None].shape' = [N_rays, N_samples, 1].
    # 'z_vals[...,:,None] * rays_d[...,None,:].shape' = [N_rays, N_samples, 3].
    # 'pts.shape' = [N_rays, N_samples, 3].
    pts = rays_o[...,None,:].expand([N_rays, z_vals.shape[-1], 3]) + \
          rays_d[...,None,:] * z_vals[...,:,None]

    """Evaluate the coarse network."""
    # The coarse network 'network_fn' is then queried
    # to predict raw network output 'raw' with shape: [N_rays, N_samples, rendered rgb + volume density = 4].
    raw = network_query_fn(pts, viewdirs, network_fn)
    # Transforms model's predictions to semantically meaningful values.
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Adopt 'Hierarchical Sampling' and consider the fine network."""
    # 'N_importance' > 0 or 'network_fine' != None.
    if N_importance > 0:

        """Log the outputs of the coarse network to distinguish from the outputs of the fine network."""
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        """For 'Hierarchical Sampling':
           generate new 'z-axis' positions of the additional fine stratified random points
           ('marching depth' along each ray) relative to the ray origin in the camera coordinate system.
        """
        # 'z_val': Relative positions of coarse stratified random points ('marching depth') along each ray.
        # 'z_vals_mid': Midpoints of coarse samples with shape = [N_rays, N_samples-1].
        # 'bins' = 'z_vals_mid' → Number of weights = 'N_partitions' = 'N_samples' - 1 - 1 = 'N_samples' - 2.
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 'z_samples': New relative positions of additional fine stratified random points along each ray.
        # 'z_samples.shape' = [N_rays, N_importance].
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        """Combination of sorted 'z-axis' positions of the coarse and fine samples
           ('marching depth' along each ray) relative to the ray origin in the camera coordinate system."""
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        """Calculate the absolute positions of coarse and fine stratified random points
           per ray in the NDC space / the world coordinate system.
        """
        # 'pts.shape' = [N_rays, N_samples + N_importance, 3].
        pts = rays_o[...,None,:].expand([N_rays, z_vals.shape[-1], 3]) + \
              rays_d[...,None,:] * z_vals[...,:,None]

        """Evaluate the fine network."""
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        # Transforms model's predictions to semantically meaningful values.
        # "The final rendering comes from C`^f(r)."
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Save volumetric rendering results."""
    ret = {'rgb_map' : rgb_map,
           'disp_map' : disp_map,
           'acc_map' : acc_map}

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False) # [N_rays].

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """
    Decompose the input tensor 'rays_flat' into minibatches of rays
    to feed into the network in sequence to avoid OOM.

    Args:
        rays_flat [Tensor of shape: batch_size * 11]: A batch of ray origins, (z-axis normalized) directions, near, far and viewing directions.
        chunk     [Int]                             : Maximum number of rays to process simultaneously / concurrently.
                                                      Used to control maximum memory usage. Does not affect final results.

    Returns:
        Volumetric rendering for a batch of rays saved in one dictionary.
            For the fine network:
                z_std    [1D Tensor with 'batch_size' elements]: Standard deviation (Std) of distances along ray for fine samples.
            From the fine network:
                rgb_map  [Tensor of shape: batch_size * 3]:                            Estimated RGB color / radiance of a ray.
                disp_map [1D Tensor with 'batch_size' elements]:                       Disparity map. Inverse of depth map.
                acc_map  [1D Tensor with 'batch_size' elements]:                       Sum of weights along each ray.
                raw      [Tensor of shape: batch_size * N_samples + N_importance * 4]: Raw, unprocessed predictions from the fine network.
            From the coarse network:
                rgb0     [Tensor of shape: batch_size * 3]:      Estimated RGB color / radiance of a ray.
                disp0    [1D Tensor with 'batch_size' elements]: Disparity map. Inverse of depth map.
                acc0     [1D Tensor with 'batch_size' elements]: Sum of weights along each ray.
    """

    # Initialize a dictionary to save all the volumetric rendering results in one batch.
    all_ret = {}

    # If 'chunk' >= 'rays_flat.shape[0]', then 'i' = 0 and 'rays_flat[i:i+chunk]' = 'rays_flat'.
    # Example for 'chunk (=3)  < rays_flat.shape[0] (=20)':
    # 'i' = 0, 3, 6, 9, 12, 15, 18.
    # 'rays_flat[0:3]', 'rays_flat[3:6]', 'rays_flat[6:9]', 'rays_flat[9:12]', 'rays_flat[12:15]', 'rays_flat[15:18]'.
    for i in range(0, rays_flat.shape[0], chunk):
        # 'ret': A dictionary to save volumetric rendering results in one batch / 'chunk'.
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        # Write results in 'all_ret'.
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # Example for 'all_ret['rgb_map']':
    # 'all_ret['rgb_map']' = [tensor([[0., 4., 3.],
    #                                 [4., 5., 6.]]),
    #                         tensor([[0., 6., 1.],
    #                                 [4., 4., 6.]]),
    #                         tensor([[0., 6., 1.],
    #                                 [4., 4., 6.]])]
    # 'torch.cat(all_ret['rgb_map'], 0)' = tensor([[0., 4., 3.],
    #                                              [4., 5., 6.],
    #                                              [0., 6., 1.],
    #                                              [4., 4., 6.],
    #                                              [0., 6., 1.],
    #                                              [4., 4., 6.]]).
    # 'all_ret['rgb_map'].shape' = batch_size * 3.
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render(H, W, K,
           chunk=1024*32,
           rays=None,
           c2w=None,
           ndc=True,
           near=0.,
           far=1.,
           use_viewdirs=False,
           c2w_staticcam=None,
           **kwargs):
    """
    Render rays.

    Args:
        H, W          [Int]                                                : Height and width of the image plane in pixels.
        focal         [Float]                                              : Focal length of pinhole camera.
        chunk         [Int]                                                : Maximum number of rays to process simultaneously / concurrently.
                                                                             Used to control maximum memory usage. Does not affect final results.
        rays          [Tensor of shape: [2, batch_size, 3]]                : A batch of ray origin positions and directions in the world coordinate system.
        c2w           [Numpy array / Tensor of shape [3, 4]]               : New defined camera-to-world transformation matrix.
        ndc           [Bool]                                               : The transformation in the NDC (normalized device coordinate) space is only good for forward-facing scenes.
                                                                             For example, the LLFF dataset, which is comprised of front-facing scenes with camera poses.
                                                                             If True, represent ray origin and directions in the NDC coordinates.
        near, far     [Float or Numpy array / Tensor of shape [batch_size]]: Nearest / Farthest relative distance from a point to the ray / camera origin
                                                                             along the z-axis of the camera coordinate system for defining the sampling intervall.
        use_viewdirs  [Bool]                                               : If True, feed viewing directions to the network for view-dependent apprearance (use full 5D input instead of 3D).
        c2w_staticcam [Numpy array / Tensor of shape [3, 4]]               : If not None, use this transformation matrix for camera,
                                                                             while using other c2w argument for viewing directions.

    Returns:
        All the volumetric rendering results from the fine network:
            rgb_map  [Tensor of shape: batch_size * 3]:      Rendered RGB color / radiance of a ray.
            disp_map [1D Tensor with 'batch_size' elements]: Disparity map. Inverse of depth map.
            acc_map  [1D Tensor with 'batch_size' elements]: Sum of weights along each ray.
        All the remaining volumetric rendering results (see the outputs from the 'batchify_rays()' function):
            extras: A dictionary with the remaining volumetric rendering results.
    """
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Get ray origins and directions
       in the world coordinate system."""
    if c2w is not None:
        # Special case to render full image.
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # Use provided ray batch and unpack the ray batch to ray origins and directions.
        rays_o, rays_d = rays
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Generate normalized viewing directions
       in the world coordinate system."""
    # If feeding viewing directions to the network for view-dependent apprearance.
    if use_viewdirs:
        # Use provided ray directions as viewing directions (Input).
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # Special case to visualize effect of viewdirs.
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 'viewdirs' [Unit vector with shape = batch_size * 3]: Normalized ray directions as viewing directions.
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """For forward-facing scenes: rays are then projected to the NDC space."""
    # Example for forward-facing scenes: the LLFF dataset,
    # which is comprised of front-facing scenes with camera poses.
    if ndc:
        # Ray origins and directions are transformed in the NDC space.
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    # 'rays_o.shape' = 'rays_d.shape' = batch_size * 3.
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # 'sh': torch.Size([batch_size, 3]).
    sh = rays_d.shape
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """All the above information are concatenated."""
    # Attention:
    # 'rays_d[...,:1].shape' = batch_size * 1.
    # 'rays_d[..., 0].shape' = batch_size.
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # 'rays.shape' = batch_size * (3 + 3 + 1 + 1 = 8)
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        # 'rays.shape' = batch_size * (3 + 3 + 1 + 1 + 3 = 11).
        rays = torch.cat([rays, viewdirs], -1)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Volumetric rendering for a batch of rays."""
    # 'chunk': Maximum number of rays to process simultaneously / concurrently.
    # 'all_ret': Volumetric rendering for a batch of rays saved in one dictionary.
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    """Read results."""
    # 'ret_list' contains all the volumetric rendering results from the fine network:
    # 'all_ret['rgb_map']'  = rgb_map  [Tensor of shape: batch_size * 3]:      Estimated RGB color / radiance of a ray.
    # 'all_ret['disp_map']' = disp_map [1D Tensor with 'batch_size' elements]: Disparity map. Inverse of depth map.
    # 'all_ret['acc_map']'  = acc_map  [1D Tensor with 'batch_size' elements]: Sum of weights along each ray.
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]

    # 'ret_dict' contains all the remaining volumetric rendering results in the 'all_ret' dictionary.
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def config_parser():
    parser = configargparse.ArgumentParser()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Basic options."""
    parser.add_argument("--config",  type=str, is_config_file=True,
                        help='Config file path')
    # ------------------------------------------------------------------------
    parser.add_argument("--expname", type=str, default=None,
                        help='Experiment name')
    parser.add_argument("--basedir", type=str, default=None,
                        help='Folder path to store checkpoints and log files.')
    parser.add_argument("--datadir", type=str, default=None,
                        help='Input dataset path')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Training options."""
    parser.add_argument("--N_rand",        type=int,   default=32*32*4,
                        help='Batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk",         type=int,   default=None,
                        help='Number of rays in a minibatch sent through the rendering process \
                              in parallel (simultaneously / concurrently) \
                              decrease if running out of memory.')
    parser.add_argument("--no_batching",   type=bool,  action='store_true',
                        help='If True : Take random rays from different images at a time. \
                              If False: Take random rays from only 1 image at a time.')
    parser.add_argument("--precrop_iters", type=int,   default=0,
                        help='If "no_batching" == False: Number of steps to train on central crops.')
    parser.add_argument("--precrop_frac",  type=float, default=.5,
                        help='If "no_batching" == False: Fraction of img taken for central crops.')
    # ------------------------------------------------------------------------
    parser.add_argument("--lrate",       type=float, default=5e-4,
                        help='Initial learning rate')
    parser.add_argument("--lrate_decay", type=int,   default=200,
                        help='Exponential learning rate decay (in 1000 steps).')
    # ------------------------------------------------------------------------
    parser.add_argument("--netdepth",      type=int, default=8,
                        help='Number of layers in the coarse network.')
    parser.add_argument("--netwidth",      type=int, default=256,
                        help='Number of channels per layer in the coarse network.')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='Number of layers in fine network.')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='Number of channels per layer in fine network.')
    parser.add_argument("--netchunk",  type=int,  default=1024*64, 
                        help='Number of points in a minibatch sent through the network in parallel \
                              decrease if running out of memory.')
    # ------------------------------------------------------------------------
    # Load saved checkpoints.
    parser.add_argument("--no_reload", type=bool, action='store_true', 
                        help='If True, do not reload weights from saved checkpoints.')
    parser.add_argument("--ft_path",   type=str,  default=None,
                        help='Numpy file path of specific weights to reload for the coarse network.')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Rendering options."""
    parser.add_argument("--use_viewdirs",  type=bool,  action='store_true',
                        help='If True, feed viewing directions to the network for view-dependent apprearance \
                              (use full 5D input instead of 3D).')
    parser.add_argument("--raw_noise_std", type=float, default=1.,
                        help='Standard deviation of Gaussian noise, \
                              which is injected to the volume density values before ReLU \
                              to regularize network for performance improvement \
                              1e0 recommended')
    # ------------------------------------------------------------------------
    # Stratified Sampling.
    parser.add_argument("--perturb", type=float, default=1.,
                        help='If 1., activate "Stratified Sampling".')
    # ------------------------------------------------------------------------
    # Hierarchical Sampling.
    parser.add_argument("--N_samples",    type=int, default=64,
                        help='Number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='Number of additional fine samples per ray')
    # ------------------------------------------------------------------------
    # Positional Encoding.
    parser.add_argument("--i_embed",        type=int, default=0,
                        help='If 0, activate "Positional Encoding", -1 for none.')
    parser.add_argument("--multires",       type=int, default=10,
                        help='log2 of max freq for "Positional Encoding" (3D location) \
                              "L"=10 for the three position components')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for "Positional Encoding" (2D direction) \
                              "L"=4 for the three unit vector components')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Testing options."""
    parser.add_argument("--render_only",   type=bool, action='store_true',
                        help='If True, do not optimize, reload weights and start testing.')
    parser.add_argument("--render_test",   type=bool, action='store_true',
                        help='Where to find poses for testing? \
                              If True : Use poses from the test dataset ("poses[i_test]"); \
                              If False: Use poses created in the "load_llff_data()" function ("render_poses").')
    parser.add_argument("--render_factor", type=int,  default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Dataset options."""
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='Choose one dataset from options: llff / blender / deepvoxels.')
    parser.add_argument("--testskip",     type=int, default=8,
                        help='Load every "args.testskip" images from test/val sets, useful for large datasets like deepvoxels')
    # ------------------------------------------------------------------------
    # Deepvoxels flags.
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')
    # ------------------------------------------------------------------------
    # Blender flags.
    parser.add_argument("--white_bkgd", type=bool, action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res",   type=bool, action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    # ------------------------------------------------------------------------
    # Llff flags.
    parser.add_argument("--factor",   type=int,  default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc",   type=bool, action='store_true',
                        help='If True, for non-forward-facing scenes, do not use normalized device coordinates.')
    parser.add_argument("--lindisp",  type=bool, action='store_true',
                        help='If True, sample linearly in inverse depth (disparity) rather than in depth.')
    parser.add_argument("--spherify", type=bool, action='store_true',
                        help='Set True for spherical 360 scenes.')
    parser.add_argument("--llffhold", type=int,  default=8,
                        help='Take every "args.llffhold" images as LLFF test set. Paper uses 8.')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Logging/saving options."""
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='Frequency of weight checkpoints to be saved.')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    """Set hyperparameters."""
    parser = config_parser()
    args = parser.parse_args()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Load data."""
    K = None

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        # 'images' [Numpy array]: RGB images of the scene.
        #                  Shape: Number of images * Height of the image plane * Width of the image plane * Three color values.
        # 'poses'  [Numpy array]: The corresponding camera poses (c2w rotation and translation matrix and hwf vector).
        #                  Shape: Number of images * 3 * 5.
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Read intrinsic parameters and camera poses."""
        hwf, poses = poses[0,:3,-1], poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Split the whole dataset into datasets for training, validation and testing."""
        if not isinstance(i_test, list):
            i_test = [i_test]

        # 'args.llffhold': Take every 1/N images as LLFF test set, paper uses 8.
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            # Example: 'i_test' = numpy.array([0, 3, 6, 9]) by using 'np.arange(10)[::3]'.
            # 'len(np.arange(images.shape[0])[::args.llffhold])': Number of images for testing.
            # 'len(np.arange(images.shape[0])[::args.llffhold])' = 'math.ceil(images.shape[0]/args.llffhold)' (Division round up).
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test

        # The remaining indice of all images are for training prepared.
        i_train = np.array([i for i in np.arange(int(images.shape[0]))
                              if (i not in i_test and i not in i_val) ])
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Define integration boundaries."""
        # 'args.no_ndc' [Bool]: If True, for non-forward-facing scenes,
        #                       don't use normalized device coordinates.
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir,
                                                                      args.half_res,
                                                                      args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir,
                                                                                    args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Read defined intrinsic parameters:
       Height 'H', width 'W' of the image plane and the camera focal length 'focal'.
    """
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    """Define the camera intrinsic matrix 'K', if not defined before."""
    # The camera intrinsics matrix 'K' = [[alpha', - beta' * cot(theta), u0],
    #                                     [     0,   beta' / sin(theta), v0],
    #                                     [     0,                    0,  1]],
    # where 'alpha' and 'beta': Scaling factores between unit vector lengthes in the camera and image coordinate system,
    #       'theta' : Angle between the two unit vectors in the image coordinate system,
    #       'alpha'': Camera focal length 'focal' * 'alpha',
    #       'beta'' : Camera focal length 'focal' * 'beta',
    #       'u0' and 'v0': Relative location of the origin of the camera coordinate system ('principle point') in the pixel coordinate system.
    if K is None:
        # Assume: 'alpha' = 'beta' = 1,
        #         'theta' = 90 degree,
        #         'u0' = 0.5*W and 'v0' = 0.5*H.
        K = np.array([[focal,     0, 0.5*W],
                      [    0, focal, 0.5*H],
                      [    0,     0,     1]])
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Create 'txt' file for registering arguments.
       (Read the python script: 'argparse_test.py'.)
       Copy the config file.
    """
    # 'args.basedir': Folder path to store checkpoints and log files.
    # 'args.expname': Experiment name.
    # Folder: 'args.basedir'
    #         Folder: 'args.expname'
    #                 File 'args.txt' for registering all the arguments
    #                      Height = 4
    #                      Radius = 2
    #                      manualSeed = None
    #                      ......
    #                 File 'config.txt' for copying the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # Create 'txt' file for registering all the arguments.
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    """
    with open(f, 'w') as file:
        for k, v in args._get_kwargs():
            file.write('{} = {}\n'.format(k, v))
    """

    # Copy the config file.
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """NeRF-Network initialization."""
    # 'render_kwargs_train' / 'render_kwargs_test': A dictionary returned after initiating a NeRF network
    #                                               with 2 more keys ('near' and 'far') injected.
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    # Inject two more keys 'near' and 'far' from the loaded dataset
    # into two dictionaries respectively.
    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 'render_kwargs_train' contains:
    # -----------------------------------------------------------
    # 'network_query_fn' [Function]: A 'black box' function, which takes data and a NeRF's MLP network as input to perform query.
    #                                Data includes absolute sample positions along each ray and normalized viewing directions.
    #                                Both are described in the world coordinate system / NDC space.
    #                                This function itself comes with two sets of 'Positional Encoding' functions
    #                                for positions and directions respectively.
    # 'N_importance'     [Int]:      Number of additional fine samples along each ray in 'Hierarchical Sampling'.
    #                                These samples are only passed to the NeRF's MLP fine network.
    # 'network_fine'     [Object]:   The NeRF's MLP fine network for predicting RGB and density at each point in space.
    # 'N_samples'        [Int]:      Number of coarse samples along each ray. These samples are passed to both the coarse and the fine NeRF-networks (MLP).
    # 'network_fn'       [Object]:   The NeRF's MLP coarse network for predicting RGB and density at each point in space.
    # 'use_viewdirs'     [Bool]:     If True, feed viewing directions to the network for view-dependent apprearance (use full 5D input instead of 3D).
    # -----------------------------------------------------------
    # 'perturb'     [Float: 0. or 1.]:                                     If 1., adopt 'Stratified Sampling' and each ray is sampled at stratified random points in time.
    # 'near', 'far' [Float or Numpy array / Tensor of shape [batch_size]]: Nearest / Farthest relative distance from a point to the ray / camera origin
    #                                                                      along the z-axis of the camera coordinate system for defining the sampling intervall.
    # -----------------------------------------------------------
    # 'white-bkgd'    [Bool]: If True, assume white background for rendering.
    #                         This applies to the synthetic dataset only, which contains images with transparent background.
    # 'raw_noise_std' [Bool]: Magnitude of noise to inject into volume density.
    # -----------------------------------------------------------
    # If we are not using dataset of forward-facing scenes or
    # we do not adopt transformation into the NDC space:
    # 'ndc'     [Bool]: False.
    # 'lindisp' [Bool]: False.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Prepare data for testing rendering."""
    # Where to find poses for testing?
    # 'args.render_test' = True:  Use poses from the test dataset ('poses[i_test]');
    # 'args.render_test' = False: Use created poses from the 'load_llff_data()' function ('render_poses').
    if args.render_test:
        render_poses = np.array(poses[i_test])

    """Turn the testing data into tensor and move them to GPU."""
    render_poses = torch.Tensor(render_poses).to(device)

    """The process for testing rendering starts."""
    # When testing rendering, 'args.render_only' is set 'True'. 
    # After the network model is trained, we save the trained network parameters.
    # We no longer train the network and get the rendering result directly.
    # Short circuit if only rendering out from trained model.
    if args.render_only:
        print('Render only.')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            # Terminate after the testing process is finished.
            return
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Prepare data for training."""
    # 'args.no_batching = True' : Take random rays from one single image (as a batch per iteration / optimization loop).
    # 'args.no_batching = False': Take random rays from different images (as a batch per iteration / optimization loop).
    use_batching = not args.no_batching
    if use_batching:
        # For each camera pose 'p':
        # first generate rays from the camera origin to each pixel on the image plane ('H', 'W' and 'K')
        # ('rays_o': Ray origin positions of all the rays in the world coordinate system.)
        # ('rays_d': Ray directions in the world coordinate system, which is 'z-axis normalized' in the camera coordinate system.)
        # and then describe rays and their origin in the world coordinate system
        # (according to different camera poses 'c2w': 'c2w[:3,:3]' and 'c2w[:3,-1]').
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [Total number of images N, ro+rd=2, H, W, 3].
        # 'images.shape' = N * H * W * 3.
        # 'images[:, None].shape' = 'images[:, np.newaxis].shape' = N * 1 * H * W * 3.
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb=3, H, W, 3].
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb=3, 3].
        # Collect images only for training.
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # [N_train, H, W, ro+rd+rgb=3, 3].
        # 'N_train*H*W': Number of rays (pixels) for training.
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [N_train*H*W, ro+rd+rgb=3, 3].
        rays_rgb = rays_rgb.astype(np.float32)

        """Shuffle all the training pixels for random sampling."""
        # 'np.random.shuffle()': Shuffle a numpy array alone the 'dim=0' axis.
        np.random.shuffle(rays_rgb) # [N_train*H*W, ro+rd+rgb=3, 3].

        i_batch = 0

    """Turn training data into tensor and move them to GPU."""
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """The process for training starts here."""
    # 'args.N_rand': Number of random rays / pixels from different images per iteration / optimization loop (batch size).
    N_rand = args.N_rand

    print('Begin training.')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    N_iters = 200000 + 1
    for i in trange(start, N_iters):
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Record training time duration per iteration."""
        time_start = time.time()
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Training preparation: prepare one raybatch tensor for the optimization loop."""
        if use_batching:
            # "Randomly sample a batch of camera rays from the set of all pixels in the dataset."
            # Take a batch of shuffled rays (pixels) for training.
            # 'batch.shape' = N_rand * 3 (ro+rd+rgb) * 3. (ro : 3 components to describe a camera origin positions in the world coordinate system,
            #                                              rd : 3 components to describe a ray direction in the world coordinate system,
            #                                              rgb: 3 color values to describe a pixel's color.)
            batch = rays_rgb[i_batch:i_batch+N_rand]
            # 'batch.shape' = 3 * N_rand * 3.
            batch = torch.transpose(batch, 0, 1)
            # 'batch_rays': A batch of camera origin positions and ray directions
            # respectively with shape = 'N_rand * 3.
            # 'target_s': A batch of true pixel colors with shape = N_rand * 3.
            batch_rays, target_s = batch[:2], batch[2]

            # Prepare for the next batch.
            i_batch += N_rand

            """Shuffle the training data after running an epoch."""
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image.
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i,:3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """During training: the rendering process starts here."""
        # The arguments 'verbose' and 'retraw' are the inputs to the 'render_rays()' function.
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Loss function for Backpropagation.
           PSNR for rendering quality evaluation.
        """
        # 'rgb'     : A batch of 'fine' rendered pixel color values with shape = N_rand * 3.
        # 'target_s': A batch of            true pixel color values with shape = N_rand * 3.
        # 'img_loss': Mean square error among all the batch of pixel color values from 'rgb' and 'target_s'. 
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            # 'extras['rgb0']': A batch of 'coarse' rendered pixel color values with shape = N_rand * 3.
            # 'target_s'      : A batch of              true pixel color values with shape = N_rand * 3.
            # 'img_loss0'     : Mean square error among all the batch of pixel color values from 'extras['rgb0']' and 'target_s'. 
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """The optimization process starts here."""
        # Zero out the historical loss gradient value.
        optimizer.zero_grad()
        # Compute the current gradient and backpropagate.
        loss.backward()
        # Update both the coarse and fine network parameters
        # according to the calculated gradient results.
        optimizer.step()
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Exponential learning rate decay."""
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # For the first iteration, 'global_step' = 0     : 'new_lrate' = 'args.lrate' = 5e-4.
        # For the last iteration,  'global_step' = 200000: 'new_lrate' = 'args.lrate' * 'decay_rate' = 5e-5.

        """Learning rate update for both the coarse and the fine network."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Record training time duration per iteration."""
        dt = time.time()-time_start
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Save checkpoints every 'args.i_weights' iterations."""
        # Contents to be saved:
        # 'global_step'            : The current iteration step.
        # 'network_fn_state_dict'  : Parameters of the coarse NeRF-network.
        # 'network_fine_state_dict': Parameters of the fine NeRF-network.
        # 'optimizer_state_dict'   : Parameters of the optimizer.
        if i%args.i_weights == 0:
            path = os.path.join(basedir, expname, 'Checkpoints', '{:06d}.tar'.format(i))
            torch.save({'global_step'            : global_step,
                        'network_fn_state_dict'  : render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict'   : optimizer.state_dict()
                       }, path)
            print('Saved checkpoints at', path)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Regularly output video."""
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        """Save the test set regularly."""
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    # Tensor张量有不同的数据类型, 每种类型分别有对应CPU和GPU版本。
    # 可通过函数'torch.set_default_tensor_type()'来修改或设置pytorch中默认的张量的浮点数类型。
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    train()
