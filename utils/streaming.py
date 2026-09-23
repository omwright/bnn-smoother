"""
Helpers shared by the online experiments.

Every experiment in this repository trains on a stream: one sample arrives,
each method updates once, and the methods are evaluated at checkpoints along
the way.  Two pieces of that loop are common to all of them and live here.

``update_single`` performs one method's online update.

``extract_weights`` and ``inject_weights`` give every Kalman method the same
starting weights (``share_init: true`` in the configs).  Without this, a method
could win or lose on its initial draw rather than on its update rule.  The two
Kalman methods store their weight covariance differently -- the proposed method
keeps a dense covariance over each layer's whole weight matrix, Wagner's KBNN a
separate covariance per output neuron -- so the shared representation is the
per-neuron one, which both can express.
"""

from __future__ import annotations

import torch

#: Methods that carry a weight covariance and can share an initialisation.
KALMAN_METHODS = {"kbnn_onestep", "wagner_kbnn"}


def is_kalman(spec: dict) -> bool:
    """True when the method keeps a weight covariance."""
    return spec["name"] in KALMAN_METHODS


def extract_weights(method):
    """Read a Kalman method's weights as (mean, per-neuron covariance) layers."""
    net = method.net
    dtype = getattr(net, "dtype", torch.float64)
    device = getattr(net, "device", torch.device("cpu"))
    weights = []

    if hasattr(net, "mw"):
        # Wagner's KBNN already stores a covariance per output neuron.
        for i in range(net.n_l - 1):
            weights.append((net.mw[i].clone(), net.Cw[i].clone()))
    else:
        # The proposed method stores one dense covariance per layer; take the
        # per-neuron diagonal blocks out of it.
        for layer in net.layers:
            mw = layer.weight_mean.clone()
            cov_4d = layer.weight_cov
            n_in, n_out = layer.n_in, layer.n_out
            Cw = torch.zeros(n_out, n_in, n_in, dtype=dtype, device=device)
            for j in range(n_out):
                Cw[j] = cov_4d[:, j, :, j]
            weights.append((mw, Cw))

    return weights


def inject_weights(method, weights):
    """Write weights from ``extract_weights`` into another Kalman method."""
    net = method.net

    if hasattr(net, "mw"):
        for i, (mw, Cw) in enumerate(weights):
            net.mw[i] = mw.clone().to(device=net.device, dtype=net.dtype)
            net.Cw[i] = Cw.clone().to(device=net.device, dtype=net.dtype)
    else:
        dtype, device = net.dtype, net.device
        for i, (mw, Cw) in enumerate(weights):
            layer = net.layers[i]
            n_in, n_out = layer.n_in, layer.n_out
            layer.weight_mean = mw.clone().to(device=device, dtype=dtype)
            # Per-neuron (n_out, n_in, n_in) -> flat (n_in*n_out, n_in*n_out).
            flat_dim = n_in * n_out
            cov_flat = torch.zeros(flat_dim, flat_dim, dtype=dtype, device=device)
            for j in range(n_out):
                idx = torch.arange(n_in, device=device) * n_out + j
                cov_flat[idx.unsqueeze(1), idx.unsqueeze(0)] = Cw[j].to(device=device, dtype=dtype)
            layer._cov.set_covariance(cov_flat)


def update_single(method, x: torch.Tensor, y: torch.Tensor):
    """Push one (x, y) pair through a method's online update.

    Parameters
    ----------
    method : BNNMethod
    x : Tensor, shape (input_dim,)
    y : Tensor, shape (output_dim,)
    """
    method.net.update(x, y)
