""" Implement linear splines.
    Based on the pytorch implementation of
    https://github.com/bayesiains/nsf """

# pylint: disable=too-many-arguments, too-many-locals

import torch
from .spline_utils import search_sorted


def unconstrained_linear_spline(
    inputs,
    unnormalized_pdf,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
):
    if not inverse:
        inside_interval_mask = (inputs >= left) & (inputs <= right)
    else:
        inside_interval_mask = (inputs >= bottom) & (inputs <= top)

    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = linear_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_pdf=unnormalized_pdf[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    return outputs, logabsdet


def linear_spline(inputs, unnormalized_pdf,
                  inverse=False,
                  left=0., right=1., bottom=0., top=1.):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    assert left <= torch.min(inputs) and torch.max(inputs) <= right, 'Inputs < 0 or > 1. Min: {}, Max: {}'.format(torch.min(inputs), torch.max(inputs))

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.
    cdf = F.pad(cdf, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        inv_bin_idx = searchsorted(cdf, inputs)

        bin_boundaries = (torch.linspace(0, 1, num_bins+1, device=inputs.device, dtype=cdf.dtype)
                          .view([1] * inputs.dim() + [-1])
                          .expand(*inputs.shape, -1))

        slopes = ((cdf[..., 1:] - cdf[..., :-1])
                  / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1]))
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.type(bin_pos.dtype)

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet
