import torch


def combine_tokens(template_tokens, search_tokens, mode='direct'):
    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    else:
        raise NotImplementedError
    return merged_feature


def recover_tokens(merged_tokens, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError
    return recovered_tokens


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): Window size.

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size.
        H (int): Height of image.
        W (int): Width of image.

    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
