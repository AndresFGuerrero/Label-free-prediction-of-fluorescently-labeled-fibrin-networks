import numpy as np
import tifffile
import tensorflow as tf

def upsample_convolution_result_symmetric(input_array, original_shape):
    """
    Upsample the result of a convolution operation symmetrically by padding with edge values.

    Args:
        input_array (numpy array): Result of the convolution.
        original_shape (tuple): Original shape of the input.

    Returns:
        numpy array: Upsampled array with symmetric edge padding.
    """
    conv_shape = input_array.shape
    padding = [(0, 0) for _ in range(len(input_array.shape))]

    for i in range(2, len(input_array.shape)):
        total_padding = original_shape[i] - conv_shape[i]
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        padding[i] = (left_padding, right_padding)

    return np.pad(input_array, padding, mode='edge')


def mean_3d(input_array):
    """
    Compute local mean using a 3D uniform convolution.

    Args:
        input_array (tf.Tensor): Shape (B, Z, Y, X, C).

    Returns:
        tf.Tensor: Local mean.
    """
    input_shape = input_array.shape
    input_array = tf.cast(input_array, tf.float32)
    kernel = tf.ones(shape=(1, 101, 101, 1, 1), dtype=tf.float32) / (1 * 101 * 101)

    conv_result = tf.nn.conv3d(input_array, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    conv_result = upsample_convolution_result_symmetric(conv_result, original_shape=input_shape)
    return conv_result


def std_3d(x, mean):
    """
    Compute local standard deviation.

    Args:
        x (tf.Tensor): Input tensor.
        mean (tf.Tensor): Precomputed local mean.

    Returns:
        tf.Tensor: Local standard deviation.
    """
    delta_squared = (x - mean) ** 2
    return tf.sqrt(mean_3d(delta_squared))


def znorm_3d(input_tensor):
    """
    Apply local z-normalization to a 3D input tensor.

    Args:
        input_tensor (tf.Tensor): Shape (B, Z, Y, X, C).

    Returns:
        dict: {'mean': mean, 'std': std, 'norm': normalized tensor}
    """
    mean = mean_3d(input_tensor)
    std = std_3d(input_tensor, mean)
    norm = (input_tensor - mean) / std
    return {'mean': mean, 'std': std, 'norm': norm}


def per_batch_perchannel_znorm_3d(input_tensor):
    """
    Apply per-batch, per-channel local z-normalization.

    Args:
        input_tensor (np.ndarray): Shape (B, Z, Y, X, C).

    Returns:
        np.ndarray: Normalized tensor.
    """
    B, Z, Y, X, C = input_tensor.shape
    out = np.zeros_like(input_tensor, dtype='float16')

    for b in range(B):
        for c in range(C):
            norm_result = znorm_3d(input_tensor[b:b+1, ..., c:c+1])['norm']
            out[b:b+1, ..., c:c+1] = norm_result.numpy().astype('float16')

    return out


# Example usage: load TIFF and normalize
file_path = '/path/raw_data.tiff'
#loads a numpy array of shape(Z,Y,X, channels)
t = tifffile.imread(file_path)


# Organize into dict
# T = {'dat': input channels, 'lbl': output channels}
T = {'dat': t[np.newaxis, ..., (0, 2, 3, 4)], 'lbl': t[np.newaxis, ..., 1:2]}

# Normalize
D = per_batch_perchannel_znorm_3d(T['dat'])
L = per_batch_perchannel_znorm_3d(T['lbl'])

# Save results
np.savez('/path/normalized_data.npz', dat=D, lbl=L)
