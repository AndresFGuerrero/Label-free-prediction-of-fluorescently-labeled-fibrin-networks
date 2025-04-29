import numpy as np

def patchify(x, size=256):
    """
    Extract non-overlapping square patches from 5D image data.

    Args:
        x (np.ndarray): Input tensor of shape (B, Z, Y, X, C).
        size (int): Patch size (applies to Y and X dimensions).

    Returns:
        np.ndarray: Tensor of shape (N, Z, size, size, C), where N = B * (Y//size) * (X//size)
    """
    patches = []
    B, Z, Y, X, C = x.shape

    for b in range(B):
        for i in range(Y // size):
            start_i = i * size
            end_i = start_i + size
            for j in range(X // size):
                start_j = j * size
                end_j = start_j + size
                patch = x[b, :, start_i:end_i, start_j:end_j, :]
                patches.append(patch)

    return np.stack(patches)

# Example: load normalized data
input_path = '/path/normalized_data.npz'
output_path = '/path/patched_data.npz'


# Load pre-normalized image dictionary
T = dict(np.load(input_path))

# Apply patch extraction
D = patchify(T['dat'])  # Shape: (N, Z, size, size, C)
L = patchify(T['lbl'])  # Shape: (N, Z, size, size, C)

# Save patched data
np.savez(output_path, dat=D, lbl=L)

# Optional verification
P = dict(np.load(output_path))
print("Patched shapes:", P['dat'].shape, P['lbl'].shape)  # Expect (N, Z, size, size, C)
