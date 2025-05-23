import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import random
from typing import Tuple


class Flatten:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img):
        return img.flatten()


class Binarize:
    def __call__(self, img):
        return img.sign()


class RandomPixelSampler:
    """
    Randomly samples a subset of pixels from square images.

    Args:
        image_shape (Tuple[int]): Shape of the image (size,) or (C, size, size)
        pixel_ratio (float): Fraction of pixels to keep (0 < pixel_ratio <= 1)
        seed (int, optional): Random seed for reproducible sampling
    """

    def __init__(
        self, image_shape: Tuple[int, int, int], pixel_ratio: float, seed=None, **kwargs
    ):
        assert 0 < pixel_ratio <= 1, "pixel_ratio must be between 0 and 1"
        self.pixel_ratio = pixel_ratio
        self.image_shape = image_shape
        _, size1, size2 = image_shape
        assert size1 == size2, "Images must be square"
        self.n_total_pixels = size1 * size2
        self.n_kept_pixels = int(self.n_total_pixels * self.pixel_ratio)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.pixel_indices = torch.randperm(self.n_total_pixels)[: self.n_kept_pixels]

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        if img.dim() == 2:
            # Single channel image (H, W)
            flat_img = img.flatten()
            sampled = flat_img[self.pixel_indices]
        elif img.dim() == 3:
            # Multi-channel image (C, H, W)
            # Sample same spatial locations across all channels
            C, H, W = img.shape
            img_spatial = img.view(C, -1)  # Shape: (C, H*W)
            sampled_spatial = img_spatial[
                :, self.pixel_indices
            ]  # Shape: (C, n_kept_pixels)
            sampled = sampled_spatial.flatten()  # Flatten all channels together
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {img.dim()}D")
        return sampled


class PatchSampler:
    """
    Divides square images into patches and samples a subset of patches.

    Args:
        image_shape (Tuple[int]): Shape of the image (size,) or (C, size, size)
        patch_size (int): Size of square patches
        n_patches (int): Number of patches to sample
        seed (int, optional): Random seed for reproducible sampling
    """

    def __init__(
        self, image_shape: Tuple[int], patch_size=7, n_patches=8, seed=None, **kwargs
    ):
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.image_shape = image_shape

        # Get image size (assuming square images)
        if len(image_shape) == 1:
            self.img_size = image_shape[0]
        elif len(image_shape) == 3:
            _, size1, size2 = image_shape
            assert size1 == size2, "Images must be square"
            self.img_size = size1
        else:
            raise ValueError("image_shape must be (size,) or (C, size, size)")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Calculate patches per dimension
        self.patches_per_dim = self.img_size // patch_size
        total_patches = self.patches_per_dim**2

        if n_patches > total_patches:
            raise ValueError(
                f"Cannot sample {n_patches} patches from {total_patches} available"
            )

        all_patch_indices = list(range(total_patches))
        self.selected_patches = random.sample(all_patch_indices, n_patches)

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        patches = []
        for patch_idx in self.selected_patches:
            # Convert 1D patch index to 2D coordinates
            row = patch_idx // self.patches_per_dim
            col = patch_idx % self.patches_per_dim

            start_row = row * self.patch_size
            end_row = start_row + self.patch_size
            start_col = col * self.patch_size
            end_col = start_col + self.patch_size
            if img.dim() == 2:
                patch = img[start_row:end_row, start_col:end_col]
                patches.append(patch.flatten())
            else:  # 3D case
                patch = img[:, start_row:end_row, start_col:end_col]
                patches.append(patch.flatten())  # Flatten across all dimensions
        return torch.cat(patches)


class StructuredMask:
    """
    Applies structured masking patterns to square images (grayscale or RGB).

    Args:
        mask_type (str): Type of mask - 'checkerboard', 'outer_ring', 'center_hole', 'stripes', 'diagonal'
        mask_value (float): Value to use for masked pixels (default: 0.0)
    """

    def __init__(self, mask_type="checkerboard", mask_value=0.0, **kwargs):
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.mask_cache = {}  # Cache for different sizes

    def _create_mask(self, size):
        mask = torch.ones(size, size, dtype=torch.bool)

        if self.mask_type == "checkerboard":
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 1:
                        mask[i, j] = False
        elif self.mask_type == "outer_ring":
            if size > 4:
                mask[2:-2, 2:-2] = False
        elif self.mask_type == "center_hole":
            margin = size // 4
            mask[margin:-margin, margin:-margin] = False
        elif self.mask_type == "stripes":
            mask[1::2, :] = False
        elif self.mask_type == "diagonal":
            mask[:] = False
            for i in range(size):
                mask[i, i] = True
                mask[i, size - 1 - i] = True
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
        return mask

    def __call__(self, img):
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        # Handle different shapes
        if img.dim() == 2:
            # Grayscale [H, W]
            img = img.unsqueeze(0)  # -> [1, H, W]
        elif img.dim() == 3 and img.shape[0] in [1, 3]:
            pass  # Already [C, H, W]
        else:
            raise ValueError("Input must be [H, W], [1, H, W], or [3, H, W]")

        C, H, W = img.shape
        if H != W:
            raise ValueError("Only square images are supported")
        # Get or compute mask
        if H not in self.mask_cache:
            self.mask_cache[H] = self._create_mask(H)
        mask = self.mask_cache[H]
        # Apply mask to all channels
        masked_img = img.clone()
        for c in range(C):
            masked_img[c][~mask] = self.mask_value
        # Return only unmasked pixels per channel as a flat vector
        return masked_img[:, mask].flatten()


corruption_registry = {
    "flatten": Flatten,
    "random": RandomPixelSampler,
    "patch": PatchSampler,
    "structured": StructuredMask,
    "rotation": transforms.RandomRotation,
}
