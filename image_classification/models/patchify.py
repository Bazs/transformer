import torch


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split the image into patches."""
    # Unfold along the height dimension to get [batch_size, channels, num_height_patches, width, height_patch_size]
    image = image.unfold(2, patch_size, patch_size)
    # Unfold along the width dimension to get [batch_size, channels, num_height_patches, num_width_patches, height_patch_size, width_patch_size]
    image = image.unfold(3, patch_size, patch_size)
    # Reorder to [batch_size, num_height_patches, num_width_patches, height_patch_size, width_patch_size, channels]
    image = image.permute(0, 2, 3, 4, 5, 1)
    # Flatten each patch to get [batch_size, num_height_patches, num_width_patches, height_patch_size * width_patch_size * channels]
    image = image.flatten(start_dim=3)
    # Flatten to get [batch_size, num_height_patches * num_width_patches, height_patch_size * width_patch_size * channels]
    image = image.flatten(start_dim=1, end_dim=2)
    return image
