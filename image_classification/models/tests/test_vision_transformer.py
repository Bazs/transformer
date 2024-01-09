import torch

from image_classification.models.patchify import patchify
from image_classification.models.vision_transformer import (
    TransformerParams,
    VisionTransformer,
)


def test_patchify():
    # The first input image is a 4x4 image with 3 channels. For each pixel, the channels have the values (i, i + 100, i + 200), where i is the
    # flattened index of the pixel.
    image = torch.arange(1, 17).reshape(1, 1, 4, 4)
    image = image.repeat(1, 3, 1, 1)
    image[:, 1, :, :] += 100
    image[:, 2, :, :] += 200

    # We have a batch size of two, the second image is the first image + 1000.
    image = torch.cat([image, image + 1000], dim=0)

    patches = patchify(image, 2)
    # The expected shape is [batch_size, num_patches, patch_size * patch_size * channels]. Num patches is width // patch_size * height // patch_size = 4.
    num_patches = 4
    assert patches.shape == (2, num_patches, 12)
    # The expected patches are: top_left_patch, top_right_patch, bottom_left_patch, bottom_right_patch.
    # The patches should be ordered as [top_left_R, top_left_G, top_left_B, top_right_R, top_right_G, top_right_B, bottom_left_R, ...]
    expected_image_1_patches = torch.tensor(
        [
            [1, 101, 201, 2, 102, 202, 5, 105, 205, 6, 106, 206],
            [3, 103, 203, 4, 104, 204, 7, 107, 207, 8, 108, 208],
            [9, 109, 209, 10, 110, 210, 13, 113, 213, 14, 114, 214],
            [11, 111, 211, 12, 112, 212, 15, 115, 215, 16, 116, 216],
        ]
    )
    expected_image_2_patches = expected_image_1_patches + 1000
    expected_batch_patches = torch.stack([expected_image_1_patches, expected_image_2_patches], dim=0)
    assert torch.allclose(patches, expected_batch_patches)


def test_vision_transformer():
    vit = VisionTransformer(
        params=TransformerParams(
            image_width=32,
            image_height=32,
            image_channels=3,
            patch_size=4,
            emb_dim=32,
            n_heads=4,
            hid_dim=64,
            n_layers=2,
            output_dim=2,
            dropout=0.1,
        )
    )

    image_batch = torch.rand(1, 3, 32, 32)

    logits, attention = vit(image_batch)
