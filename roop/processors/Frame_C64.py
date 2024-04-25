import numpy as np

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_C64():
    processorname = 'filter_c64'
    type = 'frame_colorizer'
    
    c64_palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [0x81, 0x33, 0x38],
        [0x75, 0xce, 0xc8],
        [0x8e, 0x3c, 0x97],
        [0x56, 0xac, 0x4d],
        [0x2e, 0x2c, 0x9b],
        [0xed, 0xf1, 0x71],
        [0x8e, 0x50, 0x29],
        [0x55, 0x38, 0x00],
        [0xc4, 0x6c, 0x71],
        [0x4a, 0x4a, 0x4a],
        [0x7b, 0x7b, 0x7b],
        [0xa9, 0xff, 0x9f],
        [0x70, 0x6d, 0xeb],
        [0xb2, 0xb2, 0xb2]
    ])

    def Initialize(self, devicename:str):
        return

    def Run(self, temp_frame: Frame) -> Frame:
        # Simply round the color values to the nearest color in the palette
        palette = self.c64_palette / 255.0  # Normalize palette
        img_normalized = temp_frame  / 255.0  # Normalize image

        # Calculate the index in the palette that is closest to each pixel in the image
        indices = np.sqrt(((img_normalized[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)).argmin(axis=2)
        # Map the image to the palette colors
        mapped_image = palette[indices]
        return (mapped_image * 255).astype(np.uint8)  # Denormalize and return the image


    def Release(self):
        return

