import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_Upscale():
    plugin_options:dict = None
    model_upscale = None
    devicename = None
    name = None

    processorname = 'upscale'
    type = 'frame_enhancer'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_upscale is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/Frame/real_esrgan_x4_fp16.onnx')
            self.model_upscale = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_upscale.get_inputs()
            model_outputs = self.model_upscale.get_outputs()
            self.io_binding = self.model_upscale.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def prepare_tile_frame(self, vision_tile_frame : Frame) -> Frame:
        vision_tile_frame = np.expand_dims(vision_tile_frame[:, :, ::-1], axis = 0)
        vision_tile_frame = vision_tile_frame.transpose(0, 3, 1, 2)
        vision_tile_frame = vision_tile_frame.astype(np.float32) / 255
        return vision_tile_frame


    def normalize_tile_frame(self, vision_tile_frame : Frame) -> Frame:
        vision_tile_frame = vision_tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        vision_tile_frame = vision_tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return vision_tile_frame

    def create_tile_frames(self, vision_frame : Frame, size):
        vision_frame = np.pad(vision_frame, ((size[1], size[1]), (size[1], size[1]), (0, 0)))
        tile_width = size[0] - 2 * size[2]
        pad_size_bottom = size[2] + tile_width - vision_frame.shape[0] % tile_width
        pad_size_right = size[2] + tile_width - vision_frame.shape[1] % tile_width
        pad_vision_frame = np.pad(vision_frame, ((size[2], pad_size_bottom), (size[2], pad_size_right), (0, 0)))
        pad_height, pad_width = pad_vision_frame.shape[:2]
        row_range = range(size[2], pad_height - size[2], tile_width)
        col_range = range(size[2], pad_width - size[2], tile_width)
        tile_vision_frames = []

        for row_vision_frame in row_range:
            top = row_vision_frame - size[2]
            bottom = row_vision_frame + size[2] + tile_width
            for column_vision_frame in col_range:
                left = column_vision_frame - size[2]
                right = column_vision_frame + size[2] + tile_width
                tile_vision_frames.append(pad_vision_frame[top:bottom, left:right, :])
        return tile_vision_frames, pad_width, pad_height


    def merge_tile_frames(self, tile_vision_frames, temp_width : int, temp_height : int, pad_width : int, pad_height : int, size) -> Frame:
        merge_vision_frame = np.zeros((pad_height, pad_width, 3)).astype(np.uint8)
        tile_width = tile_vision_frames[0].shape[1] - 2 * size[2]
        tiles_per_row = min(pad_width // tile_width, len(tile_vision_frames))

        for index, tile_vision_frame in enumerate(tile_vision_frames):
            tile_vision_frame = tile_vision_frame[size[2]:-size[2], size[2]:-size[2]]
            row_index = index // tiles_per_row
            col_index = index % tiles_per_row
            top = row_index * tile_vision_frame.shape[0]
            bottom = top + tile_vision_frame.shape[0]
            left = col_index * tile_vision_frame.shape[1]
            right = left + tile_vision_frame.shape[1]
            merge_vision_frame[top:bottom, left:right, :] = tile_vision_frame
        merge_vision_frame = merge_vision_frame[size[1] : size[1] + temp_height, size[1]: size[1] + temp_width, :]
        return merge_vision_frame


    def Run(self, temp_frame: Frame) -> Frame:
        size = (128, 8, 2)
        scale = 4
        temp_height, temp_width = temp_frame.shape[:2]
        tile_vision_frames, pad_width, pad_height = self.create_tile_frames(temp_frame, size)

        for index, tile_vision_frame in enumerate(tile_vision_frames):
            tile_vision_frame = self.prepare_tile_frame(tile_vision_frame)
            self.io_binding.bind_cpu_input(self.model_inputs[0].name, tile_vision_frame)
            self.model_upscale.run_with_iobinding(self.io_binding)
            ort_outs = self.io_binding.copy_outputs_to_cpu()
            result = ort_outs[0][0]
            del ort_outs
            tile_vision_frames[index] = self.normalize_tile_frame(result)
        merge_vision_frame = self.merge_tile_frames(tile_vision_frames, temp_width * scale, temp_height * scale, pad_width * scale, pad_height * scale, (size[0] * scale, size[1] * scale, size[2] * scale))
        return merge_vision_frame.astype(np.uint8)



    def Release(self):
        del self.model_upscale
        self.model_upscale = None
        del self.io_binding
        self.io_binding = None

