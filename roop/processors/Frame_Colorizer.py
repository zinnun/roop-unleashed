import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_Colorizer():
    plugin_options:dict = None
    model_deoldify = None
    devicename = None
    name = None

    processorname = 'deoldify'
    type = 'frame_colorizer'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_deoldify is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/Frame/deoldify_artistic.onnx')
            self.model_deoldify = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_deoldify.get_inputs()
            model_outputs = self.model_deoldify.get_outputs()
            self.io_binding = self.model_deoldify.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def Run(self, temp_frame: Frame) -> Frame:
        temp_vision_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
        temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_GRAY2RGB)
        temp_vision_frame = cv2.resize(temp_vision_frame, (256, 256))
        temp_vision_frame = temp_vision_frame.transpose((2, 0, 1))
        temp_vision_frame = np.expand_dims(temp_vision_frame, axis=0).astype(np.float32)
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_vision_frame)
        self.model_deoldify.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        color_vision_frame = result.transpose(1, 2, 0)
        color_vision_frame = cv2.resize(color_vision_frame, (temp_frame.shape[1], temp_frame.shape[0]))
        temp_blue_channel, _, _ = cv2.split(temp_frame)
        color_vision_frame = cv2.cvtColor(color_vision_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        color_vision_frame = cv2.cvtColor(color_vision_frame, cv2.COLOR_BGR2LAB)
        _, color_green_channel, color_red_channel = cv2.split(color_vision_frame)
        color_vision_frame = cv2.merge((temp_blue_channel, color_green_channel, color_red_channel))
        color_vision_frame = cv2.cvtColor(color_vision_frame, cv2.COLOR_LAB2BGR)
        return color_vision_frame.astype(np.uint8)


    def Release(self):
        del self.model_deoldify
        self.model_deoldify = None
        del self.io_binding
        self.io_binding = None

