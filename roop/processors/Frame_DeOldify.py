import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_DeOldify():
    model_deoldify = None
    devicename = None
    name = None

    processorname = 'deoldify'
    type = 'frame_colorizer'
    

    def Initialize(self, devicename:str):
        if self.model_deoldify is None:
            # replace Mac mps with cpu for the moment
            devicename = devicename.replace('mps', 'cpu')
            self.devicename = devicename
            model_path = resolve_relative_path('../models/Frame/deoldify_artistic.onnx')
            self.model_deoldify = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_deoldify.get_inputs()
            model_outputs = self.model_deoldify.get_outputs()
            self.io_binding = self.model_deoldify.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def Run(self, temp_frame: Frame) -> Frame:
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame) # .astype(np.float32)
        self.model_deoldify.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        return result.astype(np.uint8)


    def Release(self):
        del self.model_deoldify
        self.model_deoldify = None
        del self.io_binding
        self.io_binding = None

