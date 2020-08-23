import common
import os
import random
import tensorrt as trt
import numpy as np
from PIL import Image

onnx_model_file='./weights/resnet152_f.onnx'

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(4)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)




with build_engine_onnx(onnx_model_file) as engine:
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.

    serialized_engine = engine.serialize()

    with open('./weights/resnet152.engine', 'wb') as f:
        f.write(serialized_engine)