import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO

with open("yolov8n.engine", 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# For more information on TRT basics, refer to the introductory samples.
# This example uses a single image, repeated multiple times, to demonstrate
# batching.

print(engine)
context = engine.create_execution_context()

print(context)