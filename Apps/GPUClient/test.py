from VideoProcessorGPU import FrameProcessor
import cv2


processor = FrameProcessor()

image = cv2.imread("./original.jpg")

preprocessedImage = processor.preprocess(image)
print(preprocessedImage)