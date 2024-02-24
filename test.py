from Apps.GPUPipe.VideoProcessorGPUCV import VideoProcessor
import cv2


processor = VideoProcessor()

image = cv2.imread("./original.jpg")
print(image.shape)

preprocessedImage = processor.preprocess(image)

print(preprocessedImage.shape)

TRTOutput = processor.inference(preprocessedImage)

print(TRTOutput.shape)

postprocessedImage = processor.postProcess(image,TRTOutput)

print(postprocessedImage.shape)

cv2.imshow("processed",postprocessedImage)

cv2.waitKey(0)
