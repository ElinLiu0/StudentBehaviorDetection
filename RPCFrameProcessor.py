import zerorpc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import random
from hashlib import md5
import cv2
import yaml
import cvcuda
import torch
from numba import cuda
import numba
import numpy as np
import cProfile
import pstats

@cuda.jit
def extractFeature(
    rows,
    outputs,
    boxes,
    scores,
    class_ids,
    x_factor,
    y_factor,
    confidenceThres
):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < rows:
        max_score = 0
        class_id = 0
        classes_score = outputs[i][4:]
        max_score = max(classes_score)
        if max_score >= confidenceThres:
            for x in range(len(classes_score)):
                if classes_score[x] == max_score:
                    class_id = x
                    break
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            class_ids[i] = class_id
            scores[i] = max_score
            boxes[i][0] = left
            boxes[i][1] = top
            boxes[i][2] = width
            boxes[i][3] = height
            

class FrameProcessor(object):
    def __init__(self) -> None: 
        self.inferenceClient = httpclient.InferenceServerClient(
            "localhost:8000", verbose=False, concurrency=1
        )
        self.config = yaml.load(open("./config/demo.yaml"), Loader=yaml.FullLoader)
        self.modelName = self.config['modelName']
        self.modelVersion = self.config['modelVersion']
        self.inputName = self.config['inputName']
        self.outputName = "output0"
        self.confidenceThres = self.config['confidenceThreshold']
        self.inputWidth, self.inputHeight = self.config['inputWidth'],self.config['inputHeight']
        self.iouThres = self.config['iouThreshold']
        self.classes = self.config["names"]
        self.colorPalette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        torch._C._cuda_init() # Preinitalize CUDA
    def drawDetections(
        self,
        img,
        box,
        score,
        classIndex):
        
        # extract the coordinates of the bounding box
        x1,y1,w,h = box
        # retrieve the color for the class ID
        color = self.colorPalette[classIndex]
        # convert the color values to integers
        color = [int(c) for c in color]
        # draw the bounding box and label on the image
        cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)
        label = '{}: {:.2F}'.format(self.classes[classIndex],score)
        (labelWidth,labelHeight),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        # calculate the position of the label text
        labelX = x1
        labelY = y1 - 10 if y1 - 10 > labelHeight else y1 + 10
        cv2.rectangle(img,(labelX,labelY-labelHeight),(labelX+labelWidth,labelY+labelHeight),color,cv2.FILLED)
        cv2.putText(img, label, (labelX, labelY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    def preprocess(self,imageFrame):
        # convet the image to a cuda tensor
        imageFrame = torch.tensor(imageFrame,device="cuda",dtype=torch.uint8)
        self.imageHeight,self.imageWidth = imageFrame.shape[:2]
        imageTensor = cvcuda.as_tensor(imageFrame,"HWC")
        imageTensor = cvcuda.cvtcolor(imageTensor,cvcuda.ColorConversion.BGR2RGB)
        imageTensor = cvcuda.resize(imageTensor,(self.inputWidth,self.inputHeight,3))
        # convert torch tensor to numpy array
        imageData = torch.as_tensor(imageTensor.cuda(),device="cuda")
        imageData = imageData / 255.0
        imageData = imageData.transpose(0,2).transpose(1,2)
        # expand the dimensions of the image
        imageData = torch.unsqueeze(imageData,0)
        return imageData.cpu().numpy().tolist()
    def postprocess(self, input_image, output):
        input_image = np.array(input_image).astype(np.uint8)
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = np.zeros((rows,4),dtype=np.int32)
        scores = np.zeros(rows,dtype=np.float32)
        class_ids = np.zeros(rows,dtype=np.int32)

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = input_image.shape[1] / self.inputWidth
        y_factor = input_image.shape[0] / self.inputHeight

        # launch the kernel
        extractFeature[rows,1](rows,outputs,boxes,scores,class_ids,x_factor,y_factor,self.confidenceThres)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidenceThres, self.iouThres)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
        
            self.drawDetections(input_image,box,score,class_id)
        return input_image.tolist()
    def infer(self,frame):
        frame = np.array(frame).astype(np.float32)
        inputs, outputs = [], []
        inputs.append(httpclient.InferInput(self.inputName, frame.shape, "FP32"))
        inputs[0].set_data_from_numpy(frame)
        outputs.append(httpclient.InferRequestedOutput(self.outputName))
        try:
            results = self.inferenceClient.async_infer(
                model_name=self.modelName,
                inputs=inputs,
                outputs=outputs,
                model_version=self.modelVersion,
                request_id=md5(str(random.random()).encode()).hexdigest(),
            )
            return results.get_result().as_numpy(self.outputName).tolist()
        except InferenceServerException as e:
            return None
if __name__ == "__main__":
    print("Starting the server...")
    server = zerorpc.Server(FrameProcessor())
    server.bind("tcp://0.0.0.0:4242")
    print("Server bound to port 4242")
    print("Waiting for requests...")
    print("Press CTRL+C to exit")
    server.run()