import cvcuda
import cudf
import tritonclient.http as httpclient
import random
from hashlib import sha256
import cupy as cp
from datetime import datetime
import yaml
import numpy as np
import cv2
import torch

class VideoProcessor:
    def __init__(self) -> None: 
        self.config = yaml.load(open("./Apps/PureCPUClient/config/demo.yaml"), Loader=yaml.FullLoader)
        self.modelName = self.config['modelName']
        self.modelVersion = self.config['modelVersion']
        self.inputName = self.config['inputName']
        self.outputName = self.config['outputName']
        self.confidenceThres = self.config['confidenceThreshold']
        self.inferenceClient = httpclient.InferenceServerClient(
            f"{self.config['serverHost']}:{self.config['serverPort']}",
            verbose=self.config['verbose'],
            concurrency=self.config['concurrency']
        )
        self.inputWidth, self.inputHeight = self.config['inputWidth'],self.config['inputHeight']
        self.iouThres = self.config['iouThreshold']
        self.classes = self.config["names"]
        self.colorPalette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # create a FPS counter
        self.fps = 0
        self.fpsCounter = 0
        self.fpsTimer = datetime.now()
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
    def drawDetections(self,img,box,score,classIndex):
        # extract the coordinates of the bounding box
        x1,y1,w,h = box
        # retrieve the color for the class ID
        color = self.colorPalette[classIndex]
        # draw the bounding box and label on the image
        cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)
        label = '{}: {}'.format(self.classes[classIndex],str(score * 100)[0:5] + '%')
        (labelWidth,labelHeight),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        # calculate the position of the label text
        labelX = x1
        labelY = y1 - 10 if y1 - 10 > labelHeight else y1 + 10
        cv2.rectangle(img,(labelX,labelY-labelHeight),(labelX+labelWidth,labelY+labelHeight),color,cv2.FILLED)
        cv2.putText(img, label, (labelX, labelY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    def postProcess(self,inputFrame,output):
        output = cp.transpose(cp.squeeze(output[0]))
        df = cudf.DataFrame(output)
        x_factor = self.imageWidth / self.inputWidth
        y_factor = self.imageHeight / self.inputHeight

        df['amax'] = cp.amax(df.iloc[:,4:84],axis=1)
        df['argmax'] = cp.where(df.iloc[:,4:84]==df['amax'].values[:,None])[1]
        df = df[df['amax'] > self.confidenceThres]
        boxes = df.iloc[:,0:4].to_numpy()
        scores = df['amax'].to_numpy()
        class_ids = df['argmax'].to_numpy()

        # apply factor to boxes
        boxes[:,0] = (boxes[:,0] - boxes[:,2]/2.0) * x_factor
        boxes[:,1] = (boxes[:,1] - boxes[:,3]/2.0) * y_factor
        boxes[:,2] = boxes[:,2] * x_factor
        boxes[:,3] = boxes[:,3] * y_factor

        boxes = boxes.astype(np.int32)
        scores = scores.astype(np.float32)
        class_ids = class_ids.astype(np.int32)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidenceThres, self.iouThres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            
            self.drawDetections(inputFrame,box,score,class_id)
        # calculate the FPS
        self.fpsCounter += 1
        elapsed = (datetime.now() - self.fpsTimer).total_seconds()
        if elapsed > 1.0:
            self.fps = self.fpsCounter / elapsed
            self.fpsCounter = 0
            self.fpsTimer = datetime.now()
        # draw the FPS counter
        cv2.putText(inputFrame, "FPS: {:.2f}".format(self.fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
        # draw current time on the top right of frame
        cv2.putText(inputFrame, datetime.now().strftime("%Y %I:%M:%S%p"), (self.imageWidth - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
        return inputFrame
    def inference(self,frame):
        inputs,outputs = [],[]
        inputs.append(httpclient.InferInput(self.inputName,frame.shape,"FP32"))
        inputs[0].set_data_from_numpy(frame)
        outputs.append(httpclient.InferRequestedOutput(self.outputName))
        results = self.inferenceClient.async_infer(
            model_name = self.modelName,
            inputs = inputs,
            outputs = outputs,
            model_version = self.modelVersion,
            request_id = sha256(str(random.random()).encode()).hexdigest()
        )
        output = results.get_result().as_numpy(self.outputName)
        return output
    def processing(self,frame):
        image_data = self.preprocess(frame)
        output = self.inference(image_data)
        outputFrame = self.postprocess(frame,output)
        return outputFrame