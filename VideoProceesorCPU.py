import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import random
from hashlib import sha256
import cv2
import yaml
import av
import pymp
from datetime import datetime
import pandas as pd
class VideoProcessor:
    def __init__(self) -> None: 
        self.config = yaml.load(open("./config/demo.yaml"), Loader=yaml.FullLoader)
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
        self.detectionClasses,self.detectionConfidence,self.detectedBoxes = [],[],[]
        self.colorPalette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # create a FPS counter
        self.fps = 0
        self.fpsCounter = 0
        self.fpsTimer = datetime.now()
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
    def preprocess(self,imageFrame):
        self.imageHeight,self.imageWidth = imageFrame.shape[:2]
        imageFrame = cv2.cvtColor(imageFrame,cv2.COLOR_BGR2RGB)
        imageFrame = cv2.resize(imageFrame,(self.inputWidth,self.inputHeight))
        imageData = np.array(imageFrame) / 255.0
        imageData = np.transpose(imageData,(2,0,1))
        imageData = np.expand_dims(imageData,axis=0).astype(np.float32)
        return imageData
    def postprocess(self,inputFrame,output):
        output = np.transpose(np.squeeze(output[0]))
        rows = output.shape[0]
        boxes = pymp.shared.array((rows,4),dtype='int32')
        scores = pymp.shared.array((rows),dtype='float32')
        class_ids = pymp.shared.array((rows),dtype='int32')

        x_factor = self.imageWidth / self.inputWidth
        y_factor = self.imageHeight / self.inputHeight

        try:
            with pymp.Parallel(8) as p:
                for i in p.range(rows):
                    classes_scores = output[i][4:]
                    maxConfidence = np.amax(classes_scores)

                    if maxConfidence >= self.confidenceThres:
                        classIndex = np.argmax(classes_scores)

                        x,y,w,h = output[i][:4]

                        left = int((x - w / 2) * x_factor)
                        top = int((y - h / 2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)

                        class_ids[i] = classIndex
                        scores[i] = maxConfidence
                        boxes[i] = [left,top,width,height]
        except Exception as e:
            print(str(e))
        # convert into numpy array
        boxes = np.array(boxes,dtype='int32')
        scores = np.array(scores,dtype='float32').squeeze()
        class_ids = np.array(class_ids,dtype='int32').squeeze()
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidenceThres, self.iouThres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            # update the detections table
            className = self.classes[class_id]
            
            
            # draw the detection on the frame
            self.drawDetections(inputFrame,box,score,class_id)
            # append the detection to the table
            self.detectionClasses.append(className)
            self.detectionConfidence.append(score)
            self.detectedBoxes.append(box)
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
    def getPredictData(self):
        self.detectionsTable = pd.DataFrame()
        print(self.detectionClasses)
        print(self.detectionConfidence)
        print(self.detectedBoxes)
        self.detectionsTable['Class'] = self.detectionClasses
        self.detectionsTable['Confidence'] = self.detectionConfidence
        self.detectionsTable['Box'] = self.detectedBoxes
        return self.detectionsTable
    def processing(self,frame):
        inputs,outputs = [],[]
        frame = frame.to_ndarray(format="bgr24")
        image_data = self.preprocess(frame)
        inputs.append(httpclient.InferInput(self.inputName,image_data.shape,"FP32"))
        inputs[0].set_data_from_numpy(image_data)
        outputs.append(httpclient.InferRequestedOutput(self.outputName))
        try:
            results = self.inferenceClient.async_infer(
                model_name = self.modelName,
                inputs = inputs,
                outputs = outputs,
                model_version = self.modelVersion,
                request_id = sha256(str(random.random()).encode()).hexdigest()
            )
            output = results.get_result().as_numpy(self.outputName)
            outputFrame = self.postprocess(frame,output)
            return av.VideoFrame.from_ndarray(outputFrame,format="bgr24")
        except InferenceServerException as e:
            return str(e)