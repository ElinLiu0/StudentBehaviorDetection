
from streamlit_webrtc import webrtc_streamer,WebRtcMode
import requests 
import streamlit as st
import pathlib
from datetime import datetime
from VideoProceesorCPU import VideoProcessor
import av
import pandas as pd
import minio
processor = VideoProcessor()
minioClient = minio.Minio(
            "192.168.180.81:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
sidebar = st.sidebar

sidebar.title("课堂行为检测Demo")
checkConnectionButton = sidebar.button("检查连接")


courseName = sidebar.text_input("课程名称")
teacherName = sidebar.text_input("教师名称")


sidebar.title("录制控制")

col1,col2 = sidebar.columns(2)
with col1:
    startSavingButton = col1.button("开始录制")
with col2:
    stopSavingButton = col2.button("停止录制")
sidebar.title("分析控制")
analyseButton = sidebar.button("开始分析")
sidebar.title("上传控制")
uploadButton = sidebar.button("上传录像")


if startSavingButton:
    if courseName:
        if teacherName:
            if not pathlib.Path(f"./recordings/{teacherName}/").exists():
                pathlib.Path(f"./recordings/{teacherName}/").mkdir(parents=True,exist_ok=True)
            if not pathlib.Path(f"./recordings/{teacherName}/{courseName}").exists():
                pathlib.Path(f"./recordings/{teacherName}/{courseName}").mkdir(parents=True,exist_ok=True)
time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
videoWriter = av.open(f"./recordings/{teacherName}/{courseName}/{time}.mp4","w")
videoOutputStream = videoWriter.add_stream("mpeg4",7)    

def videoCallBack(frame:av.VideoFrame):
    try:
        frame = processor.processing(frame)
        for packet in videoOutputStream.encode(frame):
            videoWriter.mux(packet)
        return frame
    except Exception as e:
        print(str(e))

if checkConnectionButton:
    try:
        response = requests.get("http://localhost:8000/v2/health/ready")
        if response.status_code == 200:
            sidebar.success("连接成功，推理服务已准备就绪！")
            
        else:
            sidebar.error("连接失败，推理服务存在异常")
    except:
        sidebar.error("连接失败，无法与推理服务建立连接")
st.title("实时推理画面")
webrtc_streamer(
    key="example",
    video_frame_callback=videoCallBack,
    async_processing=True,
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    
)

if stopSavingButton:
    videoWriter.close()
    sidebar.success("录制完成！")

if analyseButton:
    data = processor.getPredictData()
    print(data)
    st.title("分析结果")

if uploadButton:
    if courseName:
        
        # try:
        #     minioClient.make_bucket(courseName)
        # except Exception as e:
        #     st.error("创建存储桶失败："+str(e))
        try:
            files = pathlib.Path(f"./recordings/{teacherName}/{courseName}").glob("*.mp4")
            with sidebar.status("正在上传...",expanded=True) as status:
                for file in files:
                    # 检查文件名是不是今天的
                    if datetime.strptime(file.name.split(".")[0],"%Y-%m-%d-%H-%M-%S") > datetime.now():
                        continue
                    else:
                        try:
                            with open(file,"rb") as f:
                                minioClient.put_object(
                                    bucket_name="videoupload",
                                    object_name= teacherName+ "/" + courseName +"/"+file.name ,
                                    data=f,
                                    length=file.stat().st_size
                                )
                        except Exception as e:
                            sidebar.error("上传失败："+str(e))
                        st.write(f"上传完成：{file.name}")
                status.update(label="上传完成！",state="complete",expanded=False)
        except Exception as e:
            sidebar.error("上传失败："+str(e))
