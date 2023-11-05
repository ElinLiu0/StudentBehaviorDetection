from streamlit_webrtc import webrtc_streamer
import time
import cv2
import av

# Initialize variables for FPS calculation
start_time = time.time()
frame_counter = 0
fps = 0

def videoFrameCallback(frame):
    # convert frame from av.video.frame.VideoFrame to numpy.ndarray
    frame = frame.to_ndarray(format="bgr24")
    global start_time, frame_counter, fps
    
    # Calculate FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_counter / elapsed_time
        start_time = time.time()
        frame_counter = 0
    
    # Draw FPS on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(frame, f"FPS: {fps:.2f}", bottom_left_corner, font, font_scale, font_color, line_type)
    
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=videoFrameCallback)



