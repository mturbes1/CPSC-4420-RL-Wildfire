import numpy as np
import cv2
import os
import shutil

#Get all RGB Videos
RGB_videos = [f'#1-7) All Video Pairs/#{i}) RGB Video {i}.MP4' for i in range(1, 8)]

RGB_video_captures = []

for video in RGB_videos:
    capture = cv2.VideoCapture(video)
    if capture.isOpened():
        RGB_video_captures.append(capture)
    else:
        print(f"Error opening video file: {video}")

#Get all IR Videos
IR_videos = [f'#1-7) All Video Pairs/#{i}) IR Video {i}.MP4' for i in range(1, 8)]

IR_video_captures = []

for video in IR_videos:
    capture = cv2.VideoCapture(video)
    if capture.isOpened():
        IR_video_captures.append(capture)
    else:
        print(f"Error opening video file: {video}")

#Create directories for the frame pairs
output_dir_RGB = "Segmented Frames/RGB"
output_dir_IR = "Segmented Frames/IR"

for output_dir in [output_dir_RGB, output_dir_IR]:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  
    os.makedirs(output_dir)

frame_count = 1
video_index = 0

finished_videos = [False] * 7

while not all(finished_videos):
    for video_index, (RGB_capture, IR_capture) in enumerate(zip(RGB_video_captures, IR_video_captures)):
        if finished_videos[video_index]:
            continue

        RGB_ret, RGB_frame = RGB_capture.read()
        IR_ret, IR_frame = IR_capture.read()

        if RGB_ret and IR_ret:
            #Resize frames by half to save space
            RGB_frame = cv2.resize(RGB_frame, (0, 0), fx=0.5, fy=0.5)
            IR_frame = cv2.resize(IR_frame, (0, 0), fx=0.5, fy=0.5)

            RGB_video_dir = os.path.join(output_dir_RGB, f"Video {video_index + 1}")
            IR_video_dir = os.path.join(output_dir_IR, f"Video {video_index + 1}")
            os.makedirs(RGB_video_dir, exist_ok=True)
            os.makedirs(IR_video_dir, exist_ok=True)

            #After every 180 frames
            if frame_count % 180 == 0:
                RGB_filename = os.path.join(RGB_video_dir, f"RGB_frame_{frame_count}.jpg")
                IR_filename = os.path.join(IR_video_dir, f"IR_frame_{frame_count}.jpg")
                
                #Write frames to directory
                cv2.imwrite(RGB_filename, RGB_frame)
                cv2.imwrite(IR_filename, IR_frame)
            
            frame_count += 1
            
        else:
            #When video is finished
            finished_videos[video_index] = True
            if not RGB_ret:
                print(f"End of RGB video {video_index + 1} reached.")
            if not IR_ret:
                print(f"End of IR video  {video_index + 1} reached.")

    

for capture in RGB_video_captures + IR_video_captures:
    capture.release()