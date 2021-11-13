import os
import cv2
import sys


def check_video( video_path, video_name):
    capture = cv2.VideoCapture(video_path)

    if capture.isOpened() == False:
        sys.exit("System: "+"Open VideoPath= " + str(video_path) + " Failed !! ")
    elif capture.isOpened() == True:
        current_position   = os.getcwd()
        current_position   = current_position.replace("\\","/")
        output_folder_path = current_position + "/Output/"+video_name
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
            
        output_pic_path = output_folder_path + "/pic_output/"
        if not os.path.isdir(output_pic_path):
            os.mkdir(output_pic_path) 

    ret, frame = capture.read()
    if ret == False:
        sys.exit("System: could not read from " + str(video_path) + " ! ")

    return capture, output_folder_path, output_pic_path, frame
