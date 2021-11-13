import Plot

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_main( video_name, folder_path, frame_index, frame_dict, history_contour_dict, fish_dict, fishcolor_dict):
    frame           = frame_dict[frame_index]
    nextframe       = frame_dict[frame_index+1]
    frame_index_str = "FrameIndex"+str(frame_index)
    cnt_list        = history_contour_dict[frame_index]
    nextcnt_list    = history_contour_dict[frame_index+1]
    
    fishdict_details  = {}
    cntid_dict        = {}
    nextcntid_dict    = {}
    fishid_keys       = list(fish_dict.keys())
    for i in range( 0, len(fishid_keys), 1):
        fish_id = fishid_keys[i]
        fishdict_details[fish_id] = fish_dict[fish_id][frame_index_str]
        cnt_id     = fishdict_details[fish_id]["ContourID"]
        nextcnt_id = fishdict_details[fish_id]["NextContourID"]
        if cnt_id not in list(cntid_dict.keys()):
            cntid_dict[cnt_id] = []
        if nextcnt_id not in list(nextcntid_dict.keys()):
            nextcntid_dict[nextcnt_id] = []

        cntid_dict[cnt_id].append(fish_id)
        nextcntid_dict[nextcnt_id].append(fish_id)

    rect_p1 = (20,20)
    rect_p2 = (100,60)
    index_frame     = Plot.write_frameIndex( frame.copy(), frame_index, rect_p1, rect_p2) 
    index_nextframe = Plot.write_frameIndex( nextframe.copy(), frame_index+1, rect_p1, rect_p2)

    r = 12
    contour_frame     = Plot.draw_fishid_contours( index_frame.copy(), cntid_dict, cnt_list, fishcolor_dict, r)
    nextcontour_frame = Plot.draw_fishid_contours( index_nextframe.copy(), nextcntid_dict, nextcnt_list, fishcolor_dict, r)

    mhi_frame     = Plot.draw_fishid_mhi( contour_frame, fishdict_details)
    mhi_nextframe = Plot.draw_fishid_mhi( nextcontour_frame, fishdict_details)


    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    save_original_pic( video_name, folder_path, frame_index, index_frame)
    save_fishframe_one( video_name, folder_path, frame_index, contour_frame)
    save_combineframe( video_name, folder_path, frame_index, contour_frame, nextcontour_frame)
    save_combineframe_mhi( video_name, folder_path, frame_index, mhi_frame, mhi_nextframe)


def save_original_pic( video_name, folder_path, frame_index, frame):
    file_name = "original"
    output_pic_ori_path = folder_path + "OriginalPicture"
    if not os.path.isdir(output_pic_ori_path):
        os.mkdir(output_pic_ori_path)

    frame_index = str(frame_index).replace(".0","") 
    image_name  = video_name +"_"+ file_name +"_"+ str(frame_index) +".png"
    image_path  = output_pic_ori_path + "/" + image_name 

    cv2.imwrite( image_path, frame)


def save_fishframe_one( video_name, folder_path, frame_index, contour_frame): 
    file_name = "Fishs"
    output_pic_ori_path = folder_path + "FishPicture"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(output_pic_ori_path):
        os.mkdir(output_pic_ori_path)

    frame_index = str(frame_index).replace(".0","") 
    image_name  = video_name +"_"+ file_name +"_"+ str(frame_index) +".png"
    image_path  = output_pic_ori_path + "/" + image_name 
    cv2.imwrite( image_path, contour_frame)


def save_combineframe( video_name, folder_path, frame_index, contour_frame, nextcontour_frame): 
    file_name = "Combine"
    output_pic_ori_path = folder_path + "CombinePicture"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(output_pic_ori_path):
        os.mkdir(output_pic_ori_path)

    comb_frame  = np.hstack(( contour_frame, nextcontour_frame))
    frame_index = str(frame_index).replace(".0","") 
    image_name  = video_name +"_"+ file_name +"_"+ str(frame_index) +".png"
    image_path  = output_pic_ori_path + "/" + image_name 
    cv2.imwrite( image_path, comb_frame)


def save_combineframe_mhi( video_name, folder_path, frame_index, mhi_frame, mhi_nextframe):
    file_name = "MHI"
    output_pic_ori_path = folder_path + "MotionHistoryImage"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(output_pic_ori_path):
        os.mkdir(output_pic_ori_path)

    comb_frame  = np.hstack(( mhi_frame, mhi_nextframe))
    frame_index = str(frame_index).replace(".0","") 
    image_name  = video_name +"_"+ file_name +"_"+ str(frame_index) +".png"
    image_path  = output_pic_ori_path + "/" + image_name 
    cv2.imwrite( image_path, comb_frame)

