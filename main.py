import GUI
import System
import Functions
import ImgProcessing
import DataProcessing
import FirstFindContour
import TrackFirst
import Track
import CheckInit
import ResultPlot
import ResultVideo
import ResultData
import RemoveCatch

import cv2
import numpy as np


MHI_DURATION   = 0.5
MAX_TIME_DELTA = 0.5 
MIN_TIME_DELTA = 0.05 

BOOL_INIT       = False 
BOOL_CCD_CAMERA = False 
BOOL_PLOT       = True  

if __name__ == '__main__':
    num_of_fish, video_name, video_path, boundary = GUI.initialization()

    capture, out_folder_path, out_pic_path, first_frame = System.check_video( video_path, video_name) 
    frame_index  = int(capture.get(cv2.CAP_PROP_POS_FRAMES))   
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  
    frame_size   = ( frame_width, frame_height)                
    video_fps    = int(capture.get(cv2.CAP_PROP_FPS))          
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  

    frame_dict = {} 

    history_contour_dict = {} 
    history_motion_dict  = {} 

    fish_dict      = {}
    fishcolor_dict = {}

    write_frameindex  = 1
    remove_frameindex = 1
    timestamp         = 0
    timestamp_diff    = 2.7/video_fps
    motion_history    = np.zeros(( frame_height, frame_width), np.float32)
    block_size        = None 
    offset            = None
    avgfish_bgr       = None

    frame_dict[frame_index] = first_frame.copy()
    while True: 
        ret, frame   = capture.read()
        origin_frame = frame.copy()
        frame_index  = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        frame_dict[frame_index] = origin_frame 

        prev_oriframe = frame_dict[frame_index-1]
        prev_frame    = prev_oriframe.copy()

        if BOOL_CCD_CAMERA == True :
            prev_frame = ImgProcessing.remove_ccd_time(prev_frame)
            frame      = ImgProcessing.remove_ccd_time(frame)

        frame_diff  = cv2.absdiff( frame, prev_frame)
        gray_diff   = cv2.cvtColor( frame_diff, cv2.COLOR_BGR2GRAY)
        thresh_diff = ImgProcessing.threshold_mode( "threshold_OTSU", gray_diff, None, None, None)
        morph_diff  = ImgProcessing.morphology( thresh_diff, 'Opening', 'Rect', 3, 2)

        timestamp = timestamp + timestamp_diff
        motions   = ImgProcessing.motion_history_image( timestamp, morph_diff, frame_width, frame_height, motion_history, MHI_DURATION, MAX_TIME_DELTA, MIN_TIME_DELTA) 
        history_motion_dict[frame_index-1] = motions

        multi = 0.5
        if frame_index == 2: 
            history_contour_dict[frame_index-1] = FirstFindContour.mhiroi_findcnt( prev_frame, frame_index-1, motions, multi, boundary, None, None)
            history_contour_dict[frame_index]   = FirstFindContour.mhiroi_findcnt( frame, frame_index, motions, multi, boundary, None, None)

        elif frame_index > 2:
            contours = FirstFindContour.mhiroi_findcnt( frame, frame_index, motions, multi, boundary, block_size, offset)
            filter_contours = []
            for i in range( 0, len(contours), 1):
                cnt_bgr    = ImgProcessing.compute_contour_bgr( frame, contours, i)
                bool_color = Functions.filter_contour_bgr( avgfish_bgr, cnt_bgr)
                if bool_color == True:
                    filter_contours.append(contours[i])
            history_contour_dict[frame_index] = filter_contours

        if frame_index == 2:             
            contours, next_contours, fish_dict, fishid_movement, fisharea_dict, avgfish_bgr = TrackFirst.track_main( frame_index-1, frame_dict, history_contour_dict, history_motion_dict, boundary, num_of_fish)

            history_contour_dict[frame_index-1] = contours
            history_contour_dict[frame_index]   = next_contours

            fishcolor_dict = Functions.fishid_color_create( fishcolor_dict, fish_dict)

            block_size, offset = DataProcessing.update_blocksize( fish_dict, frame_index-1, next_contours, block_size)


        elif frame_index > 2 and BOOL_INIT == False:
            contours, next_contours, fish_dict, fisharea_dict, avgfish_bgr, fishid_movement = Track.track_main( frame_index-1, frame_dict, history_contour_dict, history_motion_dict, fish_dict, fisharea_dict, avgfish_bgr, boundary, fishid_movement)
            history_contour_dict[frame_index-1] = contours
            history_contour_dict[frame_index]   = next_contours

            if len(fishcolor_dict) != len(fish_dict):
                fishcolor_dict = Functions.fishid_color_update( fishcolor_dict, fish_dict)

            block_size, offset = DataProcessing.update_blocksize( fish_dict, frame_index-1, next_contours, block_size)

            BOOL_INIT, init_fishid = CheckInit.check_init( num_of_fish, fishid_movement)
            if BOOL_INIT == True:
                fish_dict, fisharea_dict, fishcolor_dict = CheckInit.initdata_transform( init_fishid, fish_dict, fisharea_dict, fishcolor_dict)


        elif frame_index > 2 and BOOL_INIT == True:
            contours, next_contours, fish_dict, fisharea_dict, avgfish_bgr = Track.track_main( frame_index-1, frame_dict, history_contour_dict, history_motion_dict, fish_dict, fisharea_dict, avgfish_bgr, boundary, None)
            history_contour_dict[frame_index-1] = contours
            history_contour_dict[frame_index]   = next_contours

            block_size, offset = DataProcessing.update_blocksize( fish_dict, frame_index-1, next_contours, block_size)

        if BOOL_INIT == True and BOOL_PLOT == True:
            write_startindex = write_frameindex
            if write_startindex == 1:
                writer = ResultVideo.video_init( video_name, out_folder_path, video_fps, frame_size)
            for i in range( write_startindex, frame_index, 1):
                ResultPlot.plot_main( video_name, out_pic_path, write_frameindex, frame_dict, history_contour_dict, fish_dict, fishcolor_dict)
                ResultVideo.video_main( write_frameindex, frame_dict, history_contour_dict, fish_dict, fishcolor_dict, writer)
                ResultData.data_main( write_frameindex, fish_dict, video_name, out_folder_path)

                write_frameindex = write_frameindex + 1


        if frame_index > 30 and BOOL_INIT == True:
            frame_dict, history_contour_dict, history_motion_dict, fish_dict = RemoveCatch.remove_main( frame_dict, history_contour_dict, history_motion_dict, fish_dict, remove_frameindex)
            remove_frameindex = remove_frameindex + 1

        if frame_index == video_length:
            print(" Finished Reading the Video ! ")
            break
        if 0xFF & cv2.waitKey(5) == 27:
            break

    capture.release()



