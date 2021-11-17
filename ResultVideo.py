import Plot
import cv2

def video_init( video_name, folder_path, video_fps, frame_size):
    codec  = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec) 
    out_video_path = folder_path+'/'+video_name+'.mp4'
    writer = cv2.VideoWriter(filename= out_video_path, apiPreference= 0,fourcc= fourcc,
    fps= video_fps, frameSize= frame_size, isColor= True)

    return writer


def video_main( frame_index, frame_dict, history_contour_dict, fish_dict, fishcolor_dict, writer):
    frame            = frame_dict[frame_index]
    frame_index_str  = "FrameIndex"+str(frame_index)
    cnt_list         = history_contour_dict[frame_index]
    fishdict_details = {}
    cntid_dict       = {} 
    fishid_keys      = list(fish_dict.keys())
    for i in range( 0, len(fishid_keys), 1):
        fish_id = fishid_keys[i]
        fishdict_details[fish_id] = fish_dict[fish_id][frame_index_str]
        cnt_id     = fishdict_details[fish_id]["ContourID"]
        if cnt_id not in list(cntid_dict.keys()):
            cntid_dict[cnt_id] = []
        cntid_dict[cnt_id].append(fish_id)

    r       = 12
    rect_p1 = (20,20)
    rect_p2 = (100,60)
    index_frame   = Plot.write_frameIndex( frame.copy(), frame_index, rect_p1, rect_p2) 
    contour_frame = Plot.draw_fishid_contours( index_frame.copy(), cntid_dict, cnt_list, fishcolor_dict, r)

    writer.write(contour_frame)
