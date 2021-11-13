import cv2
import numpy as np


def rematch_one( compare_dict, fish_dict, frame_index, contours, next_contours):
    fishdict_keys   = list(fish_dict.keys())
    fish_id         = fishdict_keys[0]
    fish_detail_all = fish_dict[fish_id]
    fish_detail     = fish_detail_all['FrameIndex'+str(frame_index-1)]
    cnt_id          = fish_detail['NextContourID']
    cnt             = contours[cnt_id]
    cnt_m           = cv2.moments(cnt)
    cnt_cx          = int(cnt_m['m10']/cnt_m['m00'])
    cnt_cy          = int(cnt_m['m01']/cnt_m['m00'])
    

    nextcntid_list   = []
    comparedict_keys = list(compare_dict.keys())
    for i in range( 0, len(compare_dict), 1):
        nextcntid_list = nextcntid_list + compare_dict[comparedict_keys[i]]
    nextcntid_list = list(set(nextcntid_list))

    min_nextcntid = None
    min_distance  = 0
    for i in range( 0, len(nextcntid_list), 1):
        nextcnt_id = nextcntid_list[i]
        nextcnt    = next_contours[nextcnt_id]
        nextcnt_m  = cv2.moments(nextcnt)
        nextcnt_cx = int(nextcnt_m['m10']/nextcnt_m['m00'])
        nextcnt_cy = int(nextcnt_m['m01']/nextcnt_m['m00'])

        dist = np.sqrt(np.square(nextcnt_cx-cnt_cx)+np.square(nextcnt_cy-cnt_cy))
        if i == 0:
            min_nextcntid = nextcnt_id
            min_distance  = dist
        elif i > 0:
            if dist < min_distance:
                min_distance  = dist
                min_nextcntid = nextcnt_id

    unionfishid_list     = [[[fish_id],[min_nextcntid]]]
    compare_dict         = {}
    compare_dict[cnt_id] = [min_nextcntid]

    if min_nextcntid == None:
        bool_update = False
    else:
        bool_update = True


    return unionfishid_list, compare_dict, bool_update