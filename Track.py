import Contour2MHI
import MHI2Contour
import TrackFunction

import RematchOne

import numpy as np
from scipy.optimize import linear_sum_assignment


def track_main( frame_index, frame_dict, history_contour_dict, history_motion_dict, fish_dict, fisharea_dict, avgfish_bgr, boundary, fishid_movement):
    frame          = frame_dict[frame_index]
    frame_motions  = history_motion_dict[frame_index]
    frame_contours = history_contour_dict[frame_index]

    nextframe          = frame_dict[frame_index+1]
    nextframe_contours = history_contour_dict[frame_index+1]

    contours, cnt_mot_cmptable      = Contour2MHI.cnt_mhi_filter( frame, frame_contours, frame_motions, avgfish_bgr, boundary)
    next_contours, mot_cnt_cmptable = MHI2Contour.mhi_cnt_filter( nextframe, nextframe_contours, frame_motions, avgfish_bgr, boundary)


    compare_dict = TrackFunction.cntid_mhi_compare( cnt_mot_cmptable, mot_cnt_cmptable)
    compare_dict = TrackFunction.cntid_check_intersect( compare_dict, contours, next_contours, nextframe)

    bool_rematch, compare_dict = TrackFunction.cntid_checknew( frame_index, fish_dict, compare_dict, contours, next_contours, frame, nextframe, avgfish_bgr)

    if bool_rematch == True:
        fish_dict, fisharea_dict, avgfish_bgr = TrackFunction.cntid_rematch( frame_index, frame_dict, history_contour_dict, fish_dict, fisharea_dict, avgfish_bgr, compare_dict, contours, next_contours, frame, nextframe)

    unionid_list, unionfishid_list = TrackFunction.cntid_compare_union( frame_index, fish_dict, compare_dict)

    unionfishid_list, next_contours, useoldcnt_fishids = TrackFunction.check_comparedict( frame_index, fish_dict, unionfishid_list, contours, next_contours, frame, nextframe, avgfish_bgr, boundary)

    if len(fish_dict) == 1 and len(useoldcnt_fishids) == 1:
        new_unionfishid_list, new_compare_dict, bool_update = RematchOne.rematch_one( compare_dict, fish_dict, frame_index, contours, next_contours)
        if bool_update == True:
            unionfishid_list = new_unionfishid_list
            compare_dict     = new_compare_dict

    compare_result = TrackFunction.fishid_cnt_assignment( frame_index, fish_dict, fisharea_dict, unionfishid_list, contours, next_contours, nextframe, boundary)

    fish_dict, fishid_nextcnt = TrackFunction.create_fishdict( fish_dict, compare_result, next_contours, frame_index, cnt_mot_cmptable)

    fisharea_dict, avgfish_area, avgfish_bgr = TrackFunction.compute_avgfish_area( fisharea_dict, frame_index, frame_dict, fish_dict, contours, next_contours, avgfish_bgr, useoldcnt_fishids)

    if fishid_movement != None:
        fishid_movement = update_fishid_movement( compare_result, fishid_movement)
        return contours, next_contours, fish_dict, fisharea_dict, avgfish_bgr, fishid_movement
    
    elif fishid_movement == None:    
        return contours, next_contours, fish_dict, fisharea_dict, avgfish_bgr


def update_fishid_movement( compare_result, fishid_movement):
    compare_keys = list( compare_result.keys())
    for i in range( 0, len(compare_result), 1):
        fishid = compare_keys[i]
        fishid_movement[fishid] = fishid_movement[fishid] + "1"

    return fishid_movement
