import Functions
import Contour2MHI
import MHI2Contour
import DataProcessing
import ImgProcessing

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment



def track_main( frame_index, frame_dict, history_contour_dict, history_motion_dict, boundary, num_of_fish):
    frame          = frame_dict[frame_index]
    frame_motions  = history_motion_dict[frame_index]
    frame_contours = history_contour_dict[frame_index]

    nextframe          = frame_dict[frame_index+1]
    nextframe_contours = history_contour_dict[frame_index+1]

    contours, cnt_mot_cmptable      = Contour2MHI.cnt_mhi_filter( frame, frame_contours, frame_motions, None, boundary)
    next_contours, mot_cnt_cmptable = MHI2Contour.mhi_cnt_filter( nextframe, nextframe_contours, frame_motions, None, boundary)

    compare_dict = cntid_mhi_compare( cnt_mot_cmptable, mot_cnt_cmptable)

    compare_intersect = cntid_check_intersect( compare_dict, contours, next_contours, nextframe)

    compare_filter = compare_cntfilter( compare_intersect, contours, num_of_fish)

    compare_result = cntid_assignment( compare_filter, contours, next_contours)

    if len(compare_result) < num_of_fish: 
        num_of_lostfish = int(num_of_fish-len(compare_result))
        compare_result  = find_freeze( compare_result, contours, next_contours, num_of_lostfish, frame, nextframe)

    fish_dict, fishid_movement = create_fishdict( compare_result, contours, next_contours, frame_index, cnt_mot_cmptable)

    fisharea_dict, avgfish_bgr = compute_avgfish_area( frame_index, frame_dict, fish_dict, contours, next_contours)


    return contours, next_contours, fish_dict, fishid_movement, fisharea_dict, avgfish_bgr


def cntid_mhi_compare( cnt_mot_cmptable, mot_cnt_cmptable):
    compare_dict = {}
    cntid_list   = list(cnt_mot_cmptable.keys())
    motid_list   = list(mot_cnt_cmptable.keys())
    for i in range( 0, len(cnt_mot_cmptable), 1):
        cnt_id       = cntid_list[i]
        mot_list     = cnt_mot_cmptable[cnt_id]["MotionIDs"]
        nextcnt_list = []
        for j in range( 0, len(mot_list), 1):
            mot_id = mot_list[j]
            if mot_id in motid_list:
                nextcnts     = mot_cnt_cmptable[mot_id]["ContourIDs"]
                nextcnt_list = nextcnt_list + nextcnts
        
        if len(nextcnt_list) > 0: 
            nextcnt_list = list(set(nextcnt_list))
            nextcnt_list.sort()
            compare_dict[cnt_id] = nextcnt_list

    return compare_dict


def cntid_check_intersect( compare_dict, contours, next_contours, nextframe):
    compare_intersect = {}
    cntid_list        = list(compare_dict.keys())
    for i in range( 0, len(compare_dict), 1):
        cnt_id       = cntid_list[i]
        cnt          = contours[cnt_id]
        nextcnt_list = compare_dict[cnt_id]
        if len(nextcnt_list) == 1:
            compare_intersect[cnt_id] = nextcnt_list
            continue

        temp_list = []
        for j in range( 0, len(nextcnt_list), 1):
            nextcnt_id   = nextcnt_list[j]
            nextcnt      = next_contours[nextcnt_id]

            bool_intersect  = Functions.two_contourIntersect( nextframe, cnt, nextcnt)
            if bool_intersect == True:
                temp_list.append(nextcnt_id)

        if len(temp_list) == 0 or len(temp_list) == 1:
            compare_intersect[cnt_id] = temp_list
        elif len(temp_list) > 1:
            intersect_areas = []
            for j in range( 0, len(temp_list), 1):
                nextcnt_id = temp_list[j]
                nextcnt    = next_contours[nextcnt_id]
                bool_intersect, intersect_area = Functions.two_contourIntersect_area( nextframe, cnt, nextcnt)
                intersect_areas.append(intersect_area)
            max_index  = intersect_areas.index(max(intersect_areas))
            nextcnt_id = temp_list[max_index]
            temp_list  = [nextcnt_id]
            compare_intersect[cnt_id] = temp_list


    return compare_intersect


def compare_cntfilter( compare_dict, contours, num_of_fish):
    cntids = list(compare_dict.keys())
    for i in range( 0, len(compare_dict), 1):
        cnt_id      = cntids[i]
        cnt         = contours[cnt_id]
        cnt_rect    = cv2.minAreaRect(cnt)
        rect_width  = min(cnt_rect[1][0],cnt_rect[1][1])
        rect_length = max(cnt_rect[1][0],cnt_rect[1][1])
        if rect_length > rect_width * 8:
            del compare_dict[cnt_id] 

    cntids   = list(compare_dict.keys())
    cntareas = []
    for i in range( 0, len(compare_dict), 1):
        cnt_id   = cntids[i]
        cnt      = contours[cnt_id]
        cnt_area = cv2.contourArea(cnt)
        cntareas.append(cnt_area)
    
    if len(compare_dict) > num_of_fish:
        new_comparedict = {}
        copy_list = cntareas.copy()
        for i in range( 0, num_of_fish, 1):
            max_value = max(copy_list)
            cnt_id    = cntids[cntareas.index(max_value)]
            copy_list.remove(max_value)
            new_comparedict[cnt_id] = compare_dict[cnt_id]
        compare_dict = new_comparedict


    return compare_dict


def cntid_assignment( compare_dict, contours, next_contours):
    cntid_list    = list(compare_dict.keys())
    nextcntid_all = []
    for i in range( 0, len(compare_dict), 1):
        cnt_id = cntid_list[i]
        nextcntid_all += compare_dict[cnt_id]

    max_size    = max( len(cntid_list), len(nextcntid_all))
    cost_matrix = np.empty((max_size,max_size))
    cost_matrix[:,:] = -1 
    max_value   = 0
    for i in range( 0, len(compare_dict), 1):
        cnt_id = cntid_list[i]
        cnt    = contours[i]
        cnt_m  = cv2.moments(cnt)
        cnt_cx = int(cnt_m['m10']/cnt_m['m00'])
        cnt_cy = int(cnt_m['m01']/cnt_m['m00'])

        nextcntid_list = compare_dict[cnt_id]
        for j in range( 0, len(nextcntid_list), 1):
            nextcnt_id = nextcntid_list[j]
            nextcnt    = next_contours[nextcnt_id]
            nextcnt_m  = cv2.moments(nextcnt)
            nextcnt_cx = int(nextcnt_m['m10']/nextcnt_m['m00'])
            nextcnt_cy = int(nextcnt_m['m01']/nextcnt_m['m00'])

            dist  = np.sqrt(np.square(cnt_cx-nextcnt_cx)+np.square(cnt_cy-nextcnt_cy))
            index = nextcntid_all.index(nextcnt_id)
            cost_matrix[ i, index] = dist
            max_value = max(max_value, dist)
    
    cost_matrix[ cost_matrix == -1] = max_value + 1 

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    compare_result = {}
    for i in range( 0, len(compare_dict), 1):
        cnt_id = cntid_list[i]
        nextcnt_id = nextcntid_all[col_ind[i]]
        compare_result[cnt_id] = nextcnt_id


    return compare_result


def find_freeze( compare_dict, contours, next_contours, num_of_lostfish, frame, nextframe):
    cmp_cntid_list     = []
    cmp_nextcntid_list = []
    compare_dict_keys = list(compare_dict.keys())
    for i in range( 0, len(compare_dict), 1):
        cnt_id = compare_dict_keys[i]
        nextcnt_id = compare_dict[cnt_id]
        cmp_cntid_list.append(cnt_id)
        cmp_nextcntid_list.append(nextcnt_id)
    cmp_cntid_list.sort()
    cmp_nextcntid_list.sort()

    sum_b = 0
    sum_g = 0
    sum_r = 0 
    for i in range( 0, len(cmp_cntid_list), 1):
        cnt_id  = cmp_cntid_list[i]
        cnt_bgr = ImgProcessing.compute_contour_bgr( frame, contours, cnt_id)
        sum_b = sum_b + cnt_bgr[0]
        sum_g = sum_g + cnt_bgr[1]
        sum_r = sum_r + cnt_bgr[2]
    avg_b = sum_b / len(cmp_cntid_list)
    avg_g = sum_g / len(cmp_cntid_list)
    avg_r = sum_r / len(cmp_cntid_list)
    avg_bgr = [avg_b,avg_g,avg_r]

    filter_cntid_list = []
    for i in range( 0, len(contours), 1):
        cnt_id = i
        if cnt_id in cmp_cntid_list:
            continue
        cnt_bgr    = ImgProcessing.compute_contour_bgr( frame, contours, cnt_id)
        bool_color = Functions.filter_contour_bgr( avg_bgr, cnt_bgr)
        if bool_color == True:
            filter_cntid_list.append(cnt_id)

    old_filter_cntid_list = filter_cntid_list.copy()
    filter_cntid_list     = []
    for i in range( 0, len(old_filter_cntid_list), 1):
        cnt_id = old_filter_cntid_list[i]
        if cnt_id in cmp_cntid_list:
            continue
        cnt         = contours[cnt_id]
        cnt_rect    = cv2.minAreaRect(cnt)  
        rect_width  = min(cnt_rect[1][0],cnt_rect[1][1])
        rect_length = max(cnt_rect[1][0],cnt_rect[1][1])
        if rect_length < rect_width * 6: 
            filter_cntid_list.append(cnt_id)

    filter_nextcntid_list = []
    for i in range( 0, len(filter_cntid_list), 1):
        filter_cntid = filter_cntid_list[i]
        filter_cnt   = contours[filter_cntid]
        for j in range( 0, len(next_contours), 1):
            nextcnt_id = j
            if nextcnt_id in cmp_nextcntid_list:
                continue
            nextcnt = next_contours[nextcnt_id]
            bool_intersect = Functions.two_contourIntersect( nextframe, filter_cnt, nextcnt)
            if bool_intersect == True:
                filter_nextcntid_list.append(nextcnt_id)
    filter_nextcntid_list = list(set(filter_nextcntid_list))
    filter_nextcntid_list.sort()

    filtercnt_arealist = []
    for i in range( 0, len(filter_cntid_list), 1):
        filter_cntid   = filter_cntid_list[i]
        filter_cnt     = contours[filter_cntid]
        filter_cntarea = cv2.contourArea(filter_cnt)
        filtercnt_arealist.append(filter_cntarea)

    filter_maxareaid = []
    for i in range( 0, num_of_lostfish, 1):
        max_index = filtercnt_arealist.index(max(filtercnt_arealist))
        filtercnt_arealist.pop(max_index)
        filter_cntid = filter_cntid_list[max_index]
        filter_cntid_list.remove(filter_cntid)
        filter_maxareaid.append(filter_cntid)
    filter_maxareaid.sort()


    freeze_cmp = {}
    for i in range( 0, len(filter_maxareaid), 1):
        cnt_id = filter_maxareaid[i]
        cnt    = contours[cnt_id]
        intersect_list = []
        for j in range( 0, len(filter_nextcntid_list), 1):
            nextcnt_id = filter_nextcntid_list[j]
            nextcnt    = next_contours[nextcnt_id]
            bool_intersect = Functions.two_contourIntersect( nextframe, cnt, nextcnt)
            if bool_intersect == True:
                intersect_list.append(nextcnt_id)
        if len(intersect_list) == 1:
            freeze_cmp[cnt_id] = intersect_list[0]
        else: 
            max_nextcntid   = None
            max_nextcntarea = 0
            for k in range( 0, len(intersect_list), 1):
                nextcnt_id = intersect_list[k]
                nextcnt    = next_contours[nextcnt_id]
                bool_intersect, intersection_area = Functions.two_contourIntersect_area( nextframe, cnt, nextcnt)
                if bool_intersect == True:
                    if intersection_area > max_nextcntarea:
                        max_nextcntid   = nextcnt_id
                        max_nextcntarea = intersection_area
            freeze_cmp[cnt_id] = max_nextcntid

    compare_dict.update(freeze_cmp)


    return compare_dict


def create_fishdict( compare_dict, contours, next_contours, frame_index, cnt_mot_cmptable):
    fish_dict       = {} 
    num_id          = 0
    fishid_nextcnt  = {}
    fishid_movement = {}

    cntid_list = list(compare_dict.keys())
    for i in range( 0, len(compare_dict), 1):
        details    = {}
        cnt_id     = cntid_list[i]
        cnt        = contours[cnt_id]
        nextcnt_id = compare_dict[cnt_id]
        nextcnt    = next_contours[nextcnt_id]

        cnt_m  = cv2.moments(cnt)
        cnt_cx = int(cnt_m["m10"]/cnt_m["m00"])
        cnt_cy = int(cnt_m['m01']/cnt_m['m00'])

        nextcnt_m  = cv2.moments(nextcnt)
        nextcnt_cx = int(nextcnt_m["m10"]/nextcnt_m["m00"])
        nextcnt_cy = int(nextcnt_m['m01']/nextcnt_m['m00'])

        line_angle = DataProcessing.getAngleBetweenPoints( cnt_cx, cnt_cy, nextcnt_cx, nextcnt_cy)

        if cnt_id in list(cnt_mot_cmptable.keys()):
            mhi_circle_x = round( cnt_mot_cmptable[cnt_id]["Circle_X"], 2) 
            mhi_circle_y = round( cnt_mot_cmptable[cnt_id]["Circle_Y"], 2)
            mhi_circle_r = round( cnt_mot_cmptable[cnt_id]["Radius"], 2)
            mhi_circle_angle = round( cnt_mot_cmptable[cnt_id]["Angle"], 2)
        else:
            mhi_circle_x = None
            mhi_circle_y = None
            mhi_circle_r = None
            mhi_circle_angle = None

        details["ContourID"]              = cnt_id
        details["ContourIndexPoint"]      = (cnt_cx,cnt_cy)
        details["NextContourID"]          = nextcnt_id
        details["NextContourInedexPoint"] = (nextcnt_cx,nextcnt_cy)
        details["LineAngle"]              = round( line_angle, 2)
        details["Intersect"]              = None
        details["MHI_Circle_X"]           = mhi_circle_x
        details["MHI_Circle_Y"]           = mhi_circle_y
        details["MHI_Circle_R"]           = mhi_circle_r
        details["MHI_Circle_Angle"]       = mhi_circle_angle

        fishid = "Fish"+str(num_id)
        frame_index_str = "FrameIndex"+str(frame_index)
        if fishid not in list(fish_dict.keys()):
            fish_dict[fishid] = {}
        fish_dict[fishid][frame_index_str] = details
        fishid_nextcnt[fishid]             = nextcnt_id
        fishid_movement[fishid]            = "1"

        num_id = num_id + 1


    return fish_dict, fishid_movement


def compute_avgfish_area( frame_index, frame_dict, fish_dict, contours, next_contours):
    fisharea_dict  = {}
    cntid_list     = []
    nextcntid_list = []
    
    fish_ids = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fish_id    = fish_ids[i]
        cnt_id     = fish_dict[fish_id]["FrameIndex"+str(frame_index)]["ContourID"]
        nextcnt_id = fish_dict[fish_id]["FrameIndex"+str(frame_index)]["NextContourID"]

        cntid_list.append(cnt_id)
        nextcntid_list.append(nextcnt_id)

    cnt_times     = dict((x, cntid_list.count(x)) for x in set(cntid_list)) 
    nextcnt_times = dict((x, nextcntid_list.count(x)) for x in set(nextcntid_list)) 

    sum_cont = 0
    sum_b, sum_g, sum_r = 0, 0, 0
    for i in range( 0, len(fish_dict), 1):
        sum_cntarea = 0
        cont        = 0
        fish_id     = fish_ids[i]
        cnt_id      = fish_dict[fish_id]["FrameIndex"+str(frame_index)]["ContourID"]
        nextcnt_id  = fish_dict[fish_id]["FrameIndex"+str(frame_index)]["NextContourID"]
        if cnt_times[cnt_id] == 1: 
            cnt         = contours[cnt_id]
            cnt_area    = cv2.contourArea(cnt)
            cnt_area    = round( cnt_area, 2)
            sum_cntarea = sum_cntarea + cnt_area
            cont        = cont + 1

            img      = frame_dict[frame_index].copy()
            cnt_bgr  = ImgProcessing.compute_contour_bgr( img, contours, cnt_id)
            sum_b    = sum_b + cnt_bgr[0]
            sum_g    = sum_g + cnt_bgr[1]
            sum_r    = sum_r + cnt_bgr[2]

        if nextcnt_times[nextcnt_id] == 1:
            nextcnt      = next_contours[nextcnt_id]
            nextcnt_area = cv2.contourArea(nextcnt)
            sum_cntarea  = sum_cntarea + nextcnt_area
            cont         = cont + 1
        
            nextimg     = frame_dict[frame_index+1].copy()
            nextcnt_bgr = ImgProcessing.compute_contour_bgr( nextimg, next_contours, nextcnt_id)
            sum_b       = sum_b + nextcnt_bgr[0]
            sum_g       = sum_g + nextcnt_bgr[1]
            sum_r       = sum_r + nextcnt_bgr[2]

        sum_cont = sum_cont + cont 
        fisharea_dict[fish_id] = round( sum_cntarea / cont, 2)

    avgfish_bgr  = [ int(sum_b/sum_cont), int(sum_g/sum_cont), int(sum_r/sum_cont)]

    return fisharea_dict, avgfish_bgr