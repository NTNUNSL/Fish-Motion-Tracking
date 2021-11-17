import Functions
import ImgProcessing
import DataProcessing

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def bool_rematchid( unionid_list, newcnt_ids):
    newid_setlist = []
    test_drawids  = []
    for i in range( 0, len(unionid_list), 1):
        unionid_set = unionid_list[i]
        cntids      = unionid_set[0]

        set_diff = list(set(cntids).difference(set(newcnt_ids))) 
        if len(cntids) == len(set_diff):
            pass
        elif len(cntids) != len(set_diff) and len(set_diff) > 0:
            pass
        elif len(cntids) != len(set_diff) and len(set_diff) == 0:
            newid_setlist.extend([(unionid_set)])
            test_drawids = test_drawids + cntids

    if len(newid_setlist) == 0:
        bool_rematch = False
    else:
        bool_rematch = True
    
    return bool_rematch, newid_setlist


def bool_rematchid_nextcnt( newid_setlist, contours, next_contours, nextframe, avgfish_bgr):
    rm_newid_setlist = []
    for i in range( 0, len(newid_setlist), 1):
        newid_set  = newid_setlist[i]
        cntids     = newid_set[0]
        nextcntids = newid_set[1]        

        cont = 0
        for j in range( 0, len(nextcntids), 1):
            nextcntid   = nextcntids[j-cont]
            nextcnt_bgr = ImgProcessing.compute_contour_bgr( nextframe, next_contours, nextcntid)
            bool_color  = Functions.filter_contour_bgr( avgfish_bgr, nextcnt_bgr)
            if bool_color == False:
                nextcntids.remove(nextcntid)
                cont = cont + 1
        newid_setlist[i][1] = nextcntids
        if len(nextcntids) == 0: 
            newid_setlist.pop(newid_set)
            rm_newid_setlist.extend([newid_set])
            continue
    
    for i in range( 0, len(newid_setlist), 1):
        newid_set  = newid_setlist[i]
        cntids     = newid_set[0]
        nextcntids = newid_set[1]

        cont  = 0
        for j in range( 0, len(nextcntids), 1):
            nextcntid = nextcntids[j-cont]
            nextcnt   = next_contours[nextcntid]
            intersect_time = 0
            for k in range( 0, len(cntids), 1):
                cntid = cntids[k]
                cnt   = contours[cntid]
                bool_intersect = Functions.two_contourIntersect( nextframe, cnt, nextcnt)
                if bool_intersect == True:
                    intersect_time = intersect_time + 1
            if intersect_time == 0:
                nextcntids.remove(nextcntid)
                cont = cont + 1
        newid_setlist[i][1] = nextcntids
        if len(nextcntids) == 0:
            newid_setlist.pop(newid_set)
            rm_newid_setlist.extend([newid_set])
            continue
    
    if len(newid_setlist) == 0:
        bool_rematch = False
    else:
        bool_rematch = True
    

    return bool_rematch, newid_setlist, rm_newid_setlist


def remove_newids_notfish( rm_newid_setlist, compare_dict):
    for i in range( 0, len(rm_newid_setlist), 1):
        rm_newid_set = rm_newid_setlist[i]
        rm_newcntids = rm_newid_set[0]

        for j in range( 0, len(rm_newcntids), 1):
            cntid = rm_newcntids[j]
            if cntid in list(compare_dict.keys()):
                del compare_dict[cntid]
        

    return compare_dict


def fishid_rematch_cntid_less( frame_index, fish_dict, compare_dict, contours, history_contour_dict):
    pre_frameindex      = frame_index-1
    pre_contours        = history_contour_dict[pre_frameindex]
    fishid_precntid     = {}
    fishid_prenextcntid = {}
    fishid_list         = list(fish_dict.keys())
    for i in range( 0, len(fishid_list), 1):
        fish_id                      = fishid_list[i]
        fish_detail                  = fish_dict[fish_id]['FrameIndex'+str(pre_frameindex)]
        cnt_id                       = fish_detail['ContourID']
        nextcnt_id                   = fish_detail['NextContourID']
        fishid_precntid[fish_id]     = cnt_id
        fishid_prenextcntid[fish_id] = nextcnt_id
    precntid_list = list(fishid_precntid.values())
    precntid_list.sort()

    cntid_list = list(compare_dict.keys())
    cntid_list.sort()

    cost_matrix = create_costmatrix( fishid_precntid, cntid_list, pre_contours, contours)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    compare_result = {}
    pre_fishid_cntid_keys = list(fishid_precntid.keys())
    for i in range( 0, len(pre_fishid_cntid_keys), 1):
        fish_id = pre_fishid_cntid_keys[i]
        cnt_id  = cntid_list[col_ind[i]]
        compare_result[fish_id] = cnt_id

    bool_same      = True
    change_fishids = []
    pre_fishid_cntid_keys = list(fishid_precntid.keys())
    for i in range( 0, len(pre_fishid_cntid_keys), 1):
        fish_id   = pre_fishid_cntid_keys[i]
        ori_cntid = fishid_prenextcntid[fish_id]
        aft_cntid = compare_result[fish_id]
        if ori_cntid != aft_cntid:
            bool_same = False
            change_fishids.append(fish_id)


    return compare_result, bool_same, change_fishids

def fishid_rematch_cntid( frame_index, frame_dict, history_contour_dict, fish_dict, contours):
    pre_frameindex     = frame_index - 1
    pre_frameindex_str = 'FrameIndex'+str(pre_frameindex)
    pre_contours       = history_contour_dict[pre_frameindex]

    pre_fishid_cntid     = {} 
    pre_fishid_nextcntid = {} 
    pre_cntid_fishid     = {} 
    fishdict_keys        = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fish_id = fishdict_keys[i]
        fish_detail_all = fish_dict[fish_id]
        
        fish_detail                   = fish_detail_all[pre_frameindex_str]
        pre_cntid                     = fish_detail['ContourID']
        pre_nextcntid                 = fish_detail['NextContourID']
        pre_fishid_cntid[fish_id]     = pre_cntid
        pre_fishid_nextcntid[fish_id] = pre_nextcntid
        
        if pre_cntid not in list(pre_cntid_fishid.keys()):
            pre_cntid_fishid[pre_cntid] = []
        pre_cntid_fishid[pre_cntid].append(fish_id)
    

    avg_fisharea = compute_avgfisharea_by_fishdict( frame_index-1, fish_dict, history_contour_dict)

    cnts_percent, cnts_oripercent = check_contour_percent( contours, avg_fisharea)

    cntid_list = create_cntid_list( pre_fishid_cntid, cnts_percent)

    cost_matrix = create_costmatrix( pre_fishid_cntid, cntid_list, pre_contours, contours)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    compare_result = {}
    pre_fishid_cntid_keys = list(pre_fishid_cntid.keys())
    for i in range( 0, len(pre_fishid_cntid_keys), 1):
        fish_id = pre_fishid_cntid_keys[i]
        cnt_id = cntid_list[col_ind[i]]
        compare_result[fish_id] = cnt_id

    bool_same      = True
    change_fishids = []
    pre_fishid_cntid_keys = list(pre_fishid_cntid.keys())
    for i in range( 0, len(pre_fishid_cntid_keys), 1):
        fish_id   = pre_fishid_cntid_keys[i]
        ori_cntid = pre_fishid_nextcntid[fish_id]
        aft_cntid = compare_result[fish_id]
        if ori_cntid != aft_cntid:
            bool_same = False
            change_fishids.append(fish_id)


    return compare_result, bool_same, change_fishids


def create_costmatrix( fishid_cntid, cntid_list, pre_contours, contours):
    max_size    = max( len(fishid_cntid), len(cntid_list))
    cost_matrix = np.empty((max_size,max_size))
    cost_matrix[:,:] = -1
    max_value = 0
    fishid_cnt_keys = list(fishid_cntid.keys())
    for i in range( 0, len(fishid_cntid), 1):
        fish_id   = fishid_cnt_keys[i]
        precnt_id = fishid_cntid[fish_id]
        precnt    = pre_contours[precnt_id]
        precnt_m  = cv2.moments(precnt)
        precnt_cx = int(precnt_m['m10']/precnt_m['m00'])
        precnt_cy = int(precnt_m['m01']/precnt_m['m00'])

        for j in range( 0, len(cntid_list), 1):
            cnt_id = cntid_list[j]
            cnt    = contours[cnt_id]
            cnt_m  = cv2.moments(cnt)
            cnt_cx = int(cnt_m['m10']/cnt_m['m00'])
            cnt_cy = int(cnt_m['m01']/cnt_m['m00'])
        
            dist = np.sqrt(np.square(precnt_cx-cnt_cx)+np.square(precnt_cy-cnt_cy))
            cost_matrix[i,j] = dist
            max_value = max(max_value,dist)

    cost_matrix[ cost_matrix == -1] = max_value + 1

    return cost_matrix



def update_new_compare( compare_result, change_fishids, frame_index, fish_dict, fisharea_dict, avgfish_bgr, frame, contours):
    cntid_fishid = {}
    compare_result_keys = list(compare_result.keys())
    for i in range( 0, len(compare_result), 1):
        fish_id = compare_result_keys[i]
        cnt_id  = compare_result[fish_id]
        if cnt_id not in list(cntid_fishid.keys()):
            cntid_fishid[cnt_id] = []
        cntid_fishid[cnt_id].append(fish_id)

    for i in range( 0, len(change_fishids), 1):
        fish_id  = change_fishids[i]
        cnt_id   = compare_result[fish_id]
        cnt      = contours[cnt_id]
        cnt_area = cv2.contourArea(cnt)
        
        fish_detail       = fish_dict[fish_id]['FrameIndex'+str(frame_index)]
        precnt_coordinate = fish_detail['ContourIndexPoint']
        precnt_cx = precnt_coordinate[0]
        precnt_cy = precnt_coordinate[1]

        cnt_m  = cv2.moments(cnt)
        cnt_cx = int(cnt_m["m10"]/cnt_m["m00"]) 
        cnt_cy = int(cnt_m['m01']/cnt_m['m00']) 
        line_angle = DataProcessing.getAngleBetweenPoints( precnt_cx, precnt_cy, cnt_cx, cnt_cy)
        
        if len(cntid_fishid[cnt_id]) > 1:
            intersect_fishid = cntid_fishid[cnt_id]
        else:
            intersect_fishid = None

        fish_detail['NextContourID'] = cnt_id
        fish_detail['NextContourInedexPoint'] = (cnt_cx,cnt_cy)
        fish_detail['LineAngle'] = round( line_angle, 2)
        fish_detail['Intersect'] = intersect_fishid
        fish_dict[fish_id]['FrameIndex'+str(frame_index)] = fish_detail 

        if intersect_fishid != None:
            fisharea_dict[fish_id] = cnt_area

    sum_b, sum_g, sum_r = 0, 0, 0
    cntid_fishid_keys   = list(cntid_fishid.keys())
    for i in range( 0, len(cntid_fishid), 1):
        cnt_id  = cntid_fishid_keys[i]
        cnt_bgr = ImgProcessing.compute_contour_bgr( frame, contours, cnt_id)
        sum_b   = sum_b + cnt_bgr[0]
        sum_g   = sum_g + cnt_bgr[1]
        sum_r   = sum_r + cnt_bgr[2]

    num_cnt     = len(cntid_fishid)
    avgfish_bgr = [ int(sum_b/num_cnt), int(sum_g/num_cnt), int(sum_r/num_cnt)]


    return fish_dict, fisharea_dict, avgfish_bgr


def compute_avgfisharea_by_fishdict( frame_index, fish_dict, history_contour_dict):
    frameindex_str = 'FrameIndex'+str(frame_index)
    cntid_list     = []
    fishdict_keys  = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fish_id         = fishdict_keys[i]
        fish_detail_all = fish_dict[fish_id]

        fish_detail = fish_detail_all[frameindex_str]
        cntid       = fish_detail['ContourID']
        cntid_list.append(cntid)
    
    sum_fisharea = 0
    cntid_list   = list(set(cntid_list))
    contours     = history_contour_dict[frame_index]
    for i in range( 0, len(cntid_list), 1):
        cntid        = cntid_list[i]
        cnt          = contours[cntid]
        cnt_area     = cv2.contourArea(cnt)
        sum_fisharea = sum_fisharea + cnt_area
    avg_fisharea = round( sum_fisharea / len(fishdict_keys), 2)

    return avg_fisharea


def check_contour_percent( contours, avg_fisharea):
    cnts_percent    = {}
    cnts_oripercent = {}
    for i in range( 0, len(contours), 1):
        cnt_id      = i
        cnt         = contours[cnt_id]
        cnt_area    = cv2.contourArea(cnt)
        cnt_percent = round( cnt_area/avg_fisharea, 2)
        cnts_oripercent[cnt_id] = cnt_percent
        
        if 0.3  < cnt_percent and cnt_percent < 1: 
            cnt_percent = 1
        cnts_percent[cnt_id] = cnt_percent

    
    return cnts_percent, cnts_oripercent


def create_cntid_list( fishid_cntid, cnts_percent):
    cntid_list = []
    for i in range( 0, len(cnts_percent), 1):
        cnt_id  = i
        percent = int(cnts_percent[cnt_id])
        for j in range( 0, percent, 1):
            cntid_list.append(cnt_id)


    if len(fishid_cntid) > len(cntid_list):
        less_num = len(fishid_cntid) - len(cntid_list)
        arealess_list = {}
        for i in range( 0, len(cnts_percent), 1):
            cnt_id = i
            percent_ori = cnts_percent[cnt_id]
            percent_int = int(cnts_percent[cnt_id])
            arealess_list[cnt_id] = max( 0, percent_ori-percent_int)
        arealess_list = dict(sorted(arealess_list.items(), key=lambda item: item[1], reverse= True))

        cntid_oldlist = cntid_list.copy()
        arealess_list_keys = list(arealess_list.keys())
        for i in range( 0, less_num, 1):
            index = i
            if index >= len(cnts_percent):
                index = int(index - len(cnts_percent))
            cnt_id = arealess_list_keys[index]
            cntid_list.append(cnt_id)
        cntid_list.sort()


    return cntid_list


