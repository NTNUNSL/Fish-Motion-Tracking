import cv2
from scipy.optimize import linear_sum_assignment

import Functions
import Refind
import Rematch
import ImgProcessing
import DataProcessing
import Cost



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
        cnt_id           = cntid_list[i]
        cnt              = contours[cnt_id]
        nextcnt_idlidist = compare_dict[cnt_id]

        intersect_list = []
        for j in range( 0, len(nextcnt_idlidist), 1):
            nextcnt_id     = nextcnt_idlidist[j]
            nextcnt        = next_contours[nextcnt_id]
            bool_intersect = Functions.two_contourIntersect( nextframe, cnt, nextcnt)
            if bool_intersect == True:
                intersect_list.append(nextcnt_id)

        if len(intersect_list) > 0:
            compare_intersect[cnt_id] = intersect_list

    return compare_intersect


def cntid_checknew( frame_index, fish_dict, compare_dict, contours, next_contours, frame, nextframe, avgfish_bgr):
    fishid_cnt, cntid_fish = Functions.create_fishid_cntid( fish_dict, frame_index)

    cont = 0
    newcnt_ids = list(set(list(compare_dict.keys()))-set(list(cntid_fish.keys())))
    for i in range( 0, len(newcnt_ids), 1):
        newcnt_id  = newcnt_ids[i-cont]
        newcnt_bgr = ImgProcessing.compute_contour_bgr( frame, contours, newcnt_id)
        bool_color = Functions.filter_contour_bgr( avgfish_bgr, newcnt_bgr)
        if bool_color == False:
            del compare_dict[newcnt_id]
            newcnt_ids.remove(newcnt_id)
            cont = cont + 1
    if len(newcnt_ids) > 0:
        pass

    unionid_list = Functions.id_union(compare_dict)

    bool_rematch, newid_setlist = Rematch.bool_rematchid( unionid_list, newcnt_ids)
    if bool_rematch == True: 
        bool_rematch, newid_setlist, rm_newid_setlist = Rematch.bool_rematchid_nextcnt( newid_setlist, contours, next_contours, nextframe, avgfish_bgr)
        compare_dict = Rematch.remove_newids_notfish( rm_newid_setlist, compare_dict)


    return bool_rematch, compare_dict


def cntid_rematch(frame_index, frame_dict, history_contour_dict, fish_dict, fisharea_dict, avgfish_bgr, compare_dict, contours, next_contours, frame, nextframe):
    LookBackTime = 2

    if frame_index <= LookBackTime:
        compare_result, bool_same, change_fishids = Rematch.fishid_rematch_cntid_less( frame_index, fish_dict, compare_dict, contours, history_contour_dict)
    elif frame_index > LookBackTime:
        compare_result, bool_same, change_fishids = Rematch.fishid_rematch_cntid( frame_index, frame_dict, history_contour_dict, fish_dict, contours)
    
    if bool_same == False:
        pre_frameindex = frame_index - 1
        fish_dict, fisharea_dict, avgfish_bgr = Rematch.update_new_compare( compare_result, change_fishids, pre_frameindex, fish_dict, fisharea_dict, avgfish_bgr, frame, contours)
    

    return fish_dict, fisharea_dict, avgfish_bgr


def cntid_compare_union( frame_index, fish_dict, compare_dict):
    unionid_list           = Functions.id_union(compare_dict)
    fishid_cnt, cntid_fish = Functions.create_fishid_cntid( fish_dict, frame_index)
    unionfishid_list       = Functions.transfer_cntid_fishid( unionid_list, cntid_fish)
    
    return unionid_list, unionfishid_list


def check_comparedict( frame_index, fish_dict, unionfishid_list, contours, next_contours, frame, nextframe, avgfish_bgr, boundary):
    miss_fishids, unionfishid_list = Refind.check_fish_allin( fish_dict, unionfishid_list)
    useoldcnt_fishids = []
    if len(miss_fishids) > 0:
        unionfishid_list, miss_fishids = Refind.refind_fish_surface( frame_index, fish_dict, miss_fishids, unionfishid_list, contours, next_contours, nextframe, avgfish_bgr)

    if len(miss_fishids) > 0:
        unionfishid_list, next_contours, useoldcnt_fishids = Refind.refind_fish_deep( frame_index, fish_dict, miss_fishids, unionfishid_list, contours, next_contours, frame, nextframe, boundary)
        
    cont = 0
    for i in range( 0, len(unionfishid_list), 1):
        if len(unionfishid_list[i-cont][0]) == 0:
            unionfishid_list.pop(i-cont)
            cont = cont + 1

    unionfishid_list = Functions.fishid_union(unionfishid_list)


    return unionfishid_list, next_contours, useoldcnt_fishids


def fishid_cnt_assignment( frame_index, fish_dict, fisharea_dict, unionfishid_list, contours, next_contours, nextframe, boundary):
    fishid_cnt, cntid_fish = Functions.create_fishid_cntid( fish_dict, frame_index)

    nextcnt_percent     = Cost.check_contour_areapercent( fishid_cnt, unionfishid_list, fisharea_dict, contours, next_contours, nextframe, frame_index, boundary)
    cost_nextcntid_list = Cost.nextcntid_costlist_create(nextcnt_percent)

    if frame_index == 2:
        cost_matrix = Cost.costmatrix_distance( fish_dict, unionfishid_list, cost_nextcntid_list, frame_index, next_contours)
    elif frame_index > 2:
        cost_matrix = Cost.costmatrix_angle( fish_dict, unionfishid_list, cost_nextcntid_list, frame_index, contours, next_contours)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    compare_result = {}
    fishid_list    = list(fish_dict.keys())
    for i in range( 0, len(fishid_list), 1):
        fishid     = fishid_list[i]
        nextcnt_id = cost_nextcntid_list[col_ind[i]]
        compare_result[fishid] = nextcnt_id

    
    return compare_result


def create_fishdict( fish_dict, compare_result, contours, next_contours, frame_index, cnt_mot_cmptable):
    intersect_fishids = {} 
    compare_keys      = list( compare_result.keys())
    for i in range( 0, len(compare_result), 1):
        fishid     = compare_keys[i]
        nextcnt_id = compare_result[fishid]
        if nextcnt_id not in list(intersect_fishids.keys()):
            intersect_fishids[nextcnt_id] = [fishid]
        else:
            intersect_fishids[nextcnt_id].append(fishid)

    intersect_fishids_keys = list( intersect_fishids.keys())
    for i in range( 0, len(intersect_fishids), 1):
        nextcnt_id = intersect_fishids_keys[i]
        fishid_list = intersect_fishids[nextcnt_id]
        if len(fishid_list) == 1:
            intersect_fishids[nextcnt_id] = None

    
    for i in range( 0, len(compare_result), 1):
        details    = {}
        fishid     = compare_keys[i]
        cnt_id     = fish_dict[fishid]["FrameIndex"+str(frame_index-1)]["NextContourID"]
        cnt        = contours[cnt_id]
        nextcnt_id = compare_result[fishid]
        nextcnt    = next_contours[nextcnt_id]

        cnt_m  = cv2.moments(cnt)
        cnt_cx = int(cnt_m["m10"]/cnt_m["m00"])
        cnt_cy = int(cnt_m['m01']/cnt_m['m00'])

        nextcnt_m  = cv2.moments(nextcnt)
        nextcnt_cx = int(nextcnt_m["m10"]/nextcnt_m["m00"])
        nextcnt_cy = int(nextcnt_m['m01']/nextcnt_m['m00']) 

        line_angle = DataProcessing.getAngleBetweenPoints( cnt_cx, cnt_cy, nextcnt_cx, nextcnt_cy)

        details["ContourID"]              = cnt_id
        details["ContourIndexPoint"]      = (cnt_cx,cnt_cy)
        details["NextContourID"]          = nextcnt_id
        details["NextContourInedexPoint"] = (nextcnt_cx,nextcnt_cy)
        details["LineAngle"]              = round( line_angle, 2)
        details["Intersect"]              = intersect_fishids[nextcnt_id]

        if cnt_id in list(cnt_mot_cmptable.keys()):
            mhi_circle_x     = cnt_mot_cmptable[cnt_id]["Circle_X"]
            mhi_circle_y     = cnt_mot_cmptable[cnt_id]["Circle_Y"]
            mhi_circle_r     = cnt_mot_cmptable[cnt_id]["Radius"]
            mhi_circle_angle = cnt_mot_cmptable[cnt_id]["Angle"]
            details["MHI_Circle_X"]           = round( mhi_circle_x, 2)
            details["MHI_Circle_Y"]           = round( mhi_circle_y, 2)
            details["MHI_Circle_R"]           = round( mhi_circle_r, 2)
            details["MHI_Circle_Angle"]       = round( mhi_circle_angle, 2)
        else:
            details["MHI_Circle_X"]           = None
            details["MHI_Circle_Y"]           = None
            details["MHI_Circle_R"]           = None
            details["MHI_Circle_Angle"]       = None

        frame_index_str = "FrameIndex"+str(frame_index)
        fish_dict[fishid][frame_index_str] = details

    fishid_nextcnt = compare_result.copy()
    
    return fish_dict, fishid_nextcnt


def compute_avgfish_area( fisharea_dict, frame_index, frame_dict, fish_dict, next_contours, avgfish_bgr, useoldcnt_fishids): 
    fishid_cmpdict    = {}
    intersect_fishids = {}
    fishdict_keys     = list( fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fishid     = fishdict_keys[i]
        cnt_id     = fish_dict[fishid]["FrameIndex"+str(frame_index)]["ContourID"]
        nextcnt_id = fish_dict[fishid]["FrameIndex"+str(frame_index)]["NextContourID"]
        fishid_cmpdict[fishid] = [cnt_id,nextcnt_id]
        if nextcnt_id not in list(intersect_fishids.keys()):
            intersect_fishids[nextcnt_id] = [fishid]
        else:
            intersect_fishids[nextcnt_id].append(fishid)
    intersect_fishids_keys = list( intersect_fishids.keys())
    for i in range( 0, len(intersect_fishids), 1):
        nextcnt_id  = intersect_fishids_keys[i]
        fishid_list = intersect_fishids[nextcnt_id]
        if len(fishid_list) == 1:
            intersect_fishids[nextcnt_id] = None

    cont, sum_b, sum_g, sum_r = 0, 0, 0, 0
    fishid_list = list(fishid_cmpdict.keys())
    for i in range( 0, len(fishid_cmpdict), 1):
        fish_id     = fishid_list[i]
        cnt_id      = fishid_cmpdict[fish_id][0]
        nextcnt_id  = fishid_cmpdict[fish_id][1]

        if intersect_fishids[nextcnt_id] == None:
            nextcnt      = next_contours[nextcnt_id]
            nextcnt_area = cv2.contourArea(nextcnt)
            back_area    = fisharea_dict[fish_id]
            fisharea_dict[fish_id] = round((back_area + nextcnt_area)/2, 2)
        
            if fish_id not in useoldcnt_fishids:
                nextimg     = frame_dict[frame_index+1].copy()
                nextcnt_bgr = ImgProcessing.compute_contour_bgr( nextimg, next_contours, nextcnt_id)
                sum_b       = sum_b + nextcnt_bgr[0]
                sum_g       = sum_g + nextcnt_bgr[1]
                sum_r       = sum_r + nextcnt_bgr[2]
                cont        = cont  + 1


    cntarea_dict_values = list(fisharea_dict.values())
    avgfish_area        = round( sum(cntarea_dict_values) / len(fisharea_dict), 2)

    if cont == 0:
        avgfish_bgr = avgfish_bgr
    elif cont != 0: 
        grow_limit = 10
        old_b = avgfish_bgr[0]
        old_g = avgfish_bgr[1]
        old_r = avgfish_bgr[2]
        avg_b = int(sum_b/cont)
        avg_g = int(sum_g/cont)
        avg_r = int(sum_r/cont)
        avg_b = Functions.color_limit( old_b, avg_b, grow_limit)
        avg_g = Functions.color_limit( old_g, avg_g, grow_limit)
        avg_r = Functions.color_limit( old_r, avg_r, grow_limit)
        avgfish_bgr = [ avg_b, avg_g, avg_r]


    return fisharea_dict, avgfish_area, avgfish_bgr


