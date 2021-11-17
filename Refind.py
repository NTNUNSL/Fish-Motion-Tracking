import cv2

import Functions
import ImgProcessing


def check_fish_allin( fish_dict, unionfishid_list):
    cont         = 0
    unionfishids = []
    fishid_list  = list(fish_dict.keys())
    for i in range( 0, len(unionfishid_list), 1):
        if len(unionfishid_list[i-cont][1]) > 0:
            unionfishids = unionfishids + unionfishid_list[i-cont][0]
        else:
            unionfishid_list.pop(i-cont)
            cont = cont + 1
    
    miss_fishids = []
    miss_fishids = list(set(fishid_list)-set(unionfishids))
    if len(miss_fishids) > 0:
        pass

    return miss_fishids, unionfishid_list


def refind_fish_surface( frame_index, fish_dict, miss_fishids, unionfishid_list, contours, next_contours, nextframe, avgfish_bgr):
    succe_nextcntids = [] 
    for i in range( 0, len(unionfishid_list), 1):
        if len(unionfishid_list[i][0]) > 0: 
            succe_nextcntids = succe_nextcntids + unionfishid_list[i][1]

    compare_new  = {}
    for i in range( 0, len(miss_fishids), 1):
        fishid = miss_fishids[i] 
        cnt_id = fish_dict[fishid]["FrameIndex"+str(frame_index-1)]["NextContourID"]
        cnt    = contours[cnt_id]

        intersect_ids = []
        for j in range( 0, len(next_contours), 1):
            nextcnt_id     = j
            nextcnt        = next_contours[nextcnt_id]
            bool_intersect = Functions.two_contourIntersect( nextframe, cnt, nextcnt)
            if bool_intersect == True:
                intersect_ids.append(nextcnt_id)
        
        cont = 0
        rectlong_limit = 0.5
        for j in range( 0, len(intersect_ids), 1):
            nextcnt_id       = intersect_ids[j-cont]
            nextcnt          = next_contours[nextcnt_id]
            cnt_rect         = cv2.minAreaRect(cnt)
            cnt_rectlong     = max( cnt_rect[1][0], cnt_rect[1][1])
            nextcnt_rect     = cv2.minAreaRect(nextcnt)
            nextcnt_rectlong = max( nextcnt_rect[1][0], nextcnt_rect[1][1]) 
            if cnt_rectlong < nextcnt_rectlong * rectlong_limit:
                intersect_ids.remove(nextcnt_id)
                cont = cont + 1

        cont = 0
        for j in range( 0, len(intersect_ids), 1):
            nextcnt_id  = intersect_ids[j-cont]
            nextcnt     = next_contours[nextcnt_id]
            nextcnt_bgr = ImgProcessing.compute_contour_bgr( nextframe, next_contours, nextcnt_id)
            bool_color  = Functions.filter_contour_bgr( avgfish_bgr, nextcnt_bgr)
            if bool_color == False: 
                intersect_ids.remove(nextcnt_id)
                cont = cont + 1

        cont = 0
        for j in range( 0, len(intersect_ids), 1):
            nextcnt_id = intersect_ids[j-cont]
            if nextcnt_id in succe_nextcntids:
                intersect_ids.remove(nextcnt_id)
                cont = cont + 1

        compare_new[fishid] = intersect_ids

    comparenew_keys = list(compare_new.keys())
    for i in range( 0, len(compare_new), 1):
        fishid     = comparenew_keys[i]
        nextcntids = compare_new[fishid]
        
        if len(nextcntids) >0:
            unionfishid_list.extend([[[fishid], nextcntids]])
            miss_fishids.remove(fishid)


    return unionfishid_list, miss_fishids


def refind_fish_deep( frame_index, fish_dict, miss_fishids, unionfishid_list, contours, next_contours, frame, nextframe, boundary):
    nextframe_c = nextframe.copy()
    succe_nextcntids = [] 
    for i in range( 0, len(unionfishid_list), 1):
        succe_nextcntids = succe_nextcntids + unionfishid_list[i][1]

    newfind_contours = {}
    for i in range( 0, len(miss_fishids), 1):
        fishid = miss_fishids[i]
        cnt_id = fish_dict[fishid]["FrameIndex"+str(frame_index-1)]["NextContourID"]
        cnt    = contours[cnt_id]

        multiple = 1
        roi_frame ,roi_coordinate = Functions.compute_roi_cntrect( nextframe_c, cnt, multiple, boundary)

        cnt_rect     = cv2.minAreaRect(cnt)
        cnt_rectlong = max( cnt_rect[1][0], cnt_rect[1][1])
        block_size   = int(cnt_rectlong/2)
        offset       = int(block_size/10)

        gamma  = ImgProcessing.auto_gamma_correction(roi_frame)
        gray   = cv2.cvtColor( gamma, cv2.COLOR_BGR2GRAY)
        thresh = ImgProcessing.threshold_mode( "threshold_ADP", gray, None, block_size, offset)
        thresh = ImgProcessing.morphology( thresh, 'Closing', 'Rect', 3, 2)
        thresh = ImgProcessing.morphology( thresh, 'Opening', 'Rect', 3, 2)
        new_contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = Functions.contour_coordinate_transform( new_contours, roi_coordinate['minx'], roi_coordinate['miny'])

        cont = 0        
        for j in range( 0, len(new_contours), 1):
            newcnt = new_contours[j-cont]
            bool_intersect = Functions.two_contourIntersect( nextframe, cnt, newcnt)
            if bool_intersect == False:
                new_contours.pop(j-cont)
                cont = cont + 1

        cont = 0
        cnt_bgr = ImgProcessing.compute_contour_bgr( frame, contours, cnt_id)
        for j in range( 0, len(new_contours), 1):
            newcnt_bgr = ImgProcessing.compute_contour_bgr( nextframe, new_contours, j-cont)
            bool_color = Functions.filter_contour_bgr( cnt_bgr, newcnt_bgr)
            if bool_color == False:
                new_contours.pop(j-cont)
                cont = cont + 1

        cont = 0
        percent_limit = 7.5
        for j in range( 0, len(new_contours), 1):
            newcnt     = new_contours[j-cont]
            rect       = cv2.minAreaRect(newcnt)
            rect_long  = max(rect[1][0], rect[1][1])
            rect_short = min(rect[1][0], rect[1][1])
            rect_p     = round( (rect_long/rect_short), 3)
            if rect_p > percent_limit:
                new_contours.pop(j-cont)
                cont = cont + 1

        useoldcnt_fishids = []
        if len(new_contours) > 1:
            for j in range( 0, len(new_contours), 1):
                newcnt = new_contours[j]
                bool_intersect, intersection_area = Functions.two_contourIntersect_area( nextframe, cnt, newcnt)
                if j == 0:
                    intersection_maxarea = intersection_area
                    determine_cntindex   = j
                
                if intersection_area > intersection_maxarea:
                    intersection_maxarea = intersection_area
                    determine_cntindex   = j
            newfind_contours[(fishid,)] = new_contours[determine_cntindex]

        elif len(new_contours) == 0:
            newfind_contours[(fishid,)] = contours[cnt_id]
            useoldcnt_fishids.append(fishid)
        else:
            newfind_contours[(fishid,)] = new_contours[0]

    filter_contours = {}
    combine_list    = []
    newfind_keys    = list(newfind_contours.keys())
    for i in range( 0, len(newfind_contours), 1):
        fishid_list1 = newfind_keys[i]
        newcnt_1     = newfind_contours[fishid_list1] 
        if i in combine_list:
            continue
        fishid_listcombine = fishid_list1
        for j in range( 0, len(newfind_contours), 1):
            if i == j or i > j:
                continue
            
            fishid_list2   = newfind_keys[j]
            newcnt_2       = newfind_contours[fishid_list2]
            bool_intersect = Functions.two_contourIntersect( nextframe, newcnt_1, newcnt_2)
            if bool_intersect == True:
                fishid_listcombine = fishid_listcombine + fishid_list2
                combine_list.append(j)
        filter_contours[fishid_listcombine] = newcnt_1

    newfind_keys = list(filter_contours.keys())
    for i in range( 0, len(filter_contours), 1):
        fishid_list    = newfind_keys[i]
        newcnt         = filter_contours[fishid_list]
        nextcnt_index  = len(next_contours)
        nextcntid_list = [nextcnt_index]
        next_contours.append(newcnt)

        unionfishid_list.extend([[ list(fishid_list), nextcntid_list]]) 
        for j in range( 0, len(fishid_list), 1):
            fishid = fishid_list[j]
            miss_fishids.remove(fishid)



    return unionfishid_list, next_contours, useoldcnt_fishids


