import cv2
import numpy as np
import random

import shapely.affinity
from shapely.geometry import Polygon

import DataProcessing

def two_shape_intersect( shape1, shape1_type, shape2, shape2_type):
    mode = ''
    if shape1_type == 'Rect':
        rect1 = shape1
        mode  += 'r'
        if shape2_type == 'Rect':
            rect2 = shape2
            mode += 'r'
        elif shape2_type == 'Circle':
            circle1 = shape2
            mode += 'c'
    elif shape1_type == 'Circle':
        circle1 = shape1
        mode += 'c'
        if shape2_type == 'Rect':
            rect1 = shape2
            mode += 'r'
        elif shape2_type == 'Circle':
            circle2 = shape2
            mode += 'c'
    mode =''.join(sorted(mode))

    if mode == 'rr':
        p1 = Polygon(rect1)
        p2 = Polygon(rect2)
        intersect = p1.intersects(p2)
    
    return intersect


def mhis_intersect( frame, motions, no_cmp_mot, scale_rate, boundary):
    b_minx, b_miny = boundary[0][0], boundary[0][1]
    b_maxx, b_maxy = boundary[1][0], boundary[1][1]
    region_list = []
    for i in range( 0, len(no_cmp_mot), 1):
        bool_intersect = False
        for j in range( 0, len(no_cmp_mot), 1):
            if i < j :
                temp_list  = [no_cmp_mot[i],no_cmp_mot[j]]
                motion_box = []
                for k in range( 0, len(temp_list), 1): # 
                    mot_id = temp_list[k]
                    mot_x, mot_y = motions[mot_id]["X"], motions[mot_id]["Y"]
                    mot_w, mot_h = motions[mot_id]["Width"], motions[mot_id]["Height"]
                    p1_x = max(min( mot_x-scale_rate*mot_w, b_maxx), b_minx)
                    p1_y = max(min( mot_y-scale_rate*mot_h, b_maxy), b_miny)
                    p2_x = max(min( mot_x+(1+scale_rate)*mot_w, b_maxx), b_minx)
                    p2_y = max(min( mot_y+(1+scale_rate)*mot_h, b_maxy), b_miny)
                    pbox = [[p1_x,p1_y],[p1_x,p2_y],[p2_x,p2_y],[p2_x,p1_y]]
                    motion_box.append(pbox)

                intersect = two_shape_intersect( motion_box[0], 'Rect', motion_box[1], 'Rect')
                if intersect == True:
                    region_list.append(( no_cmp_mot[i], no_cmp_mot[j]))
                    if bool_intersect == False:
                        bool_intersect = True

        if bool_intersect == False:
            region_list.append(( no_cmp_mot[i], no_cmp_mot[i]))
    
    res_list = []
    for i in range( 0, len(region_list), 1):
        id1 = region_list[i][0]
        id2 = region_list[i][1]
        bool1, index1 = DataProcessing.find_in_list( res_list, id1)
        bool2, index2 = DataProcessing.find_in_list( res_list, id2)
        if bool1 == True and bool2 == True:
            if index1 != index2:
                group1 = res_list[index1]
                group2 = res_list[index2]
                res_list.append(group1+group2)
                res_list.remove(group1)
                res_list.remove(group2)

        elif bool1 == True and bool2 == False:
            temp = res_list[index1]+(id2,)
            res_list[index1] = temp
        elif bool1 == False and bool2 == True:
            temp = res_list[index2]+(id1,)
            res_list[index2] = temp
        elif bool1 == False and bool2 == False:
            temp = region_list[i]
            res_list.append(temp)

    for i in range( 0, len(res_list), 1):
        res_list[i] = sorted(list(set(res_list[i])))

    roi_box_list = [] 
    for i in range( 0, len(res_list), 1):
        res_sublist = res_list[i]
        minx, miny  = 0, 0
        maxx, maxy  = 0, 0
        for j in range( 0, len(res_sublist), 1):
            res_index  = res_sublist[j]
            x, y, w, h = motions[res_index]["X"], motions[res_index]["Y"], motions[res_index]["Width"], motions[res_index]["Height"]
            p1_x = int(max(min( x-scale_rate*w, b_maxx), b_minx))
            p1_y = int(max(min( y-scale_rate*h, b_maxy), b_miny))
            p2_x = int(max(min( x+(1+scale_rate)*w , b_maxx), b_minx))
            p2_y = int(max(min( y+(1+scale_rate)*h , b_maxy), b_miny))
            if j == 0:
                minx = p1_x
                miny = p1_y
                maxx = p2_x
                maxy = p2_y
            elif j > 0:
                minx = min( p1_x, minx)
                miny = min( p1_y, miny)
                maxx = max( p2_x, maxx)
                maxy = max( p2_y, maxy)

        roi_box_list.append(((minx,miny),(maxx,maxy)))

    uni_roi_box_list  = []
    bool_intersect    = False
    bool_no_intersect = True
    for i in range( 0, len(roi_box_list), 1):
        for j in range( 0, len(roi_box_list), 1): 
            if i < j:
                box1_x1, box1_y1 = roi_box_list[i][0][0], roi_box_list[i][0][1]
                box1_x2, box1_y2 = roi_box_list[i][1][0], roi_box_list[i][1][1]
                box2_x1, box2_y1 = roi_box_list[j][0][0], roi_box_list[j][0][1]
                box2_x2, box2_y2 = roi_box_list[j][1][0], roi_box_list[j][1][1] 

                box1 = [[box1_x1,box1_y1],[box1_x1,box1_y2],[box1_x2,box1_y2],[box1_x2,box1_y1]]
                box2 = [[box2_x1,box2_y1],[box2_x1,box2_y2],[box2_x2,box2_y2],[box2_x2,box2_y1]]
                intersect = two_shape_intersect( box1, 'Rect', box2, 'Rect')
                if intersect == True:
                    bool_intersect    = True
                    bool_no_intersect = False
                    bool1, index1 = DataProcessing.find_in_list( uni_roi_box_list, i)
                    bool2, index2 = DataProcessing.find_in_list( uni_roi_box_list, j)
                    if bool1 == True and bool2 == True:
                        if index1 != index2:
                            group1 = uni_roi_box_list[index1]
                            group2 = uni_roi_box_list[index2]
                            uni_roi_box_list.append(group1+group2)
                            uni_roi_box_list.remove(group1)
                            uni_roi_box_list.remove(group2)

                    elif bool1 == True and bool2 == False:
                        temp = uni_roi_box_list[index1]+(j,)
                        uni_roi_box_list[index1] = temp

                    elif bool1 == False and bool2 == True:
                        temp = uni_roi_box_list[index2]+(i,)
                        uni_roi_box_list[index2] = temp

                    elif bool1 == False and bool2 == False:
                        temp = (i,j)
                        uni_roi_box_list.append(temp)
            
        if bool_no_intersect == True:
            uni_roi_box_list.append((i,i)) 

        elif bool_no_intersect == False:
            bool_no_intersect == True
    
    if bool_intersect == True:
        temp_roi_box_list = []
        temp_res_list     = []
        for i in range( 0, len(uni_roi_box_list), 1):
            temp_res = []
            for j in range( 0, len(uni_roi_box_list[i]), 1):
                temp_index = uni_roi_box_list[i][j]
                temp_res   = temp_res + res_list[temp_index]
                if j == 0:
                    minx, miny = roi_box_list[temp_index][0][0],roi_box_list[temp_index][0][1]
                    maxx, maxy = roi_box_list[temp_index][1][0],roi_box_list[temp_index][1][1]
                elif j > 0:
                    minx = min( minx, roi_box_list[temp_index][0][0])
                    miny = min( miny, roi_box_list[temp_index][0][1])
                    maxx = max( maxx, roi_box_list[temp_index][1][0])
                    maxy = max( maxy, roi_box_list[temp_index][1][1])
            temp_res = list(set(temp_res))
            temp_res.sort()

            temp_roi_box_list.append([(minx,miny),(maxx,maxy)])
            temp_res_list.append(temp_res)

        res_list     = temp_res_list
        roi_box_list = temp_roi_box_list

    return res_list, roi_box_list


def two_contourIntersect( original_image, contour1, contour2):
    contours = [contour1, contour2]

    blank  = np.zeros(original_image.shape[0:2])
    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    intersection = np.logical_and(image1, image2)

    return intersection.any()


def two_contourIntersect_area( original_image, contour1, contour2):
    contours = [contour1, contour2]

    blank = np.zeros(original_image.shape[0:2])

    image1 = cv2.drawContours(blank.copy(), contours, 0, 1, -1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1, -1)

    intersection = image1 + image2

    intersection_area = np.sum( intersection == 2)

    return intersection.any(), intersection_area


def filter_contour_bgr( cnt1_bgr, cnt2_bgr):
    BGRDiff_Limit = 25
    bool_color    = True 

    if abs(cnt1_bgr[1]-cnt2_bgr[1]) > BGRDiff_Limit:
        bool_color = False

    if abs(cnt1_bgr[2]-cnt2_bgr[2]) > BGRDiff_Limit:
        bool_color = False 

    return bool_color


def fishid_color_create( fishcolor_dict, fish_dict):
    pixel_deep    = 170
    fishdict_keys = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        color_b = random.randint(pixel_deep,255)
        color_g = random.randint(pixel_deep,255)
        color_r = random.randint(pixel_deep,255)
        while color_b <= 30 and color_g <= 30 and color_r <= 235:
            color_b = random.randint(pixel_deep,255)
            color_g = random.randint(pixel_deep,255)
            color_r = random.randint(pixel_deep,255)

        color  = ( color_b, color_g, color_r)
        fishid = fishdict_keys[i]
        fishcolor_dict[fishid] = color

    return fishcolor_dict

def fishid_color_update( fishcolor_dict, fish_dict):
    pixel_deep     = 170
    fishcolor_keys = list(fishcolor_dict.keys())
    fishdict_keys  = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fish_id = fishdict_keys[i]
        if fish_id in fishcolor_keys:
            pass
        elif fish_id not in fishcolor_keys:
            color_b = random.randint(pixel_deep,255)
            color_g = random.randint(pixel_deep,255)
            color_r = random.randint(pixel_deep,255)
            while color_b <= 30 and color_g <= 30 and color_r <= 235:
                color_b = random.randint(pixel_deep,255)
                color_g = random.randint(pixel_deep,255)
                color_r = random.randint(pixel_deep,255)
            color  = ( color_b, color_g, color_r)
            fishcolor_dict[fish_id] = color

    return fishcolor_dict


def create_fishid_cntid( fish_dict, frame_index):
    fishid_cnt  = {}
    cntid_fish  = {}
    fishid_list = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fishid = fishid_list[i]
        cnt_id = fish_dict[fishid]["FrameIndex"+str(frame_index-1)]["NextContourID"]
        fishid_cnt[fishid] = cnt_id
        if cnt_id not in list(cntid_fish.keys()):
            cntid_fish[cnt_id] = [fishid]
        else:
            cntid_fish[cnt_id].append(fishid)


    return fishid_cnt, cntid_fish


def transfer_cntid_fishid( unionid_list, cntid_fish):
    unionfishid_list = []
    for i in range( 0, len(unionid_list), 1):
        cntid_list     = unionid_list[i][0]
        nextcntid_list = unionid_list[i][1]
        fishid_list    = []
        for j in range( 0, len(cntid_list), 1):
            cntid = cntid_list[j]
            if cntid in list(cntid_fish.keys()): 
                fishids = cntid_fish[cntid]
            else:
                continue
            fishid_list = fishid_list + fishids

        fishid_list = list(set(fishid_list))
        fishid_list.sort()

        unionfishid_list.extend([[ fishid_list, nextcntid_list]])

    return unionfishid_list


def id_union( compare_dict):
    unionlist  = []
    cntid_keys = list(compare_dict.keys())
    for i in range( 0, len(cntid_keys), 1):
        cnt_id      = cntid_keys[i]
        nextcnt_ids = compare_dict[cnt_id]
        index_list  = []
        for j in range( 0, len(unionlist), 1):
            unionlist_nextcnts = unionlist[j][1]
            bool_in = set(nextcnt_ids) & set(unionlist_nextcnts)
            if bool_in :
                index_list.append(j)
        if len(index_list) == 0:
            unionlist.extend([[ [cnt_id],nextcnt_ids ]])
        elif len(index_list) > 0:
            new_cntids     = []
            new_nextcntids = []
            for j in range( len(index_list)-1, -1, -1):
                index = index_list[j]
                new_cntids = new_cntids + unionlist[index][0]
                new_nextcntids = new_nextcntids + unionlist[index][1]
                unionlist.pop(index)
            new_cntids     = new_cntids + [cnt_id]
            new_nextcntids = new_nextcntids + nextcnt_ids
            unionlist.extend([[ new_cntids,new_nextcntids ]])
            
    for i in range( 0, len(unionlist), 1):
        unionlist[i][0] = list(set(unionlist[i][0]))
        unionlist[i][1] = list(set(unionlist[i][1]))    

    return unionlist


def compute_roi_cntrect( frame, cnt, multiple, boundary):
    x, y, w, h = cv2.boundingRect(cnt)
    b_minx, b_miny = boundary[0][0], boundary[0][1]
    b_maxx, b_maxy = boundary[1][0], boundary[1][1]

    minx = int(max( x-multiple*w, b_minx))
    miny = int(max( y-multiple*h, b_miny))
    maxx = int(min( x+(1+multiple)*w, b_maxx))
    maxy = int(min( y+(1+multiple)*h, b_maxy))

    roi_coordinate = {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
    roi_frame = frame[miny:maxy, minx:maxx]
    
    return roi_frame, roi_coordinate


def contour_coordinate_transform( contours, minx, miny):
    for i in range( 0, len(contours), 1):
        contours[i][:] += [int(minx),int(miny)] 
    
    return contours


def fishid_union( unionfishid_list):
    unionlist = []
    for i in range( 0, len(unionfishid_list), 1):
        fish_ids    = unionfishid_list[i][0]
        nextcnt_ids = unionfishid_list[i][1]

        cont     = 0
        new_list = [[],[]]
        for j in range( 0, len(unionlist), 1):
            unionlist_detail = unionlist[j-cont]
    
            bool_in = set(nextcnt_ids) & set(unionlist_detail[1])
            if bool_in:
                new_list[0] = new_list[0] + unionlist_detail[0]
                new_list[1] = new_list[1] + unionlist_detail[1]
                unionlist.remove(unionlist_detail)
                cont = cont + 1
        new_list[0] = new_list[0] + fish_ids
        new_list[1] = new_list[1] + nextcnt_ids
        unionlist.append(new_list)

    for i in range( 0, len(unionlist), 1):
        unionlist[i][0] = list(set(unionlist[i][0]))
        unionlist[i][1] = list(set(unionlist[i][1]))

        unionlist[i][0].sort()
        unionlist[i][1].sort()

    return unionlist


def color_limit( old_p, new_p, grow_limit):
    diff_p = abs(old_p - new_p)
    grow_p = diff_p * grow_limit * 0.01
    min_p  = int(max(0, old_p - grow_p))
    max_p  = int(min(254, old_p + grow_p))
    new_p  = max( min_p, new_p)
    new_p  = min( max_p, new_p)

    return new_p

