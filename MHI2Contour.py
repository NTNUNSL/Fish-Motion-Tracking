import numpy as np
import cv2

import Functions
import ImgProcessing



def mhi_cnt_filter( frame, contours, motions, avgfish_bgr, boundary):
    bool_contour = np.full( len(contours), False) 

    no_cmp_mot        = [] 
    cmp_mot_cnt_table = {}
    for i in range( 0, len(motions), 1):
        mot_index = i
        mot_x, mot_y = motions[i]["X"], motions[i]["Y"] 
        mot_w, mot_h = motions[i]["Width"], motions[i]["Height"]
        mot_box = [[mot_x,mot_y],[mot_x+mot_w,mot_y],[mot_x+mot_w,mot_y+mot_h],[mot_x,mot_y+mot_h]]        

        cnt_list = []
        bool_intersect = False
        for j in range( 0, len(contours), 1):
            cnt_rect = cv2.minAreaRect(contours[j])
            cnt_box  = cv2.boxPoints(cnt_rect)
            intersect = Functions.two_shape_intersect( cnt_box, 'Rect', mot_box, 'Rect')
            if bool_intersect == False and intersect == True:
                bool_intersect = True

            if intersect == True: 
                cnt_list.append(j)
                if bool_contour[j] == False: 
                    bool_contour[j] = True

        if bool_intersect == False:
            no_cmp_mot.append(i)

        elif bool_intersect == True:
            cmp_mot_cnt_table[mot_index] = {"ContourIDs": cnt_list}
    
    if len(no_cmp_mot) != 0:
        num_of_cnt = len(contours)
        temp_contour, temp_cmp_mot_cnt_table = region_detect_contour( frame, no_cmp_mot, motions, num_of_cnt, avgfish_bgr, boundary)

        contours = contours + temp_contour 
        cmp_mot_cnt_table.update(temp_cmp_mot_cnt_table)

    cmp_mot_cnt_table = dict(sorted(cmp_mot_cnt_table.items())) 
 
    
    return contours, cmp_mot_cnt_table


def region_detect_contour( frame, no_cmp_mot, motions, num_of_cnt, avgfish_bgr, boundary): 
    scale_rate  = 0.6
    sorted(no_cmp_mot)
    cmp_mot, cmp_mot_box = Functions.mhis_intersect( frame, motions, no_cmp_mot, scale_rate, boundary) 


    cmp_cnt_mot_table = []
    temp_contour_list = []
    for i in range( 0, len(cmp_mot_box), 1): 
        roi_minx, roi_miny = int(cmp_mot_box[i][0][0]), int(cmp_mot_box[i][0][1])
        roi_maxx, roi_maxy = int(cmp_mot_box[i][1][0]), int(cmp_mot_box[i][1][1])
        roi_width  = int( roi_maxx - roi_minx)
        roi_high   = int( roi_maxy - roi_miny)
        roi_area   = roi_width * roi_high
        roi_long   = max( roi_width, roi_high)
        roi_short  = min( roi_width, roi_high)
        if roi_minx == roi_maxx or roi_miny == roi_maxy:
            continue

        roi_frame           = frame[ roi_miny:roi_maxy, roi_minx:roi_maxx].copy()
        roi_frame           = ImgProcessing.auto_gamma_correction( roi_frame)
        gray                = cv2.cvtColor( roi_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh         = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
        morph               = ImgProcessing.morphology( thresh, 'Closing', 'Rect', 5, 3)        
        contours, hierarchy = cv2.findContours( morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for j in range( 0, len(contours), 1):
            bool_intersect = False
            contours[j][:]+= [ int(roi_minx), int(roi_miny)] 
            cnt_area       = cv2.contourArea(contours[j])  
            cnt_rect       = cv2.minAreaRect(contours[j]) 
            cnt_rect_width = cnt_rect[1][0]
            cnt_rect_high  = cnt_rect[1][1]
            cnt_rect_long  = max( cnt_rect_width, cnt_rect_high)
            cnt_rect_short = min( cnt_rect_width, cnt_rect_high)
            cnt_box        = cv2.boxPoints(cnt_rect) 

            if cnt_area < roi_area * 0.01 :
                continue
            limit = 0.7
            if cnt_rect_long >= roi_long * limit: 
                continue
            if cnt_rect_short >= roi_short * limit: 
                continue
            
            if avgfish_bgr != None:
                cnt_bgr = ImgProcessing.compute_contour_bgr( frame, contours, j)
                bool_color = Functions.filter_contour_bgr( avgfish_bgr, cnt_bgr)
                if bool_color == False: 
                    continue

            mhi_list  = []
            for k in range( 0, len(cmp_mot[i]), 1):
                mot_index    = cmp_mot[i][k]
                mot_x, mot_y = motions[mot_index]["X"], motions[mot_index]["Y"] 
                mot_w, mot_h = motions[mot_index]["Width"], motions[mot_index]["Height"]
                mot_box = [[mot_x,mot_y],[mot_x+mot_w,mot_y],[mot_x+mot_w,mot_y+mot_h],[mot_x,mot_y+mot_h]]
                intersect = Functions.two_shape_intersect( cnt_box, 'Rect', mot_box, 'Rect')
                if intersect == True:
                    bool_intersect = True
                    mhi_list.append(mot_index)

            if bool_intersect == True:
                temp_contour_list.append(contours[j])
                cmp_cnt_mot_table.extend([( num_of_cnt, mhi_list)])
                num_of_cnt        = num_of_cnt + 1


    cmp_mot_cnt_table = {}
    for i in range( 0, len(no_cmp_mot), 1):
        cnt_list = []
        mhi_id   = no_cmp_mot[i]
        bool_in  = False
        for j in range( 0, len(cmp_cnt_mot_table), 1):
            if mhi_id in cmp_cnt_mot_table[j][1]:
                bool_in = True
                cnt_list.append(cmp_cnt_mot_table[j][0])

        if bool_in == True:
            cmp_mot_cnt_table[mhi_id] = {"ContourIDs": cnt_list}


    return temp_contour_list, cmp_mot_cnt_table