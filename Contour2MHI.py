import cv2

import Functions
import ImgProcessing


def cnt_mhi_filter( frame, contours, motions, avgfish_bgr, boundary):
    cmp_cnt_mot_table = {}
    no_cmp_mot        = list(range(0, len(motions)))
    for i in range( 0, len(contours), 1):
        cnt_rect = cv2.minAreaRect(contours[i])
        cnt_box  = cv2.boxPoints(cnt_rect)

        bool_intersect   = False
        motion_list      = []
        cx_sum, cy_sum   = 0, 0
        angle_sum, r_sum = 0, 0
        intersect_time   = 0
        for j in range( 0, len(motions), 1):
            mot_x, mot_y = motions[j]["X"], motions[j]["Y"]
            mot_w, mot_h = motions[j]["Width"], motions[j]["Height"]
            mot_box      = [[mot_x,mot_y],[mot_x+mot_w,mot_y],[mot_x+mot_w,mot_y+mot_h],[mot_x,mot_y+mot_h]]
            intersect    = Functions.two_shape_intersect( cnt_box, 'Rect', mot_box, 'Rect')            

            if bool_intersect == False and intersect == True :
                bool_intersect = True

            if intersect == True :
                motion_list.append(j)
                cx        = motions[j]["Circle_X"]
                cy        = motions[j]["Circle_Y"]
                r         = motions[j]["Radius"]
                angle     = motions[j]["Angle"]
                cx_sum    = cx_sum + cx
                cy_sum    = cy_sum + cy
                r_sum     = r_sum + r
                angle_sum = angle_sum + angle * r

                intersect_time += 1
                if j in no_cmp_mot:
                    no_cmp_mot.remove(j)

        if bool_intersect == True :
            cnt_id    = i
            cx        = cx_sum / intersect_time
            cy        = cy_sum / intersect_time
            r         = r_sum  / intersect_time
            angle     = angle_sum / r_sum
            cmp_cnt_mot_table[cnt_id] = {"MotionIDs": motion_list, "Circle_X": cx, "Circle_Y": cy, "Radius": r, "Angle": angle}

    if len(no_cmp_mot) != 0:
        num_of_cnt = len(contours)
        temp_contour, temp_cmp_cnt_mot_table = region_detect_contour( frame, no_cmp_mot, motions, num_of_cnt, avgfish_bgr, boundary)

        contours = contours + temp_contour
        cmp_cnt_mot_table.update(temp_cmp_cnt_mot_table)

    cmp_cnt_mot_table = dict(sorted(cmp_cnt_mot_table.items()))

    return contours, cmp_cnt_mot_table


def region_detect_contour( frame, no_cmp_mot, motions, num_of_cnt, avgfish_bgr, boundary): 
    scale_rate  = 0.6
    cmp_mot, cmp_mot_box = Functions.mhis_intersect( frame, motions, no_cmp_mot, scale_rate, boundary) 

    cmp_cnt_mot_table = {}
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
        roi_frame           = ImgProcessing.auto_gamma_correction(roi_frame)
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

            motion_list    = []
            cx_sum         = 0
            cy_sum         = 0
            r_sum          = 0
            angle_sum      = 0
            intersect_time = 0
            for k in range( 0, len(cmp_mot[i]), 1):
                mot_index   = cmp_mot[i][k]
                mot_x, mot_y = motions[mot_index]["X"], motions[mot_index]["Y"]
                mot_w, mot_h = motions[mot_index]["Width"], motions[mot_index]["Height"]
                mot_box   = [[mot_x,mot_y],[mot_x+mot_w,mot_y],[mot_x+mot_w,mot_y+mot_h],[mot_x,mot_y+mot_h]]
                intersect = Functions.two_shape_intersect( cnt_box, 'Rect', mot_box, 'Rect')

                if intersect == True:
                    bool_intersect = True
                    motion_list.append(mot_index)
                    cx_sum    = cx_sum + motions[mot_index]["Circle_X"]
                    cy_sum    = cy_sum + motions[mot_index]["Circle_Y"]
                    r_sum     = r_sum  + motions[mot_index]["Radius"]
                    angle_sum = angle_sum + (motions[mot_index]["Angle"] * motions[mot_index]["Radius"])

                    intersect_time += 1

            if bool_intersect == True:
                temp_contour_list.append(contours[j])

                cx    = cx_sum / intersect_time
                cy    = cy_sum / intersect_time
                r     = r_sum  / intersect_time
                angle = angle_sum / r_sum

                cmp_cnt_mot_table[num_of_cnt] = {"MotionIDs": motion_list, "Circle_X": cx, "Circle_Y": cy, "Radius": r, "Angle": angle}
                num_of_cnt = num_of_cnt + 1


    return temp_contour_list, cmp_cnt_mot_table