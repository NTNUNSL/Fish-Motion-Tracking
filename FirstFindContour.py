import cv2

import ImgProcessing
import Functions


def mhiroi_findcnt( frame, frame_index, motions, multi, boundary, block_size, offset):
    if frame_index <= 2:
        new_roi, roi_frame = compute_roi_motion( frame, motions, multi, boundary)
        roi_frame = frame[boundary[0][1]:boundary[1][1],boundary[0][0]:boundary[1][0]]
        roi_minx  = boundary[0][0]
        roi_miny  = boundary[0][1]
    elif frame_index > 2:
        new_roi, roi_frame = compute_roi_motion( frame, motions, multi, boundary)
        roi_minx = new_roi[0][0]
        roi_miny = new_roi[0][1]
    gray     = cv2.cvtColor( roi_frame, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(gray, 5) 

    if frame_index <= 2:
        thresh = ImgProcessing.threshold_mode( "threshold_OTSU_INV", blur_img, None, None, None) 
    elif frame_index > 2:
        thresh = ImgProcessing.threshold_mode( "threshold_ADP", blur_img, None, block_size, offset)

    thresh = ImgProcessing.morphology( thresh, 'Opening', 'Rect', 3, 2)
    contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = Functions.contour_coordinate_transform( contours, roi_minx, roi_miny)

    if frame_index <= 2:
        filter_contours = contours
    elif frame_index > 2:
        filter_contours = filter_contour_motion( contours, new_roi)

    
    return filter_contours


def compute_roi_motion( frame, motions, multi, boundary):
    roi_minx, roi_maxx = 0, 0
    roi_miny, roi_maxy = 0, 0
    for i in range( 0, len(motions), 1):
        x, y   = motions[i]["X"], motions[i]["Y"]
        w, h   = motions[i]["Width"], motions[i]["Height"]
        rx     = x + w
        ry     = y + h
        if i == 0 :
            roi_minx, roi_maxx =  x, rx
            roi_miny, roi_maxy =  y, ry
        
        roi_minx = min(  x, roi_minx)
        roi_miny = min(  y, roi_miny)
        roi_maxx = max( rx, roi_maxx)
        roi_maxy = max( ry, roi_maxy)

    range_x = (roi_maxx - roi_minx)* multi 
    range_y = (roi_maxy - roi_miny)* multi
    roi_minx, roi_maxx = (roi_minx-range_x),(roi_maxx+range_x)
    roi_miny, roi_maxy = (roi_miny-range_y),(roi_maxy+range_y)


    roi_minx = int(max( roi_minx, boundary[0][0]))
    roi_miny = int(max( roi_miny, boundary[0][1]))
    roi_maxx = int(min( roi_maxx, boundary[1][0]))
    roi_maxy = int(min( roi_maxy, boundary[1][1]))

    if roi_minx > roi_maxx:
        roi_minx = boundary[0][0]
        roi_maxx = boundary[1][0]
    if roi_miny > roi_maxy:
        roi_miny = boundary[0][1]
        roi_maxy = boundary[1][1]


    new_roi   = [[roi_minx,roi_miny], [roi_maxx,roi_maxy]]
    roi_frame = frame[roi_miny:roi_maxy,roi_minx:roi_maxx].copy()

    return new_roi, roi_frame


def filter_contour_motion( contours, roi): 
    filter_contour = []
    roi_minx, roi_miny = roi[0][0], roi[0][1]
    roi_maxx, roi_maxy = roi[1][0], roi[1][1]
    roi_sub_x = int( roi_maxx - roi_minx)
    roi_sub_y = int( roi_maxy - roi_miny)
    roi_long  = max( roi_sub_x, roi_sub_y)
    roi_short = min( roi_sub_x, roi_sub_y)
    roi_area  = roi_long * roi_short

    limit = 0.7 
    for i in range( 0, len(contours), 1):
        cnt_area = cv2.contourArea(contours[i])
        if cnt_area < roi_area * 0.001 : 
            continue

        rect       = cv2.minAreaRect(contours[i])
        rect_long  = max(rect[1][0], rect[1][1])
        rect_short = min(rect[1][0], rect[1][1])
        if rect_long >= roi_long * limit:
            continue
        if rect_short >= roi_short * limit:
            continue

        percent_limit = 7.5
        rect_p = round( (rect_long/rect_short), 3)
        if rect_p > percent_limit:
            continue

        filter_contour.append(contours[i])


    return filter_contour





