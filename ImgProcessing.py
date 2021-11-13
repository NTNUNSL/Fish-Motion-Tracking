import cv2
import numpy as np
import math

def threshold_mode( threshold_choose, gray_diff, thresh, block_size, offset):
    if threshold_choose == "threshold":
        ret, thresh = cv2.threshold( gray_diff, thresh, 1, cv2.THRESH_BINARY) 

    elif threshold_choose == "threshold_ADP":
        if block_size % 2 == 0:
            block_size += 1 
        thresh = cv2.adaptiveThreshold( gray_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, block_size, offset)

    elif threshold_choose == "threshold_OTSU":
        ret, thresh = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    
    elif threshold_choose == "threshold_OTSU_INV":
        ret, thresh = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

    return thresh


def morphology( frame, kernel_mode, kernel_shape, kernel_size, iterations):
    kernel_shape_all = {
        'Rect' :  cv2.getStructuringElement( cv2.MORPH_RECT,(kernel_size,kernel_size)),
        'Cross' : cv2.getStructuringElement( cv2.MORPH_CROSS,(kernel_size,kernel_size)),
        'Ellipse' : cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (kernel_size,kernel_size)) 
    }
    kernel_shape_choose = kernel_shape_all.get( kernel_shape, None)

    kernel_mode_all = {
        'Erosion' : cv2.erode( frame, kernel_shape_choose, iterations= iterations),
        'Dilation' : cv2.dilate( frame, kernel_shape_choose, iterations= iterations),
        'Opening' : cv2.morphologyEx( frame, cv2.MORPH_OPEN, kernel_shape_choose, iterations= iterations),
        'Closing' : cv2.morphologyEx( frame, cv2.MORPH_CLOSE, kernel_shape_choose, iterations= iterations)
    }
    morph  = kernel_mode_all.get( kernel_mode, None)

    return morph


def motion_history_image ( timestamp, thresh, frame_width, frame_height, motion_history, MHI_DURATION, MAX_TIME_DELTA, MIN_TIME_DELTA):
    cv2.motempl.updateMotionHistory( thresh, motion_history, timestamp, MHI_DURATION)

    mg_mask, mg_orient = cv2.motempl.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize= 5)

    seg_mask, seg_bounds = cv2.motempl.segmentMotion( motion_history, timestamp, MAX_TIME_DELTA)

    motions = []
    for i, rect in enumerate([(0, 0, frame_width, frame_height)] + list(seg_bounds)):
        x, y, w, h = rect 
        area = w * h

        silh_roi   = thresh        [y:y+h,x:x+w]
        orient_roi = mg_orient     [y:y+h,x:x+w]
        mask_roi   = mg_mask       [y:y+h,x:x+w] 
        mhi_roi    = motion_history[y:y+h,x:x+w]
        
        area_limit = 0.01
        if cv2.norm(silh_roi, cv2.NORM_L1) < area*area_limit: 
            continue

        angle  = cv2.motempl.calcGlobalOrientation( orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION) 
        r      = min( w//2, h//2)
        cx, cy = x+w//2 , y+h//2
        if i != 0:
            motions.append({"X": x,"Y":y,"Width": w,"Height": h,"Circle_X": cx,"Circle_Y": cy,"Radius": r,"Angle": angle})
            
    return motions


def remove_ccd_time(img):
    minx, miny = 1055, 0
    maxx, maxy = 1920, 70
    img[ miny:maxy, minx:maxx] = 0

    return img


def compute_contour_bgr( frame, contours, cnt_id):
    lst_intensities = []

    cimg = np.zeros_like(frame)
    cv2.drawContours(cimg, contours, cnt_id, color=255, thickness=-1)

    pts = np.where(cimg == 255)
    lst_intensities.append(frame[pts[0], pts[1]])

    sum_b,sum_g,sum_r = 0,0,0
    for i in range( 0, len(lst_intensities[0]), 1):
        b,g,r = lst_intensities[0][i][:]
        sum_b = sum_b + b
        sum_g = sum_g + g
        sum_r = sum_r + r
    avg_b = int(sum_b / len(lst_intensities[0]))
    avg_g = int(sum_g / len(lst_intensities[0]))
    avg_r = int(sum_r / len(lst_intensities[0]))

    
    return [avg_b,avg_g,avg_r]


def auto_gamma_correction(img):
    gray  = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    mean  = np.mean(gray)
    if mean == 255:
        mean = 254
    gamma = math.log10(0.5)/math.log10(mean/255)/2

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)

