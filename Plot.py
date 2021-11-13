import cv2
import numpy as np
 
def write_frameIndex( frame, frame_index, coordinate1, coordinate2):
    img = frame.copy()
    cv2.rectangle( img, coordinate1, coordinate2, (0,0,0), -1)

    text_x = coordinate1[0]+10
    text_y = int((coordinate1[1]+coordinate2[1])/2+10)
    cv2.putText( img , str(frame_index), (text_x,text_y), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 

    return img


def draw_fishid_contours( frame, cntid_dict, cnt_list, fishcolor_dict, r):
    img = frame.copy()
    
    cntiddict_keys = list(cntid_dict.keys())
    for i in range( 0, len(cntid_dict), 1):
        cnt_id      = cntiddict_keys[i]
        cnt         = cnt_list[cnt_id]
        fishid_list = cntid_dict[cnt_id]
        if len(fishid_list) == 1:
            fish_id    = fishid_list[0]
            fish_num   = fish_id.replace("Fish","")
            fish_color = fishcolor_dict[fish_id]
            status     = 'single'

        elif len(fishid_list) > 0:
            fishid_str = ""
            for j in range( 0, len(fishid_list), 1):
                fish_id    = fishid_list[j]
                fish_num   = fish_id.replace("Fish","")
                fishid_str = fishid_str + fish_num
                if j < len(fishid_list)-1:
                    fishid_str = fishid_str + " "
            
            fish_color = ( 0, 0, 255)
            fish_num   = fishid_str
            status     = 'interlace'

        cv2.drawContours( img, cnt_list, cnt_id, fish_color, 2)
        write_fishid_text( img, cnt, r, fish_color, fish_num, status)

    return img


def draw_fishid_mhi( frame, fishdict_details):
    img = frame.copy()
    
    circle_color = (255,0,0)
    line_color   = (255,0,0)
    
    fishdictdetail_keys = list(fishdict_details.keys())
    for i in range( 0, len(fishdict_details), 1):
        fish_id     = fishdictdetail_keys[i]
        fish_detail = fishdict_details[fish_id]
        
        if fish_detail["MHI_Circle_X"] == None and fish_detail["MHI_Circle_Y"] == None:
            continue
        mhi_cx    = int(fish_detail["MHI_Circle_X"])
        mhi_cy    = int(fish_detail["MHI_Circle_Y"])
        mhi_cr    = int(fish_detail["MHI_Circle_R"])
        mhi_angle = fish_detail["MHI_Circle_Angle"]

        circle_coordinate = ( mhi_cx, mhi_cy)
        linep_coordinate  = (int(mhi_cx+np.cos(mhi_angle)*mhi_cr),int(mhi_cy+np.sin(mhi_angle)*mhi_cr))

        cv2.circle( img, circle_coordinate, mhi_cr, circle_color, 3)
        cv2.line( img, circle_coordinate, linep_coordinate, line_color, 3)

    return img


def write_fishid_text( img, cnt, r, color, fish_id, status):
    rect = cv2.minAreaRect(cnt)
    cx   = int(rect[0][0])
    cy   = int(rect[0][1])
    if status == 'single':
        r2 = int(r/2)
        cv2.circle( img, (cx,cy), r, color, -1)
        cv2.putText( img, fish_id, (cx-r2,cy+r2), cv2.FONT_HERSHEY_DUPLEX, 0.5, ( 0, 0, 0), 2, cv2.LINE_AA)
    
    elif status == 'interlace':
        axis_s = r
        axis_l = int(len(fish_id)/2*r)
        cv2.ellipse( img, (cx,cy), (axis_l,axis_s), 0, 0, 360, color,  -1)
        cv2.putText( img, fish_id, (cx+int(-0.75*axis_l), cy+int(axis_s/2)), cv2.FONT_HERSHEY_DUPLEX, 0.5, ( 0, 0, 0), 2, cv2.LINE_AA)


