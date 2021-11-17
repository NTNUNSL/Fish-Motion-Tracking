import cv2
from math import *

def getAngleBetweenPoints(x_orig, y_orig, x_target, y_target):
    deltaY = y_orig - y_target
    deltaX = x_orig - x_target

    tan   = atan2(deltaY, deltaX)
    angle = degrees(tan)
    if angle < 0:
    	angle += 360

    return angle


def update_blocksize( fish_dict, frame_index, next_contours, old_block_size): 
    nextcntid_list = []
    fishid_keys    = list(fish_dict.keys())
    frame_index    = "FrameIndex"+str(frame_index)
    for i in range( 0, len(fish_dict), 1):
        fish_id      = fishid_keys[i]
        nextcnt_id   = fish_dict[fish_id][frame_index]["NextContourID"]
        nextcntid_list.append(nextcnt_id)

    sum_rect_short = 0
    sum_rect_long  = 0
    nextcntid_list = list(set(nextcntid_list))
    for i in range( 0, len(nextcntid_list), 1):
        nextcnt_id   = nextcntid_list[i]
        nextcnt      = next_contours[nextcnt_id]
        nextcnt_rect = cv2.minAreaRect(nextcnt)
        sum_rect_long  += max( nextcnt_rect[1][0], nextcnt_rect[1][1])
        sum_rect_short += min( nextcnt_rect[1][0], nextcnt_rect[1][1])

    roi_rect_long  = int(sum_rect_long / len(nextcntid_list))
    new_block_size = int((roi_rect_long)/2) 
    if old_block_size != None:
        growup_limit = 0.2 
        if new_block_size > (old_block_size *(1 + growup_limit)):
            block_size = int(old_block_size * (1+growup_limit))
        else:
            block_size = new_block_size
    else:
        block_size = new_block_size

    offset = int(block_size/10) 

    default_blocksize = 45
    default_offset    = 4
    if block_size < default_blocksize:
        block_size = default_blocksize
        offset     = default_offset


    return block_size, offset


def find_in_list( mylist, elem): 
    for sub_list in mylist:
        if elem in sub_list:
            return True, mylist.index(sub_list)
    return False, None

