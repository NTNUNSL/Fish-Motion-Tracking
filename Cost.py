import cv2
import numpy as np

import Functions
import DataProcessing


def check_contour_areapercent( fishid_cnt, unionfishid_list, fisharea_dict, contours, next_contours, nextframe, frame_index, boundary):
    nextcnt_percent = {}
    avgfish_area    = sum(list(fisharea_dict.values())) / len(fisharea_dict) 
    for i in range( 0, len(unionfishid_list), 1):
        fishids       = unionfishid_list[i][0]
        nextcntids    = unionfishid_list[i][1]
        nextcnt_areas = []
        for j in range( 0, len(nextcntids), 1):
            nextcntid    = nextcntids[j]
            nextcnt      = next_contours[nextcntid]
            nextcnt_area = cv2.contourArea(nextcnt)

            nextcnt_p = nextcnt_area/avgfish_area
            if 1 > nextcnt_p and nextcnt_p > 0.3:
                nextcnt_p = 1
            nextcnt_areas.append(nextcnt_p)

        num_fish = len(fishids)
        if num_fish == 1 and len(nextcntids) > 1:
            cnt_id   = fishid_cnt[fishids[0]]
            cnt      = contours[cnt_id]
            cnt_area = cv2.contourArea(cnt)

            intersection_maxpercent = 0
            for j in range( 0, len(nextcntids), 1):
                nextcnt_id   = nextcntids[j]
                nextcnt      = next_contours[nextcnt_id]
                nextcnt_area = cv2.contourArea(nextcnt)
                nextcnt_m    = cv2.moments(nextcnt)
                nextcnt_cy   = int(nextcnt_m["m01"] / nextcnt_m["m00"])

                surface_range = (boundary[1][1]-boundary[0][1])/20
                if (nextcnt_cy - boundary[0][1]) < surface_range:
                    pass

                bool_intersect, intersection_area = Functions.two_contourIntersect_area( nextframe, cnt, nextcnt)
                intersection_percent = intersection_area / cnt_area            
                if intersection_percent > intersection_maxpercent:
                    intersection_maxpercent = intersection_percent
                    determine_cntindex = j
                
            for j in range( 0, len(nextcnt_areas), 1):
                if j == determine_cntindex:
                    nextcnt_areas[j] = 1
                else:
                    nextcnt_areas[j] = 0
        elif len(fishids) == len(nextcntids) and len(fishids) == 2:
            if 0 in nextcnt_areas:
                pass
            else:
                for j in range( 0, len(nextcnt_areas), 1):
                    nextcnt_areas[j] = 1

        result = proportional( num_fish, nextcnt_areas)
        if sum(result) != num_fish:
            for j in range( 0, len( fishids), 1):
                cnt_id = fishid_cnt[fishids[j]]
            for j in range( 0, len(nextcntids), 1):
                nextcnt_id = nextcntids[j]

            nextcnt_areas = []
            for j in range( 0, len(nextcntids), 1):
                nextcntid    = nextcntids[j]
                nextcnt      = next_contours[nextcntid]
                nextcnt_area = cv2.contourArea(nextcnt)
                nextcnt_areas.append(nextcnt_area)
            copy_list = nextcnt_areas.copy()
            diff_num = sum(result)-num_fish
            if diff_num > 0:
                for j in range( 0, diff_num, 1):
                    min_value = min(copy_list)
                    index = nextcnt_areas.index(min_value)
                    result[index] = result[index]-1
                    copy_list.remove(min_value)

        for j in range( 0, len(nextcntids), 1):
            nextcntid = nextcntids[j]
            nextcnt_percent[nextcntid] = result[j]

    nextcnt_percent = dict(sorted(nextcnt_percent.items()))

    return nextcnt_percent


def proportional( nseats, votes): 
    quota=sum(votes)/(1.+nseats)
    frac=[vote/quota for vote in votes]
    res=[int(f) for f in frac]
    n=nseats-sum(res)
    if n==0: return res 
    if n<0 : return [min(x,nseats) for x in res]
    remainders=[ai-bi for ai,bi in zip(frac,res)]
    limit=sorted(remainders,reverse=True)[n-1]

    for i,r in enumerate(remainders):
        if r>=limit:
            res[i]+=1
            n-=1
            if n==0: return res

    raise


def nextcntid_costlist_create( nextcnt_percent):
    cost_nextcntid_list = []
    nextcntid_keys      = list(nextcnt_percent.keys())
    for i in range( 0, len(nextcntid_keys), 1):
        nextcnt_id   = nextcntid_keys[i]
        area_percent = nextcnt_percent[nextcnt_id]
        for j in range( 0, area_percent, 1):
            cost_nextcntid_list.append(nextcnt_id)
    
    return cost_nextcntid_list


def costmatrix_distance( fish_dict, unionfishid_list, cost_nextcntid_list, frame_index, next_contours):
    nextcntid_list = []
    for i in range( 0, len(unionfishid_list), 1):
        nextcntid_list = nextcntid_list + unionfishid_list[i][1]
    nextcntid_list.sort()

    fishid_list = list(fish_dict.keys())
    max_size    = max( len(fishid_list), len(cost_nextcntid_list))
    cost_matrix = np.empty((max_size,max_size))
    cost_matrix[:,:] = -1
    max_value   = 0
    
    for i in range( 0, len(fishid_list), 1):
        fish_id = fishid_list[i]
        cnt_x   = fish_dict[fish_id]["FrameIndex"+str(frame_index-1)]["NextContourInedexPoint"][0]
        cnt_y   = fish_dict[fish_id]["FrameIndex"+str(frame_index-1)]["NextContourInedexPoint"][1]
        
        for j in range( 0, len(nextcntid_list), 1):
            nextcnt_id = nextcntid_list[j]
            nextcnt    = next_contours[nextcnt_id]
            nextcnt_m  = cv2.moments(nextcnt)
            nextcnt_x  = int(nextcnt_m["m10"] / nextcnt_m["m00"])
            nextcnt_y  = int(nextcnt_m["m01"] / nextcnt_m["m00"])

            dist       = np.sqrt(np.square(cnt_x-nextcnt_x)+np.square(cnt_y-nextcnt_y))
            max_value  = max( dist, max_value)
            index_list = [i for i,val in enumerate(cost_nextcntid_list) if val== nextcnt_id]
            for k in range( 0, len(index_list), 1):
                index = index_list[k]
                cost_matrix[ i, index] = dist
    
    cost_matrix[ cost_matrix == -1] = max_value + 1


    return cost_matrix


def costmatrix_angle( fish_dict, unionfishid_list, cost_nextcntid_list, frame_index, contours, next_contours):
    frameindex_range = 5
    cost_dict        = {}
    fishid_list      = list(fish_dict.keys())
    for i in range( 0, len( fishid_list), 1):
        fish_id           = fishid_list[i]
        fish_detail       = fish_dict[fish_id]        
        frameindex_keys   = list(fish_detail.keys())
        frameindex_nums   = [s.replace('FrameIndex', '') for s in frameindex_keys]
        frameindex_nums   = list(map(int, frameindex_nums))
        frameindex_start  = max( max(frame_index - frameindex_range, 1), min(frameindex_nums))
        frameindex_end    = frame_index - 1 

        anglediff_sum = 0
        for j in range( frameindex_start, frameindex_end, 1):
            fish_angle1   = fish_detail["FrameIndex"+str(j)]["LineAngle"]
            fish_angle2   = fish_detail["FrameIndex"+str(j+1)]["LineAngle"]
            anglediff     = min(abs(fish_angle1-fish_angle2), abs((fish_angle1+360)-fish_angle2), abs((fish_angle2+360)-fish_angle1))
            anglediff_sum = anglediff_sum + anglediff
            if j == frameindex_start:
                angle_start = fish_angle1
            if j == frameindex_end-1:
                angle_last  = fish_angle2


        cnt_x = fish_detail["FrameIndex"+str(frameindex_end)]["NextContourInedexPoint"][0]
        cnt_y = fish_detail["FrameIndex"+str(frameindex_end)]["NextContourInedexPoint"][1]
        nextcntid_list = []
        for j in range( 0, len(unionfishid_list), 1):
            if fish_id in unionfishid_list[j][0]:
                nextcntid_list = unionfishid_list[j][1]
        
        for j in range( 0, len(nextcntid_list), 1):
            nextcnt_id    = nextcntid_list[j]
            nextcnt       = next_contours[nextcnt_id]
            nextcnt_m     = cv2.moments(nextcnt)
            nextcnt_x     = int(nextcnt_m["m10"]/nextcnt_m["m00"])
            nextcnt_y     = int(nextcnt_m['m01']/nextcnt_m['m00'])
            line_angle    = DataProcessing.getAngleBetweenPoints( cnt_x, cnt_y, nextcnt_x, nextcnt_y)
            anglediff     = min(abs(line_angle-angle_last), abs((line_angle+360)-angle_last), abs((angle_last+360)-line_angle))
            anglediff_sum = anglediff_sum + anglediff
            
            before_diff = anglediff_sum / (frameindex_end-frameindex_start +1)
            after_diff  = min(abs(angle_start-line_angle),abs((angle_start+360)-line_angle),abs((line_angle+360)-angle_start))

            if before_diff == 0: 
                before_diff = 1
            if after_diff == 0:
                after_diff = 1

            result_diff = 1/(before_diff/after_diff)

            if fish_id not in list(cost_dict.keys()):
                cost_dict[fish_id] = {}
            cost_dict[fish_id][nextcnt_id] = result_diff

    
    max_value        = 0
    max_size         = max( len(fishid_list), len(cost_nextcntid_list))
    cost_matrix      = np.empty((max_size,max_size))
    cost_matrix[:,:] = -1
    fishid_list      = list(cost_dict.keys())
    for i in range( 0, len(cost_dict), 1):
        fish_id      = fishid_list[i]
        fishid_num   = int(fish_id.replace("Fish",""))
        cost_detail  = cost_dict[fish_id]
        nextcnt_keys = list(cost_detail.keys())
        for j in range( 0, len(cost_detail), 1):
            nextcnt_id = nextcnt_keys[j]
            cost       = cost_detail[nextcnt_id]
            max_value  = max( cost, max_value)
            index_list = [i for i,x in enumerate(cost_nextcntid_list) if x == nextcnt_id]
            for k in range( 0, len(index_list), 1):
                index = index_list[k]
                cost_matrix[fishid_num][index] = cost
    cost_matrix[ cost_matrix == -1] = max_value + 1



    return cost_matrix

