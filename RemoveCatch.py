
def remove_main( frame_dict, history_contour_dict, history_motion_dict, fish_dict, frame_index):
    frame_dict           = remove_framedict( frame_dict, frame_index)
    fish_dict            = remove_fish_dict( fish_dict, frame_index)
    history_contour_dict = remove_history_contour_dict( history_contour_dict, frame_index)
    history_motion_dict  = remove_history_motion_dict( history_motion_dict, frame_index)


    return frame_dict, history_contour_dict, history_motion_dict, fish_dict


def remove_framedict( frame_dict, frame_index):
    del frame_dict[frame_index]

    return frame_dict


def remove_fish_dict( fish_dict, frame_index):
    frameindex_str = 'FrameIndex'+str(frame_index)
    fish_ids = list(fish_dict.keys())
    for i in range( 0, len(fish_dict), 1):
        fish_id = fish_ids[i]
        del fish_dict[fish_id][frameindex_str]

    return fish_dict

def remove_history_contour_dict( history_contour_dict, frame_index):
    del history_contour_dict[frame_index]

    return history_contour_dict

def remove_history_motion_dict( history_motion_dict, frame_index):
    del history_motion_dict[frame_index]

    return history_motion_dict