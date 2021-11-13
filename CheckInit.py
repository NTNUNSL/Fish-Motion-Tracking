
def check_init( num_of_fish, fish_movent):
    successive_moves = 10
    compare_string   = ''
    for i in range( 0, successive_moves, 1):
        compare_string = compare_string + "1"

    init_fishid = {}
    if len(fish_movent) != num_of_fish:
        return False, None
    elif len(fish_movent) >= num_of_fish:
        cont = 0
        for i in range( 0, len(fish_movent), 1):
            fish_id     = "Fish"+str(i)
            temp_string = fish_movent[fish_id]
            if compare_string in temp_string:
                init_fishid[fish_id] = "Fish"+str(cont)
                cont = cont + 1

        if cont == num_of_fish:
            return True, init_fishid
        elif cont < num_of_fish:
            return False, None


def initdata_transform( init_fishid, fish_dict, fisharea_dict, fishcolor_dict):
    fish_dict      = transform_fishdict( init_fishid, fish_dict)
    fisharea_dict  = transform_fishareadict( init_fishid, fisharea_dict)
    fishcolor_dict = transform_fishcolordict( init_fishid, fishcolor_dict)


    return fish_dict, fisharea_dict, fishcolor_dict


def transform_fishdict( init_fishid, fish_dict):
    new_fishdict   = {}
    oldfishid_keys = list(fish_dict.keys())
    for i in range( 0, len(oldfishid_keys), 1):
        old_fishid = oldfishid_keys[i]
        if old_fishid not in list(init_fishid.keys()):
            continue
        new_fishid = init_fishid[old_fishid]
        
        new_details = {}
        frameindex_keys = list(fish_dict[old_fishid])
        for j in range( 0, len(frameindex_keys), 1):
            frame_index = frameindex_keys[j]
            old_details = fish_dict[old_fishid][frame_index]
            if old_details["Intersect"] != None:
                new_intersect_fishids = []
                intersect_fishids     = old_details["Intersect"]
                for k in range( 0, len(intersect_fishids), 1):
                    intersect_fishid = intersect_fishids[k]
                    if intersect_fishid in list(init_fishid.keys()):
                        new_intersect_fishids.append(init_fishid[intersect_fishid])

                new_intersect_fishids.sort()
                old_details["Intersect"] = new_intersect_fishids
            
            new_details[frame_index] = old_details
        new_fishdict[new_fishid] = new_details

    return new_fishdict


def transform_fishareadict( init_fishid, fisharea_dict):
    new_fisharea_dict = {}
    oldfishid_keys    = list(fisharea_dict.keys())
    for i in range( 0, len(oldfishid_keys), 1):
        old_fishid = oldfishid_keys[i]
        if old_fishid not in list(init_fishid.keys()):
            continue
        new_fishid   = init_fishid[old_fishid]
        old_fisharea = fisharea_dict[old_fishid]
        new_fisharea_dict[new_fishid] = old_fisharea

    return new_fisharea_dict


def transform_fishcolordict( init_fishid, fishcolor_dict):
    new_fishcolor_dict = {}
    oldfishid_keys     = list(fishcolor_dict.keys())
    for i in range( 0, len(oldfishid_keys), 1):
        old_fishid = oldfishid_keys[i]
        if old_fishid not in list(init_fishid.keys()):
            continue
        new_fishid    = init_fishid[old_fishid]
        old_fishcolor = fishcolor_dict[old_fishid]
        new_fishcolor_dict[new_fishid] = old_fishcolor

    return new_fishcolor_dict

