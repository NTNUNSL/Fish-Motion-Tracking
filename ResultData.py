import csv


def data_main( frame_index, fish_dict, video_name, folder_path):
    fishposition_csvpath = folder_path+'/'+video_name+'_tracked.csv'
    fishdict_to_csv( fishposition_csvpath, frame_index, fish_dict)

    

def fishdict_to_csv( csv_path, frame_index, fish_dict):
    csv_dict = {}
    frameindex_key = 'FrameIndex'+str(frame_index)

    csv_dict['FrameIndex'] = frame_index

    csv_fishdict  = {}
    attributes    = ['x','y']
    fishdict_keys = list(fish_dict.keys())
    for i in range( 0, len(fishdict_keys), 1):
        fish_id = fishdict_keys[i]
        for j in range( 0, len(attributes), 1):
            fish_attr_str = fish_id+'_'+attributes[j]

            get_data = {
                'x':fish_dict[fish_id][frameindex_key]['ContourIndexPoint'][0],
                'y':fish_dict[fish_id][frameindex_key]['ContourIndexPoint'][1],
            }
            fish_attr = get_data.get( attributes[j], None)
            csv_fishdict[fish_attr_str] = fish_attr
    csv_fishdict = dict(sorted(csv_fishdict.items()))

    csv_dict.update(csv_fishdict)
    
    with open( csv_path, 'a', newline='') as csvfile:
        fieldnames = list(csv_dict.keys())
        writer     = csv.DictWriter( csvfile, fieldnames= fieldnames)
        if frame_index == 1:
            writer.writeheader()

        writer.writerow(csv_dict)

    csvfile.close()
