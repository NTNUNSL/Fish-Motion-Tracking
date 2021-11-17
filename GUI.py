

def initialization():
    videofolder_path = 'C:/Users/user/Downloads/FishTracktor_handover/'
    video_name, num_of_fish, boundary = 'Fish4-1',int(4),[(230,200),(1650,1040)]



    video_path = videofolder_path + video_name + ".mp4" 
    

    return num_of_fish, video_name, video_path, boundary
