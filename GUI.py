

def initialization():
    videofolder_path = 'C:/Users/user/Downloads/FishTracktor_handover/'
    video_name, num_of_fish, boundary = 'Fish',int(4),[(245,100),(1118,700)]

    video_path = videofolder_path + video_name + ".mp4"
    

    return num_of_fish, video_name, video_path, boundary
