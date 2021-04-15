def Physilogical_width_and_length(rect):
    length = []
    #rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    if(angle < -45):
        angle = -(90 + angle)
        a = rect[1][0] #width
        b = rect[1][1] #height
        if(a < b):
            w = rect[1][1] #Major axis length
            h = rect[1][0] #Minor axis length
        else:
            w = rect[1][0] 
            h = rect[1][1]
    else:
        angle = -angle
        a = rect[1][0] #width
        b = rect[1][1] #height
        if(a < b):
            w = rect[1][1] #Major axis length
            h = rect[1][0] #Minor axis length
        else:
            w = rect[1][0] 
            h = rect[1][1]
    lw_val = [w,h] 
    return(lw_val)
