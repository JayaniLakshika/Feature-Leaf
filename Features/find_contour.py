import cv2

def find_contour(resize_img1,cnts):
    contains = []
    y_ri,x_ri = resize_img1.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp>0]
    #print(contains)
    if (len(val)==0):
        return(len(cnts)-1) #Best contour selected
    else:
        return val[0] #If center is in the contour
