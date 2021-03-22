import cv2

def eccentricity(contour):
    if(len(contour)>=5):
        (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
        a = ma/2
        b = MA/2
        ecc = np.sqrt(a**2 - b**2)/a
    else:
        ecc = 0 
    return ecc
