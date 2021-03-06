cy, cx = ndi.center_of_mass(resize_img1)
#Shape features
contours, image= cv2.findContours(resize_img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
index = find_contour(resize_img1,contours)
cnt = contours[index]
M = cv2.moments(cnt)
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)
        #rect = cv2.minAreaRect(cnt)
        #lw_val = Physilogical_width_and_length(rect)
        #w = lw_val[0]
        #h = lw_val[1]
x,y,w,h = cv2.boundingRect(cnt)
ecc = eccentricity(cnt)
diameter = diamater(cnt)
aspect_ratio = h/w
rectangularity = area/(w*h)
circularity = (4 * np.pi *area)/((perimeter)**2)
compactness = ((perimeter)**2)/area
NF = diameter/h
Perimeter_ratio_diameter = perimeter/diameter
Perimeter_ratio_length = perimeter/h
Perimeter_ratio_lw = perimeter/(h+w)
Hull = cv2.convexHull(cnt)
No_of_Convex_points = len(Hull)
perimeter_convex = cv2.arcLength(Hull,True)
area_convex = cv2.contourArea(Hull)
perimeter_convexity = perimeter_convex/perimeter
area_convexity = (area_convex-area)/area
area_ratio_convexity = area/area_convex
equivalent_diameter = np.sqrt(4*area/np.pi)
