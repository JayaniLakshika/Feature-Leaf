#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Numeric python library
import numpy as np
#To read images to numpy arrays
import matplotlib.image as mpimg
#To plot graphs
import matplotlib.pyplot as plt
#To draw a circle at the mean contour
import matplotlib.patches as mpatches
#To find shape contour
from skimage import measure
#To determine shape centrality
import scipy.ndimage as ndi
#To solve scientific and mathematical problems
import scipy as sp
#To calculate relative extrema of data
from scipy.signal import argrelextrema
#To manipulate data
import pandas as pd
#To measure classification performance
from sklearn import metrics


# In[2]:


import cv2


# In[3]:


import myutils


# In[4]:


from itertools import combinations


# In[5]:


#To determine shape centrality
import scipy.ndimage as ndi


# In[59]:


#To set the backend of matplot to the inline backend
get_ipython().run_line_magic('matplotlib', 'inline')
#To customize the properties and default styles of matplotlib
from pylab import rcParams
#To set default size of plots
rcParams['figure.figsize'] = (6, 6)


# In[7]:


#To provide functions for interacting with operating system
import os
import os.path
import itertools


# In[8]:


import glob


# In[9]:


import mahotas as mt


# In[10]:


def diamater(contour):
    dist_val = []
    for pair in combinations(contour,2):
        dist = np.linalg.norm(pair[1]-pair[0])
        dist_val.append(dist)
    max_dist = np.max(dist_val)
    return(max_dist)


# In[11]:


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


# In[12]:


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


# In[13]:


def eccentricity(contour):
    if(len(contour)>=5):
        (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
        a = ma/2
        b = MA/2
        ecc = np.sqrt(a**2 - b**2)/a
    else:
        ecc = 0 
    return ecc


# In[14]:


# To transform cartesian to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]


# In[13]:


###########Since there are simple and compound leaf, I selected simple type only for the analysis


# In[15]:


pwd


# In[15]:


#To add the paths of all images
img_dir_pathList=[]
for filename in glob.glob('Leaves\\Straight\\*.jpg'):
    img_path = filename
    if(os.path.exists(img_path)):
        img_dir_pathList.append(img_path)
    #im = Image.open(filename)
    #img_dir_pathList.append(im)


# In[16]:


img_dir_pathList


# In[17]:


len(img_dir_pathList)


# In[18]:


num1 = list(range(1,len(img_dir_pathList)+1))


# In[19]:


len(img_dir_pathList)


# In[20]:


num1


# In[21]:


#Straight images
def data_set():
    names = ['id','diameter','area','perimeter','physiological_length','physiological_width','aspect_ratio','rectangularity','circularity','compactness','NF','Perimeter_ratio_diameter','Perimeter_ratio_length','Perimeter_ratio_lw','No_of_Convex_points','perimeter_convexity','area_convexity','area_ratio_convexity','equivalent_diameter',             'cx','cy','eccentriciry','contrast','correlation_texture','inverse_difference_moments','entropy','Mean_R_val','Mean_G_val','Mean_B_val','Std_R_val','Std_G_val','Std_B_val']
    df = pd.DataFrame([], columns=names)
    for i in range(len(img_dir_pathList)):
        path = img_dir_pathList[i]
        #To read images in opencv
        Acutal_img1 = cv2.imread(path)
        cv2.imwrite('Images\\'+str(num1[i])+'.jpg',Acutal_img1)
        #Convert to RGB image
        img1 = cv2.cvtColor(Acutal_img1, cv2.COLOR_BGR2RGB)
        
        #To convert gray image in opencv
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #########Remove noise
        blur_img1 = cv2.GaussianBlur(gray_img1, (55, 55), 0)
        ##########Binary image
        #img1_binary = cv2.threshold(blur_img1,thresh, 255, cv2.THRESH_BINARY_INV)[1]
        ret1,img1_binary = cv2.threshold(blur_img1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        #Closing of holes using morphological transformation
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(img1_binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('Binary_Images\\'+str(num1[i])+'.jpg',closing)
        
        #Resize the image
        height = 1200
        width = 1600
        dim = (width, height)
        resize_img1 = cv2.resize(closing, dim)
        #by using image processing module of scipy to find the center of the leaf\n",
        
        cy, cx = ndi.center_of_mass(resize_img1)
        #Shape features
        contours, image= cv2.findContours(resize_img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        index = find_contour(resize_img1,contours)
        cnt = contours[index]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if(area == 0.0):
            cnt = contours[1]
        
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
        
        #Texture features
        textures = mt.features.haralick(gray_img1)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation_texture = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]
        
        #Color features
        red_channel = Acutal_img1[:,:,0]
        green_channel = Acutal_img1[:,:,1]
        blue_channel = Acutal_img1[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0
        
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)
        
        vector = [num1[i],diameter,area,perimeter,w,h,aspect_ratio,rectangularity,circularity,compactness,NF,Perimeter_ratio_diameter,Perimeter_ratio_length,Perimeter_ratio_lw,No_of_Convex_points,perimeter_convexity,area_convexity,area_ratio_convexity,equivalent_diameter,                  cx,cy,ecc,contrast,correlation_texture,inverse_diff_moments,entropy,red_mean,green_mean,blue_mean,red_std,green_std,blue_std]
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        
        contours1 = measure.find_contours(resize_img1, .8)
        # from which we choose the longest one
        contour = max(contours1, key=len)
        # To transform on all pairs in the set
        polar_contour = np.array([cart2pol(x, y) for x, y in contour])
        # if we substract a number from an array of numbers,
        # it assumes that we wanted to substract from all members
        contour[::,1] -= cx  # demean X
        contour[::,0] -= cy  # demean Y
        data_polar_contour = {'x':polar_contour[::,1], 'y':polar_contour[::,0]}
        df_polar_contour = pd.DataFrame(data_polar_contour)
        df_polar_contour
        #Read to csv
        df_polar_contour.to_csv('df_polar_contour\\df_polar_contour_img'+str(num1[i])+'.csv', index = False)
        # for local maxima
        c_max_index = argrelextrema(polar_contour[::,0], np.greater, order=50)
        data_polar_contour_maxima = {'x':polar_contour[::,1][c_max_index], 'y':polar_contour[::,0][c_max_index]}
        df_polar_contour_maxima = pd.DataFrame(data_polar_contour_maxima)
        df_polar_contour_maxima
        #Read to csv
        df_polar_contour_maxima.to_csv('df_polar_contour_maxima\\df_polar_contour_maxima_img'+str(num1[i])+'.csv', index = False)
        # for local maxima
        c_min_index = argrelextrema(polar_contour[::,0], np.less, order=50)
        data_polar_contour_minima = {'x':polar_contour[::,1][c_min_index], 'y':polar_contour[::,0][c_min_index]}
        df_polar_contour_minima = pd.DataFrame(data_polar_contour_minima)
        df_polar_contour_minima
        #Read to csv
        df_polar_contour_minima.to_csv('df_polar_contour_minima\\df_polar_contour_minima_img'+str(num1[i])+'.csv', index = False)
        data_contour = {'x':contour[::,1], 'y':contour[::,0]}
        df_contour = pd.DataFrame(data_contour)
        df_contour
        #Read to csv
        df_contour.to_csv('df_contour\\df_contour_img'+str(num1[i])+'.csv', index = False)
        #print(img_dir_pathList)
    return df


# In[22]:


data1 = data_set()


# In[23]:


data1.head(10)


# In[24]:


data1.to_csv("New_dataset_Images_with_all_features_set1.csv")


# In[15]:


#To add the paths of all images
img_dir_pathList = []
for filename in glob.glob('Leaves\\Oriented\\*.jpg'):
    img_path = filename
    if(os.path.exists(img_path)):
        img_dir_pathList.append(img_path)
    #im = Image.open(filename)
    #img_dir_pathList.append(im)


# In[16]:


len(img_dir_pathList)


# In[17]:


img_dir_pathList


# In[18]:


num2 = list(range(621,len(img_dir_pathList)+621))


# In[19]:


len(num2)


# In[20]:


len(img_dir_pathList)+621


# In[21]:


num2


# In[22]:


#Oriented images
def data_set1():
    names = ['id','diameter','area','perimeter','physiological_length','physiological_width','aspect_ratio','rectangularity','circularity','compactness','NF','Perimeter_ratio_diameter','Perimeter_ratio_length','Perimeter_ratio_lw','No_of_Convex_points','perimeter_convexity','area_convexity','area_ratio_convexity','equivalent_diameter',             'cx','cy','eccentriciry','contrast','correlation_texture','inverse_difference_moments','entropy','Mean_R_val','Mean_G_val','Mean_B_val','Std_R_val','Std_G_val','Std_B_val']
    df = pd.DataFrame([], columns=names)
    for i in range(len(img_dir_pathList)):
        path = img_dir_pathList[i]
        #To read images in opencv
        Acutal_img1 = cv2.imread(path)
        cv2.imwrite('Images\\'+str(num2[i])+'.jpg',Acutal_img1)
        #Convert to RGB image
        img1 = cv2.cvtColor(Acutal_img1, cv2.COLOR_BGR2RGB)
        
        #To convert gray image in opencv
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #########Remove noise
        blur_img1 = cv2.GaussianBlur(gray_img1, (55, 55), 0)
        ##########Binary image
        #img1_binary = cv2.threshold(blur_img1,thresh, 255, cv2.THRESH_BINARY_INV)[1]
        ret1,img1_binary = cv2.threshold(blur_img1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        #Closing of holes using morphological transformation
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(img1_binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('Binary_Images\\'+str(num2[i])+'.jpg',closing)
        
        #Resize the image
        height = 1200
        width = 1600
        dim = (width, height)
        resize_img1 = cv2.resize(closing, dim)
        #by using image processing module of scipy to find the center of the leaf\n",
        
        cy, cx = ndi.center_of_mass(resize_img1)
        #Shape features
        contours, image= cv2.findContours(resize_img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        index = find_contour(resize_img1,contours)
        cnt = contours[index]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if(area == 0.0):
            cnt = contours[1]
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        rect = cv2.minAreaRect(cnt)
        lw_val = Physilogical_width_and_length(rect)
        w = lw_val[0]
        h = lw_val[1]
        #x,y,w,h = cv2.boundingRect(cnt)
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
        
        #Texture features
        textures = mt.features.haralick(gray_img1)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation_texture = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]
        
        #Color features
        red_channel = Acutal_img1[:,:,0]
        green_channel = Acutal_img1[:,:,1]
        blue_channel = Acutal_img1[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0
        
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)
        
        vector = [num2[i],diameter,area,perimeter,w,h,aspect_ratio,rectangularity,circularity,compactness,NF,Perimeter_ratio_diameter,Perimeter_ratio_length,Perimeter_ratio_lw,No_of_Convex_points,perimeter_convexity,area_convexity,area_ratio_convexity,equivalent_diameter,                  cx,cy,ecc,contrast,correlation_texture,inverse_diff_moments,entropy,red_mean,green_mean,blue_mean,red_std,green_std,blue_std]
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        
        contours1 = measure.find_contours(resize_img1, .8)
        # from which we choose the longest one
        contour = max(contours1, key=len)
        # To transform on all pairs in the set
        polar_contour = np.array([cart2pol(x, y) for x, y in contour])
        # if we substract a number from an array of numbers,
        # it assumes that we wanted to substract from all members
        contour[::,1] -= cx  # demean X
        contour[::,0] -= cy  # demean Y
        data_polar_contour = {'x':polar_contour[::,1], 'y':polar_contour[::,0]}
        df_polar_contour = pd.DataFrame(data_polar_contour)
        df_polar_contour
        #Read to csv
        df_polar_contour.to_csv('df_polar_contour\\df_polar_contour_img'+str(num2[i])+'.csv', index = False)
        # for local maxima
        c_max_index = argrelextrema(polar_contour[::,0], np.greater, order=50)
        data_polar_contour_maxima = {'x':polar_contour[::,1][c_max_index], 'y':polar_contour[::,0][c_max_index]}
        df_polar_contour_maxima = pd.DataFrame(data_polar_contour_maxima)
        df_polar_contour_maxima
        #Read to csv
        df_polar_contour_maxima.to_csv('df_polar_contour_maxima\\df_polar_contour_maxima_img'+str(num2[i])+'.csv', index = False)
        # for local maxima
        c_min_index = argrelextrema(polar_contour[::,0], np.less, order=50)
        data_polar_contour_minima = {'x':polar_contour[::,1][c_min_index], 'y':polar_contour[::,0][c_min_index]}
        df_polar_contour_minima = pd.DataFrame(data_polar_contour_minima)
        df_polar_contour_minima
        #Read to csv
        df_polar_contour_minima.to_csv('df_polar_contour_minima\\df_polar_contour_minima_img'+str(num2[i])+'.csv', index = False)
        data_contour = {'x':contour[::,1], 'y':contour[::,0]}
        df_contour = pd.DataFrame(data_contour)
        df_contour
        #Read to csv
        df_contour.to_csv('df_contour\\df_contour_img'+str(num2[i])+'.csv', index = False)
        #print(img_dir_pathList)
    return df


# In[ ]:


data2 = data_set1()


# In[ ]:


data2.head(10)


# In[ ]:


data2.tail(10)


# In[ ]:


data2.to_csv("New_dataset_Images_with_all_features_set2.csv")


# In[28]:


data1 = pd.read_csv("New_dataset_Images_with_all_features_set1.csv")


# In[29]:


data1.head(10)


# In[31]:


del data1["Unnamed: 0"]


# In[33]:


data2.tail(10)


# In[35]:


data = pd.concat([data1,data2])


# In[36]:


data.head(10)


# In[37]:


data.tail(10)


# In[39]:


data.shape


# In[38]:


data.to_csv("New_dataset_Images_with_all_features_new.csv")

