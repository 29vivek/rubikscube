import cv2
import numpy as np
from multiprocessing import Pool, Array
import time
import os

# import multiprocessing
# We must import this explicitly, it is not imported by the top-level multiprocessing module.
# import multiprocessing.pool

arr = Array('i', 54, lock=False)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

lower_white = ((0,0,100),)
upper_white = ((180,150,255),)

lower_green = ((60, 150, 100),)   
upper_green = ((80, 255, 255),) 

lower_orange = ((5, 120, 150),)
upper_orange = ((12, 255, 255),)

lower_red = ((0, 100, 80), (175, 100, 80))    
upper_red = ((5, 255, 200), (180, 255, 200))

lower_blue = ((80,150,100),)
upper_blue = ((120,255,255),)

lower_yellow = ((20, 150, 170),)
upper_yellow = ((50, 255, 255),)

def show(im, x):
    cv2.imshow('im{}'.format(x), im)
    cv2.waitKey(0) # milliseconds
    cv2.destroyAllWindows()

def facelet_number(x_cen,y_cen):
    if x_cen>=0 and x_cen<=100:
        row=0
    elif x_cen>100 and x_cen<=200:
        row=1
    elif x_cen>200 and x_cen<=300:   
        row=2
    if y_cen>=0 and y_cen<=100:
        column=0
    elif y_cen>100 and y_cen<=200:
        column=3
    elif y_cen>200 and y_cen<=300:   
        column=6
    return row+column   

def get_color_number(thresh):
    if thresh==lower_orange:
        return 0
    if thresh==lower_blue:
        return 1
    if thresh==lower_red:
        return 2
    if thresh==lower_green:
        return 3
    if thresh==lower_yellow:
        return 4
    if thresh==lower_white:
        return 5

def find(obj):
    dim = (300,300)

    obj[0] = cv2.resize(obj[0], dim)
    obj[0] = cv2.bilateralFilter(obj[0], 9, 75, 75)
    # print(obj[0].shape)
    
    # filter strength, for colors, templateWindowSize and searchWindowSize
    obj[0] = cv2.fastNlMeansDenoisingColored(obj[0], None, 10, 10, 7, 24)
    # show(obj[0], 3)
    hsvImage = cv2.cvtColor(obj[0], cv2.COLOR_BGR2HSV)
    # show(hsvImage, 4)

    frame_threshed = np.zeros(shape=(300, 300), dtype = np.uint8)
        
    for low, high in zip(obj[2], obj[3]):
        mask = cv2.inRange(hsvImage, np.array(low), np.array(high))
        frame_threshed = cv2.bitwise_or(frame_threshed, mask)
    # show(frame_threshed, obj[1])   
    
    ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
    # show(thresh, 6)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    # print(areas) 
    
    validContours = []
    maxArea = 0
    for area in areas:
        if area > maxArea:
            maxArea = area
        
    for i in range(len(areas)):
        if((-3000 <= areas[i]-maxArea <= 3000) and areas[i] >= 4000):
            validContours.append(contours[i])
    
    for i in validContours:
        x,y,w,h = cv2.boundingRect(i)
        obj[0]=cv2.rectangle(obj[0],(x,y),(x+w,y+h),(0,255,0),3)
        x_cen=x+(w/2)
        y_cen=y+(h/2)
        num=facelet_number(x_cen,y_cen)
        arr[obj[1]*9+num]=get_color_number(obj[2])
    
    # show(obj[0],7)
    
def run_in_parallel(im, face):
    ranges = [[im, face, lower_white, upper_white], 
              [im, face, lower_blue, upper_blue],
              [im, face, lower_orange, upper_orange],
              [im, face, lower_green, upper_green],
              [im, face, lower_red, upper_red],
              [im, face, lower_yellow, upper_yellow]]
    pool = Pool(processes = len(ranges))
    pool.map(find, ranges)

def get_char(temp):
    if temp==0:
        return 'L'
    if temp==1:
        return 'F'
    if temp==2:
        return 'R'
    if temp==3:
        return 'B'
    if temp==4:
        return 'U'
    if temp==5:
        return 'D'

if __name__ == '__main__':
    t1=time.time()
    
    file_names=['left.png','front.png','right.png','back.png','up.png','down.png']
    
    for i in range(len(file_names)):
        im=cv2.imread(file_names[i])
        run_in_parallel(im, i)
    
    t2=time.time()
    
    print('execution time: ', t2-t1)
    # for x in arr:
    #     print(x, end=' ')
    
    edges_num = [[43,10],[41,19],[37,28],[39,1],
                [46,16],[50,25],[52,34],[48,7],
                [14,21],[12,5],[30,23],[32,3]]
    corners_num = [[44,11,18],[38,20,27],[36,29,0],[42,2,9],
                    [47,24,17],[45,15,8],[51,6,35],[53,33,26]]
    
    edges_char=[]
    corner_char=[]
    string=''
    for i in edges_num:
        string = ''
        for j in i:
            string=string+get_char(arr[j])
        edges_char.append(string) 
    
    for i in corners_num:
        string=''
        for j in i:
            string=string+get_char(arr[j])
        corner_char.append(string)

    cubeString = ' '.join(edges_char + corner_char)
    print('input: ', cubeString)
    solution = os.popen('./a.out ' + cubeString).read()
    print('solution: ', solution)
