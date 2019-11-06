import cv2
import numpy as np
from multiprocessing import Pool
import time
import os
import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


arr = multiprocessing.Array('i', 54, lock=False)
file_names=['left.png','front.png','right.png','back.png','up.png','down.png']
dim = (300,300)

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
    if x_cen>=0 and x_cen<=dim[0]/3:
        row=0
    elif x_cen>dim[0]/3 and x_cen<=2*dim[0]/3:
        row=1
    elif x_cen>2*dim[0]/3 and x_cen<=dim[0]:   
        row=2
    if y_cen>=0 and y_cen<=dim[1]/3:
        column=0
    elif y_cen>dim[1]/3 and y_cen<=2*dim[1]/3:
        column=3
    elif y_cen>2*dim[1]/3 and y_cen<=dim[1]:   
        column=6
    return row+column   

def get_color_number(thresh):
    if thresh==lower_red:
        return 0
    if thresh==lower_blue:
        return 1
    if thresh==lower_orange:
        return 2
    if thresh==lower_green:
        return 3
    if thresh==lower_yellow:
        return 4
    if thresh==lower_white:
        return 5

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

def find(obj):
    
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
    
    # show(obj[0], 8)
    return obj[0]
    
def run_in_parallel(im, face):
    ranges = [[im, face, lower_white, upper_white], 
              [im, face, lower_blue, upper_blue],
              [im, face, lower_orange, upper_orange],
              [im, face, lower_green, upper_green],
              [im, face, lower_red, upper_red],
              [im, face, lower_yellow, upper_yellow]]
    pool = Pool(processes = len(ranges))
    results = pool.map(find, ranges)
    pool.close()
    pool.join()

    color_order = ['white', 'blue', 'orange', 'green', 'red', 'yellow']
    for i, result in enumerate(results):
        cv2.imwrite(file_names[face].split('.')[0]+'_{}.png'.format(color_order[i]), result)

def run_just_linear(im, face):
    ranges = [[im, face, lower_white, upper_white], 
              [im, face, lower_blue, upper_blue],
              [im, face, lower_orange, upper_orange],
              [im, face, lower_green, upper_green],
              [im, face, lower_red, upper_red],
              [im, face, lower_yellow, upper_yellow]]
    
    results = []
    for color in ranges:
        results.append(find(color))

    color_order = ['white', 'blue', 'orange', 'green', 'red', 'yellow']
    for i, result in enumerate(results):
        cv2.imwrite(file_names[face].split('.')[0]+'_{}.png'.format(color_order[i]), result)

def run_face_parallel(obj):
    # only obj is defined, which is say left.png
    im = cv2.imread(obj)
    im = cv2.resize(im, dim)
    run_in_parallel(im, file_names.index(obj))

def run_face_linear(obj):
    im = cv2.imread(obj)
    im = cv2.resize(im, dim)
    run_just_linear(im, file_names.index(obj))


if __name__ == '__main__':
    
    t1=time.time()

    pool = MyPool(processes=len(file_names))
    pool.map(run_face_parallel, file_names)
    pool.close()
    pool.join()

    t2=time.time()

    print('execution time for parallel face, parallel color: ', t2-t1)

    t1=time.time()

    pool = MyPool(processes=len(file_names))
    pool.map(run_face_linear, file_names)
    pool.close()
    pool.join()

    t2=time.time()

    print('execution time for parallel face, linear color: ', t2-t1)


    t1=time.time()
    
    for i in range(len(file_names)):
        im=cv2.imread(file_names[i])
        im = cv2.resize(im, dim)
        run_in_parallel(im, i)
    
    t2=time.time()
    
    print('execution time for linear face, parallel color: ', t2-t1)
    
    t1=time.time()
    
    for i in range(len(file_names)):
        im=cv2.imread(file_names[i])
        im = cv2.resize(im, dim)
        run_just_linear(im, i)
    
    t2=time.time()

    print('execution time for linear face, linear color: ', t2-t1)

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
    os.popen('gcc ./solver/ct5.C -w')
    solution = os.popen('./a.out ' + cubeString).read()
    print('solution: ', solution)
