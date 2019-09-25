import cv2
import numpy as np

def show(im, suffix):
    cv2.imshow(f'im-{suffix}', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processColor(im, hsv, lower, upper, color):

    frameThreshed = np.zeros(shape=[300, 300], dtype = np.uint8)
    
    for low, high in zip(lower, upper):
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        frameThreshed = cv2.bitwise_or(frameThreshed, mask)

    ret, thresh = cv2.threshold(frameThreshed, 127, 255, 0)
    show(thresh, 'threshed')

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    print(areas)

    validContours = []

    maxArea = 0
    for area in areas:
        if area > maxArea:
            maxArea = area
    
    for i in range(len(areas)):
        if((-3000 <= areas[i]-maxArea <= 3000) and areas[i] >= 4000):
            validContours.append(contours[i])

    for i in validContours:
        x, y, w, h = cv2.boundingRect(i)
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

    show(im,f'det-{color}')

def processImage(imageName, face):

    # uses BGR by default
    im = cv2.imread(imageName)
    dimensions = (300,300)
    im = cv2.resize(im, dimensions)
    width, height, numberOfChannels = im.shape

    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html d=9 for heavy noise filtering.
    im = cv2.bilateralFilter(im, 9, 75, 75)

    # filter strength, for colors, templateWindowSize and searchWindowSize
    im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 24)
    
    hsvImage = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    show(hsvImage, 'hsv')

    # use gimp to retrieve hsv values. gimp uses (0-360) for hue, 0-100 for saturation and value.
    # opencv uses 0-180 for hue, 0-255 for the rest two.


    lower_white = [[0,0,170]]
    upper_white = [[180,110,255]]
        
    lower_green = [[60, 60, 60]]	
    upper_green = [[70, 255, 255]]	

    lower_orange = [[5, 170, 210]]
    upper_orange = [[10, 225, 255]]

    lower_red = [[0, 170, 130], [175, 170, 130]]	
    upper_red = [[5, 240, 180], [180, 240, 180]]

    lower_blue = [[80,180,190]]
    upper_blue = [[120,255,255]]
	
    lower_yellow = [[35, 220, 220]]
    upper_yellow = [[45, 255, 255]]

    processColor(im, hsvImage, lower_yellow, upper_yellow, 'yellow')
    processColor(im, hsvImage, lower_white, upper_white, 'white')
    processColor(im, hsvImage, lower_green, upper_green, 'green')
    processColor(im, hsvImage, lower_orange, upper_orange, 'orange')
    processColor(im, hsvImage, lower_red, upper_red, 'red')
    processColor(im, hsvImage, lower_blue, upper_blue, 'blue')

class 



def main():
    processImage('back.jpg', 'back')


if __name__ == "__main__":
    main()
