import cv2
import numpy as np

class Rubiks:
    def __init__(self, images):
        self.images = images
    
    lower_white = [[0,0,100]]
    upper_white = [[180,150,255]]
        
    lower_green = [[60, 60, 60]]	
    upper_green = [[70, 255, 255]]	

    lower_orange = [[5, 170, 210]]
    upper_orange = [[10, 225, 255]]

    lower_red = [[0, 170, 130], [175, 170, 130]]	
    upper_red = [[5, 240, 180], [180, 240, 180]]

    lower_blue = [[80,180,190]]
    upper_blue = [[120,255,255]]
	
    lower_yellow = [[30, 200, 200]]
    upper_yellow = [[50, 255, 255]]

    dimensions = (300, 300)

    @classmethod
    def show(cls, im):
        cv2.imshow('', im)
        cv2.waitKey(500) # 500 milliseconds
        cv2.destroyAllWindows()

    @classmethod
    def processColor(cls, im, hsv, lower, upper):

        frameThreshed = np.zeros(shape=cls.dimensions, dtype = np.uint8)
        
        for low, high in zip(lower, upper):
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            frameThreshed = cv2.bitwise_or(frameThreshed, mask)

        ret, thresh = cv2.threshold(frameThreshed, 127, 255, 0)
        cls.show(thresh)

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

        cls.show(im)


    def processCube(self):
        for image in self.images:
            # uses BGR by default
            im = cv2.imread(image)
            im = cv2.resize(im, Rubiks.dimensions)
            width, height, numberOfChannels = im.shape

            # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html d=9 for heavy noise filtering.
            im = cv2.bilateralFilter(im, 9, 75, 75)

            # filter strength, for colors, templateWindowSize and searchWindowSize
            im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 24)
            
            hsvImage = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            Rubiks.show(hsvImage)
            # use gimp to retrieve hsv values. gimp uses (0-360) for hue, 0-100 for saturation and value.
            # opencv uses 0-180 for hue, 0-255 for the rest two.
            
            Rubiks.processColor(im, hsvImage, Rubiks.lower_yellow, Rubiks.upper_yellow)
            Rubiks.processColor(im, hsvImage, Rubiks.lower_white, Rubiks.upper_white)
            Rubiks.processColor(im, hsvImage, Rubiks.lower_green, Rubiks.upper_green)
            Rubiks.processColor(im, hsvImage, Rubiks.lower_orange, Rubiks.upper_orange)
            Rubiks.processColor(im, hsvImage, Rubiks.lower_red, Rubiks.upper_red)
            Rubiks.processColor(im, hsvImage, Rubiks.lower_blue, Rubiks.upper_blue)


def main():
    rubiks = Rubiks(['back.jpg', 'up.jpg', 'right.jpg', 'left.jpg', 'front.jpg', 'down.jpg'])
    rubiks.processCube()

if __name__ == "__main__":
    main()
