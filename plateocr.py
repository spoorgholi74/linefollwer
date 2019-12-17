import cv2
import glob
#import pytesseract
import os
import numpy as np
import shutil


def contrast(image, a, b):
    alpha = 2 # Simple contrast control[1-3]
    beta = 0    # Simple brightness control [0-100]
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imshow('Contrasted', image)
    #cv2.waitKey(0)
    return image


def adaptiveThreshold(image, method):
    new_image = contrast(image, a = 1, b = 0)

    gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    #rotated_image = rotate(image = gray)
    #gray = rotated_image
    #cv2.imshow('Gray', gray)
    
    if method == 'binary':
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        #cv2.imshow('binary thresh', thresh)
    elif method == 'otsu':
        blur = cv2.GaussianBlur(gray,(5,5),0)
        #blur = cv2.bilateralFilter(gray, 11, 17, 17)
        high_thresh,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('OTSU thresh', thresh)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        #cv2.imshow('adaptive guassian thresh', thresh)
    else:
        raise Exception("The method is not valid!")

    #cv2.imwrite("processed//plate{}.jpg".format(i), thresh)
    return thresh


def addBorder(image, size):  
    border = cv2.copyMakeBorder(image, top=size, bottom=size, left=size, right=size,
                                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])       
        #cv2.imwrite("borders/plate{}.jpg".format(i), border)
    return border


def preprocess(image, morph=2):
    edged = cv2.Canny(image, 170, 200, apertureSize=3)
    #cv2.imshow("Canny Edges", edged)

    lines = cv2.HoughLinesP(image=edged, rho=1, theta=np.pi / 180, threshold=60
        , lines=np.array([]), minLineLength=100, maxLineGap=80)
    
    if lines is None:
        #print("No lines!!!")
        pass
    else:
        a, b,  c = lines.shape
        for i in range(a):
            x = lines[i][0][0] - lines[i][0][2]
            y = lines[i][0][1] - lines[i][0][3]
            if x != 0:
                if abs(y / x) < 1:
                    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255),
                             1, cv2.LINE_AA)

    #cv2.imshow('lines', image)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    morph_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    #cv2.imshow('Final_image after morph', morph_img)

    return morph_img


def get_text(image):
    config = '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --oem 1 --psm 7'

    #text = pytesseract.image_to_string(image, config = config)
    dframe = pytesseract.image_to_data(image, config = config, output_type='data.frame')
    #print(dframe)
    words = list(dframe.text.values)
    text = 'SS'
    for i in words:
        if 5 < len(str(i)):
            text = i

    cleanText = []
    #print(text)
    for char in str(text):
        if type(char) == float:
            continue
        if char.isalnum():
            char = char.upper()
            cleanText.append(char)

    plate = ''.join(cleanText)
    return plate

def cleanOCR(image):
    edges = cv2.Canny(image, 170, 200, apertureSize=3)    
    #cv2.imshow('edges', edges)
    
    gray = preprocess(edges, 1)
    '''
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=60
        , lines=np.array([]), minLineLength=100, maxLineGap=80)
    
    if lines is None:
        #print("No lines!!!")
        pass
    else:
        a, b,  c = lines.shape
        for i in range(a):
            x = lines[i][0][0] - lines[i][0][2]
            y = lines[i][0][1] - lines[i][0][3]
            if x != 0:
                if abs(y / x) < 1:
                    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255),
                             1, cv2.LINE_AA)

    #cv2.imshow('lines', image)
    

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    #cv2.imshow('Morphology', gray)
    '''

    txt_result = get_text(gray)
    return txt_result

def resize(image, w_size):
    ratio = float(w_size) / image.shape[1]
    dim = (w_size, int(image.shape[0] * ratio))
    resizedCubic = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resizedCubic

def Method1(image):
    image = cv2.imread(image)
    resizedCubic = resize(image, 200)
    #cv2.imshow('Normal', resizedCubic)
    thresh = adaptiveThreshold(image = resizedCubic, method = 'adaptive')
    bordered = addBorder(image = thresh, size = 10)
    final1 = cleanOCR(bordered)
    #cv2.waitKey(0)
    #cv2.imwrite("resized/plate{}.jpg".format(i), resizedCubic)
    return final1

######################################################################################################
#####################################################################################################
def is_contour_bad(c): 
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # the contour is 'bad' if it is not a rectangle
    return len(approx) == 4

def get_mask(original, image):
    cnts, hiers = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[1:30] 
    #print('contours: ', cnts)

    '''
    filtered_cnt = []
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)       #I have used min Area rect for better result
        width = rect[1][0]
        height = rect[1][1]
        if (width<30) and (height <40) and (width >= 5) and (height > 10):
            #put your code here if desired contour is detected
            filtered_cnt.append(cnt)
            #print(cv2.contourArea(cnt))
    '''   

    # Masking the part other than the number plate
    mask = np.zeros(image.shape, dtype = 'uint8')

    '''
    for c in cnts:
        # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 255, 1)

    '''

    new_image = cv2.drawContours(mask,cnts,-1,(255),-1)
    #cv2.imshow("Mask",mask)
    #new_image = cv2.bitwise_and(image, mask)
    invert = cv2.bitwise_not(mask)
    #cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    #cv2.imshow("Final_image",new_image)
    #cv2.imshow("Inverted",invert)

    return invert

def rotate(image):
    thresh = cv2.threshold(image, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #deskew
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
     
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     
    # show the output image
    #print("[INFO] angle: {:.3f}".format(angle))
    ##cv2.imshow("Input", image)
    #cv2.imshow("Rotated", rotated)


def findcontours(image):
    image = cv2.imread(image)
    image = resize(image, 200)
    #cv2.imshow('Normal', image)

    threshGauss_1 = adaptiveThreshold(image = image, method = 'otsu')
    cv2.imshow('x', threshGauss_1)
    new_image_1 = get_mask(original = image, image = threshGauss_1)
    cv2.imshow('xx', new_image_1)
    new_image_1 =  addBorder(new_image_1, 5)
    cv2.imshow('xxx', new_image_1)
    processed_img_1 = preprocess(image = new_image_1, morph=4)
    cv2.imshow('2', processed_img_1)

    threshGauss_2 = adaptiveThreshold(image = image, method = 'adaptive')
    cv2.imshow('y', threshGauss_2)
    new_image_2 = get_mask(original = image, image = threshGauss_2)
    new_image_2 =  addBorder(new_image_2, 5)
    processed_img_2 = preprocess(image = new_image_2, morph=4)
    cv2.imshow('2', processed_img_2)
    cv2.waitKey(0)
    '''
    # Run tesseract OCR on image
    b = get_text(processed_img_1)
    c = get_text(threshGauss_1)
    d = get_text(processed_img_2)

    response = []

    if len(b) == 6:
        response.append(b)
    if len(c) == 6:
        response.append(c)
    if len(d) == 6:
       response.append(d)
    
    '''
    '''
    config = '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -l eng --oem 1 --psm 7'

    dframe = pytesseract.image_to_data(threshGauss, config = config, output_type='data.frame')
    #print(dframe)
    words = list(dframe.text.values)
    for i in words:
        if 5 < len(str(i)):
            text = i
        else:
            text = pytesseract.image_to_string(threshGauss, config = config)

    cleanText = []
    for char in text:
        if char.isalnum():
            char = char.upper()
            cleanText.append(char)

    plate = ''.join(cleanText)
    '''
    #cv2.waitKey(0)
    return 

def main():
    method = 2
    for plate in glob.glob("Data/*.jpg"):
        #a = Method1(image = plate)
        l = findcontours(image = plate)

if __name__ == '__main__':
    main()
