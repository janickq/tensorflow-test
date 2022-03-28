import cv2
import numpy as np


IMAGE_PATHS = 'WOB\WOB IPAD\Photo 21-3-22, 2 59 14 PM.jpg'
def crop_minAreaRect(img, rect):

    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated

def getWOB(image):
    imagecopy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,41,11)
    
    cv2.imshow("thresh", thresh)
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    max_area = 0
    c = 0
    s = 0
    for i in cnts:
      area = cv2.contourArea(i)
      if area > 1000:
              if area > max_area:
                  max_area = area
                  best_cnt = i
                  image = cv2.drawContours(image, cnts, c, (0, 255, 0), 3)
                  s = s+1
      c+=1
    alist = best_cnt.reshape(best_cnt.shape[0], best_cnt.shape[2])
    xmax, ymax = np.max(alist, axis = 0)
    xmin, ymin = np.min(alist, axis = 0)
    rect = [[xmax, ymin], [xmin, ymin], [xmax, ymax], [xmin, ymax]]
    mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    result = perspective_transform(mask, imagecopy, rect)
    result = result[ymin:ymax, xmin:xmax]
    cv2.imshow("mask", result)
  
  
def perspective_transform(mask, img, rect):
    corners = find_corners(mask)
    print(corners)
    rows,cols = mask.shape
    print(mask.shape)
    # pts1 = np.float32([[corners[0,0],corners[0,1]],[corners[1,0],corners[1,1]],[corners[2,0],corners[2,1]],[corners[3,0],corners[3,1]]])
    pts1 = np.float32([corners[1], corners[2], corners[3], corners[4]])
    # pts2 = np.float32([[rows,0], [0, 0], [0,cols], [rows,cols]])
    pts2 = np.float32(rect)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    print(M)
    dst = cv2.warpPerspective(img,M,(1000,1000))
    return dst

def find_corners(img):
    
    dst = cv2.cornerHarris(img,20,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
    x = 0
    for i in range(1, len(corners)):
        print(corners[i])
        cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (255,255,255), 2)
        cv2.putText(img, str(x), (int(corners[i,0]), int(corners[i,1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (125,125,125), 2)
        x = x+1
    
    cv2.imshow('image', img)
    
    return corners

image = cv2.imread(IMAGE_PATHS)
image = cv2.resize(image, (1000,1000))
getWOB(image)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
