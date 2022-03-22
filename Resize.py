import cv2
import glob

for filename in glob.glob('WOB\WOB3\*.jpg'): # path to your images folder
    print(filename)
    img=cv2.imread(filename) 
    rl=cv2.resize(img, (1024,1024))
    cv2.imwrite(f'{filename}resized.jpg', rl)