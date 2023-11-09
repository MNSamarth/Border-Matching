#source code
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def compareimg(path1,path2):
    image= cv2.imread(path2)
    ch,cw,_=image.shape
    ratio=ch/cw
    print(ratio)

    img2= cv2.imread(path1)
    template=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    height,width=template.shape
    print(height,width)

    height=int((ratio)*width)
    image = cv2.resize(image, (width,height))
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(height,width)

    result= cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val,max_val,min_loc, max_loc= cv2.minMaxLoc(result)
    min=min_loc[1]
    max=max_loc[1]
    height, width= template.shape[:2]

    top_left= max_loc
    bottom_right= (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image, top_left, bottom_right, (0,0,255),5)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image_crop=image[max:max+height,0:width]

    title=["screenshot","image_extraction"]
    image=[image,image_crop]

    #for i in range(2):
       # plt.subplot(1,2,i+1)
        #plt.title(title[i])
       # plt.imshow(image[i])

    #plt.show()

    height,width,_=image_crop.shape
    print(height,width)
    img1=cv2.resize(img2,(width,height))
    img2=image_crop

    #converting to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (similar, diff) = compare_ssim(gray1, gray2, full=True) 
    diff = (diff*255).astype("uint8") 

    #to display the difference image

    #applying treshold value
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the differences on both images
    sum=0
    for contour in contours:
        if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 10000:
            sum=sum+cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.putText(img2, "Similarity: " + str(similar), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2) 

    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if(sum>80000):
        print(f"\n{path1} The capture and screenshot are differenet\n")

"""
    titles=["capture","higlighted_ss"]

    images=[img1,img2]
    for i in range(2):
       plt.subplot(1,3,i+1)
       plt.title(titles[i])
       plt.imshow(images[i])

    plt.subplot(1,3,3)
    plt.title("Difference image")
    plt.imshow(diff,cmap='gray')

    #plt.show()
"""


dir_path="Sample_dataset"

count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('Image count:', count/2)
count=count/2

for i in range(1,int(count+1)):
    p1 = dir_path+"\\cap"+str(i)+".jpg" #cam
    #p1=p1.replace(" ","")
    p2 = dir_path+"\\ss"+str(i)+".jpg" #ss
    #p2=p2.replace(" ","")
    print(p1+p2)
    compareimg(p1,p2)