import cv2 as cv
from numpy.f2py.auxfuncs import istrue

import numpy as np
from numpy.ma.core import empty

################################################Images#########################################################
'''
img= cv.imread(r'../Image_Vedio/Orange.jpeg', flags= 1)
     #add 'r' at the beginning of path to read it correctly
     #imread() flags :
     #0 ----> loads image in grey scale(Black & White)
     #1 ----> loads image with colors

img_Resize = cv.resize(img ,(300 ,300) , interpolation=cv.INTER_CUBIC) # change size of Image    cv.resize( image , size(x,y) , "Constant"interpolation=cv.INTER_CUBIC)

cv.imshow('Orange1',img) #name of window : "Orange"
cv.imshow('Orange2', img_Resize)

print(img.shape) # to Know size of Image
print(img_Resize.shape)

cv.waitKey(0)  #waits 't' time until a button is pressed, when 't' = 300 ms then it will loop for that time until user
    #press a key to break the loop, Unique case : if 't' = 0 then it will loop forever until a keypress

cv.destroyAllWindows() #closes all windows opened

'''
'''
################################################Empty_Image#########################################################
empty_img = np.zeros((500,400,3) ,dtype="uint8")

cv.imshow("empty1" , empty_img)

empty_img[:] = 0 , 0, 255
cv.imshow("empty2" , empty_img)

empty_img[100:200 , 100:200] = 0,0,0
cv.imshow("Shape" , empty_img)

cv.rectangle(empty_img , (0,0) , (300,300) ,(0,255,255) ,thickness=-3) #Draw rectangle
# cv.rectangle(image, start , End , color , thickness=3) thickness->if I need solid thickness= -value
cv.imshow("rectangle" , empty_img)

cv.line(empty_img , (0,0) , (300,300) ,(0,0,0) ,thickness=4) #Draw Line
cv.imshow("line" , empty_img)

cv.circle(empty_img,(100,100),100, (200,50,0) ,thickness=-2) #Draw circle
cv.imshow("circle" , empty_img)

cv.putText(empty_img ,"Elsayed Hossny" ,(100,200) ,cv.FONT_ITALIC ,1.0 ,(255,0,255) ,thickness=5)
cv.imshow("putText" , empty_img)

cv.waitKey(0)

cv.destroyAllWindows()
'''

################################################Vedio#########################################################
'''
###############################################video##############################################
capture = cv.VideoCapture('../Image_Vedio/Vedio.mp4')  # video in var
while True:
    isTrue , Frame = capture.read() # return 2 var... 1->if true success read video / 2-> return Frames of Video
# Convert video to images
    if isTrue :
        cv.imshow('Video' , Frame)
        if cv.waitKey(20) & 0xFF == ord('x') :
            break
    else:
        break

capture.release() # more protect to Know all end Video
cv.destroyAllWindows()

'''
''' 
###############################################video_Live#########################################
video_Live = cv.VideoCapture(0)

while True :
    isTrue, Frame = video_Live.read()
    img_Resize = cv.resize(Frame, (1000, 300), interpolation=cv.INTER_CUBIC) #chane size of video
    if isTrue:
        cv.imshow('Video_live' , img_Resize)
        if cv.waitKey(20) & 0xFF==ord('x') :
            break
    else:
        break

video_Live.release() # more protect to Know all end Video
cv.destroyAllWindows()
'''

'''
#------------------------------------------------  Working with arrays  ------------------------------------------------
import array as arr
import  numpy as np

a = [1,2,3]
z = arr.array('b',a) #converts an object to array of specific datatype, 'b': convert to signed char
print(z) #-->array('b', [1, 2, 3])
z = arr.array('B',a) #converts an object to array of specific datatype, 'B': convert to unsigned char
print(z) #-->array('B', [1, 2, 3])
z = arr.array('i',a) #converts an object to array of specific datatype, 'i': convert to signed int (2 bytes)
print(z) #-->array('i', [1, 2, 3])
z = arr.array('f',a) #converts an object to array of specific datatype, 'f': convert to float
print(z) #-->array('f', [1.0, 2.0, 3.0])

z = z * 2 #this duplicates the array
print(z) #-->array('f', [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

z = z * 3 #this triples the array
print(z) #-->array('f', [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


#What if we want to make calculations on array elements? ,, use numpy functions
z = np.array(a)
print(z) #-->[1 2 3]
z = z * 2
print(z) #-->[2 4 6]
#-----------------------------------------------------------------------------------------------------------------------
'''

'''
#----------------------------------------------  Useful methods in numpy  ----------------------------------------------

#to show number of rows & cols. in a matrix
user_matrix = np.array([[1,2,3],[4,5,6],[7,8,9],[0,0,0]])
# user_matrix : # 1 2 3
                # 4 5 6
                # 7 8 9
                # 0 0 0

#print(user_matrix)

rows,cols = np.shape(user_matrix)
print(rows) #-->4
print(cols) #-->3

col_sum = np.sum(user_matrix,0) #calculates sum of each column , 0 : means we select columns to get their sum
print(col_sum) #-->[12 15 18]

row_sum = np.sum(user_matrix,1) #calculates sum of each row , 1 : means we select rows to get their sum
print(row_sum) #-->[ 6 15 24  0]

max_value = np.max(user_matrix) #returns maximum value in the whole matrix
print(max_value) #-->9

min_value = np.min(user_matrix) #returns minimum value in the whole matrix
print(min_value) #-->0

user_matrix_T= user_matrix.T #transposed of matrix a
print(user_matrix_T) # 1 4 7 0
                     # 2 5 8 0
                     # 3 6 9 0
#-----------------------------------------------------------------------------------------------------------------------
'''

'''
#------------------------------------------------------  OpenCV  -------------------------------------------------------
import cv2
#to read image :
orange_image = cv2.imread(r"orange.png", flags =1) #add 'r' at the beginning of path to read it correctly
#imread() flags :
# 0 ----> loads image in grey scale(Black & White)
# 1 ----> loads image with colors

#to show image :
cv2.imshow("Fruit",orange_image) #name of window : "Fruit"

cv2.waitKey(0) #waits 't' time until a button is pressed, when 't' = 300 ms then it will loop for that time until user
               #press a key to break the loop, Unique case : if 't' = 0 then it will loop forever until a keypress

cv2.destroyAllWindows()  #closes all windows opened

#to control keypress: it will not break until hitting the 'esc' or 'q' buttons
key_pressed = ''
while key_pressed != 27 and key_pressed != ord('q'):
    print("invalid Key")
    key_pressed = cv2.waitKey(0)
else:
    cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------  Pillow  -------------------------------------------------------
from PIL import Image

# Open an image
orange = Image.open("orange.png")

# Show the image
orange.show()

# Convert to grayscale
grey_orange = orange.convert("L")

#save new orange B&W photo
grey_orange.save(r"grey_orange.png")
#-----------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------  skimage  -------------------------------------------------------
import matplotlib.pyplot as plt
from skimage import io

# Load an image
orange_image = io.imread("orange.png")

#plotting the image
plt.imshow(orange_image)

#showing the plot
plt.show()
#-----------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------  Sheet  -------------------------------------------------------
#1. Create an array by two ways (array, NumPy) and multiply it by 2 and write the difference between the outputs.
import array as arr
import  numpy as np

my_list = [7,7,7]

first_array = np.array(my_list)
second_array = arr.array('i',my_list)

print(first_array * 2)
print(second_array * 2)
#first array is a numpy array when multiplying it by a number
#it multiplies the number by each value inside the array
#while in second array it repeats the array

#-----------------------------------------------------------------------------------------------------------------------
#2. Create matrix of ones and matrix of zeros in any size.
import  numpy as np

ones_matrix = np.ones((4,2),dtype = int)
print(ones_matrix) #--> 1 1
                   #    1 1
                   #    1 1
                   #    1 1

zeros_matrix = np.zeros((2,4),dtype = int)
print(zeros_matrix) #-->0 0 0 0
                    #   0 0 0 0
#-----------------------------------------------------------------------------------------------------------------------
#3. Read an image and show it using OpenCV library.
import cv2

orange_image = cv2.imread(r"orange.png", flags = 1)
cv2.imshow("Fruit", orange_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#4. Read an image and show it using pillow library.
from PIL import Image

orange = Image.open(r"orange.png")
orange.show()
#-----------------------------------------------------------------------------------------------------------------------
#5. Read an image and show it using scikit-image library.
import matplotlib.pyplot as plt
from skimage import io

orange_image = io.imread(r"orange.png")
plt.imshow(orange_image)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#6. Show image using matplotlib library.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #to read images

image = mpimg.imread(r"orange.png")
plt.imshow(image) #plotting image
plt.show()  #showing image


'''