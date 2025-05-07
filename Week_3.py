from PIL import Image
img = Image.open("orange.png") #to load image into object 'img'

img.show() #to show image

img_pixels = img.load() #to load pixels values of img image

img_pixels[30,50] #access pixels values at coordinates (x = 30, y = 50)

print(img_pixels[30,50]) # prints 4 values as image is in RGBA mode

#**************** Image Format *****************************
img.format #returns the  Kind of image (PNG,JPG)
print(img.format)

#**************** Image Size ********************************
img.size #returns width and height of image
print(img.size)   #-->(600, 346)
print(img.width)  #-->600
print(img.height) #-->346

#*************** Image Modes Color*********************************
img.mode #the mode where the image in (RGB,RGBA,L,1,CMYK,...)
print(img.mode) #-->RGBA

#*************** Image Crop **************************************
img.crop((0,0,200,200)) #(left, upper, right, lower)

cropped_img = img.crop((0,0,300,346))

#**************************************************** Image Resize *****************************************************
resized_img = img.resize((100,100)) #to give the image specific width and height
resized_img.show() #image shown here is not as original, because the ratio between width and height has changed
resized_img = img.resize((img.width//2, img.height//2)) #this keeps the ratio w/h constant
resized_img.show() #we used '//' not '/' in [previous line] so we can floor division result to integer value
resized_img.reduce(2) #this code makes the same resize with keeping the ratio w/h unchanged
#***********************************************************************************************************************

#***************************************************** Save Image ******************************************************
resized_img.save("resized_img.png") #taking the full path of the image and its format to save it
#***********************************************************************************************************************


#***************************************************** Transpose *******************************************************
trans_img = img.transpose(Image.FLIP_TOP_BOTTOM)
trans_img = img.transpose(Image.FLIP_LEFT_RIGHT)
trans_img = img.transpose(Image.ROTATE_90)
trans_img = img.transpose(Image.ROTATE_180)
trans_img = img.transpose(Image.ROTATE_270)
trans_img = img.transpose(Image.TRANSPOSE)
trans_img = img.transpose(Image.TRANSVERSE)

#*********************************************** function **************************************************
img.rotate(90,expand=True) #to rotate the image, expand to rotate with the original size
img.getbands() #return ('R', 'G', 'B', 'A')
print(img.getbands()) #-->('R', 'G', 'B', 'A')
img.convert('RGB') #converting image mood
R,G,B,A = img.split()  #splits all the layers
reconstructed_img = Image.merge("RGBA",(R,G,B,A)) #recreate the image again after splitting it


#****************************************************** Sheet 3 ********************************************************
# 1. By using pillow library read an image and get the
# following information:
# a)Type of image  b)Name of image  c)Mode of image  d)Size of image  e)Format of image  f) Pixel’s values of image

from PIL import Image
import numpy as np
img = Image.open("img.png")
#a)
type(img)
#b)
img.filename #name file contain image
#c)
img.mode #RGB
#d)
img.size #(W,H)
#e)
img.format #(png)
#f)
img.getpixel((0,0)) #values of a single pixel
np.array(img) #values of all pixels
#-----------------------------------------------------------------------------------------------------------------------
# 2. By Using functions of pillow library perform the following manipulations on the image, and plot all
# images in the same figure:
# a)Crop the image.  b)Resample the image using resize function and reduce function.

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("img.png")

#a)
cropped_img = img.crop((0,0,350,360))
#b)
resampled_img = img.resize((img.width//2,img.height//2))
reduced_img = img.reduce(2)

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 2

plt.subplot(rows,columns,1) #plt.subplot(rows, columns, index)
plt.imshow(img)
plt.title("Original img")
plt.axis("off")

plt.subplot(rows,columns,2)#plt.subplot(rows, columns, index)
plt.imshow(cropped_img)
plt.title("Cropped img ")
plt.axis("off")

plt.subplot(rows,columns,3) #plt.subplot(rows, columns, index)
plt.imshow(resampled_img)
plt.title("Resampled img")
plt.axis("off")

plt.subplot(rows,columns,4) #plt.subplot(rows, columns, index)
plt.imshow(reduced_img)
plt.title("Reduced img")
plt.axis("off")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 3. Save the cropped image.

from PIL import Image

img = Image.open("img.png")
cropped_img = img.crop((0,0,300,300))
cropped_img.save(f"cropped_img.{img.format}") #saves the image according to format of original one
#-----------------------------------------------------------------------------------------------------------------------
# 4. Perform the following transformations using transpose function and plot all images in the same figure:
# a)Flip the image left to right.  b)Flip the image top to bottom.  c)rotate the image 90,180 and 270.
# d)Transposes the rows and columns using the top-left pixel as the origin.
# e)Transposes the rows and columns using the bottom-left pixel as the origin.

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("img.png")
#a)
left_to_right_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#b)
top_to_bottom_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#c)
rotate_90_img = img.transpose(Image.ROTATE_90)
rotate_180_img = img.transpose(Image.ROTATE_180)
rotate_270_img = img.transpose(Image.ROTATE_270)
#d)
transpose_img = img.transpose(Image.TRANSPOSE)
#e)
transverse_img = img.transpose(Image.TRANSVERSE)

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 4

plt.subplot(rows,columns,1) #plt.subplot(rows, columns, index)
plt.imshow(img)
plt.title("Original img")
plt.axis("off")

plt.subplot(rows,columns,2)#plt.subplot(rows, columns, index)
plt.imshow(left_to_right_img)
plt.title("left_to_right_img")
plt.axis("off")

plt.subplot(rows,columns,3)#plt.subplot(rows, columns, index)
plt.imshow(top_to_bottom_img)
plt.title("top_to_bottom_img")
plt.axis("off")

plt.subplot(rows,columns,4)#plt.subplot(rows, columns, index)
plt.imshow(rotate_90_img)
plt.title("rotate_90_img")
plt.axis("off")

plt.subplot(rows,columns,5)
plt.imshow(rotate_180_img)
plt.title("rotate_180_img")
plt.axis("off")

plt.subplot(rows,columns,6)
plt.imshow(rotate_270_img)
plt.title("rotate_270_img")
plt.axis("off")

plt.subplot(rows,columns,7)
plt.imshow(transpose_img)
plt.title("transpose_img")
plt.axis("off")

plt.subplot(rows,columns,8)
plt.imshow(transverse_img)
plt.title("transverse_img")
plt.axis("off")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 5. Rotate image using rotate function.

from PIL import Image
img = Image.open("img.png")
img.rotate(45,expand=False).show()
img.rotate(45,expand=True).show() #prevents cropping the image while rotating
#-----------------------------------------------------------------------------------------------------------------------
# 6. Print the bands of image.

from PIL import Image
img = Image.open("img.png")
print(img.getbands()) #-->('R', 'G', 'B', 'A')
#-----------------------------------------------------------------------------------------------------------------------
# 7. Convert image to another modes.

from PIL import Image

img = Image.open("img.png")
RGB_img = img.convert("RGB")
L_img = img.convert("L")
#-----------------------------------------------------------------------------------------------------------------------
# 8. separate an image into its bands and plot each band individually.

from PIL import Image
import matplotlib.pyplot as plt

pi = Image.open("pi.jpg")
R,G,B = pi.split()

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 2

plt.subplot(rows,columns,1)
plt.imshow(pi)
plt.title("Original pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.imshow(R)
plt.title("Red pi")
plt.axis("off")

plt.subplot(rows,columns,3)
plt.imshow(G)
plt.title("Green pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.imshow(B)
plt.title("Blue pi")
plt.axis("off")

plt.show()
