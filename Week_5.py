
'''

# Histogram Processing Techniques:
# 1- grayscale
# 2- Sliding
# 3- Stretching
# 4- Equalization
#********************************Histogram توزيع قيم البكسلات في صورة  by grayscale.******************************************************
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("orange.png").convert("L")  # Convert to grayscale

histogram = image.histogram() # create

plt.figure(figsize=(10, 5)) #حجم الرسم البياني
plt.bar(range(256), histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.title('Grayscale Image Histogram')
plt.show()

#******************************************* Histogram Sliding(سطوع الصورة) **************************************************
# هي تقنية نغير بها درجة (سطوع الصورة) عن طريق إضافة أو طرح قيمة ثابتة إلى جميع قيم البكسلات.
# إذا أضفنا قيمة = نزيد السطوع → يُسمى Right Sliding.
# إذا طرحنا قيمة = نقلل السطوع → يُسمى Left Sliding.

def slide_hist(img,shift):
    fill = np.ones(img.shape,dtype = np.uint8) * shift
    return cv2.add(img,fill)

#*************************************** Histogram Stretching(زيادة التباين في الصورة) ************************************************
import numpy as np
import cv2

pi_grey = cv2.imread("orange.png", 0)
constant = (255-0)/(pi_grey.max()-pi_grey.min())  # --First step--

pi_grey_stretch = (pi_grey - pi_grey.min()) * constant #نحول كل قيمة بكسل من مداها الحالي إلى المدى الجديد.
pi_grey_stretch = np.clip(pi_grey_stretch,0,255).astype(np.uint8) #يضمن أن القيم ما تطلعش عن 0 أو 255.

#************************************** Histogram Equalization(تحسين التباين)***********************************************
# تحسين التباين، خاصة في الصور ذات الإضاءة الضعيفة أو التباين المحدود.
import cv2
pi_grey = cv2.imread("orange.png", 0)
pi_grey_equalized = cv2.equalizeHist(pi_grey) # equalizeHist() تعمل فقط على الصور الرمادية (grayscale)


'''

#******************************************************* Sheet 5 *******************************************************
#1.  Read an image and convert it to gray scale, then by using matplotlib library then show the image and its histogram.
import matplotlib.pyplot as plt
from PIL import Image

pi_grey = Image.open("orange.png").convert("L")
pi_grey_histogram = pi_grey.histogram() #create
rows = 1
columns = 2

figure_all_in_one = plt.figure(figsize=(6,8)) #size of histogram

plt.subplot(rows,columns,1) #1->index
plt.imshow(pi_grey)
plt.title("Pi Image")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.title("Histogram")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#2. Read an image and convert it to gray scale using pillow library, then show the image and its histogram.
from PIL import Image
import matplotlib.pyplot as plt

pi_grey = Image.open("orange.png").convert("L")  # Convert to grayscale
pi_grey.show()

pi_grey_histogram = pi_grey.histogram()

plt.figure(figsize=(8, 5))
plt.plot(pi_grey_histogram)
plt.title("Grayscale Image Histogram")
plt.show()

# Another application in question 2 is to obtain the histogram of a colored image, then extract the intensity ranges
# for the Red, Green, and Blue (RGB) channels and visualize them separately.from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt

pi = Image.open("orange.png")
pi.show()
var = pi.histogram()

pi_red   = var[0:256]
pi_green = var[256:512]
pi_blue  = var[512:768]

plt.plot(pi_red,color = "red")
plt.plot(pi_green,color = "green")
plt.plot(pi_blue,color = "blue")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#3.Read an image and convert it to gray scale using open CV library, then show the image and its histogram.
import cv2
import matplotlib.pyplot as plt

pi_grey = cv2.imread("orange.png",flags = 0)

var  = cv2.calcHist([pi_grey],[0],None,[256],[0, 255]) # حسب التوزيع الإحصائي لقيم البكسلات من 0 إلى 255
# [pi_grey] : the image that we will calculate its histogram, this parameter can take more than one image
# [0]       : the channel number targeted (in case the image is grey scale so channel = 0)
# None      : disable masking, it means we want to calculate the histogram of the entire image not part of it
# [256]     : the number of bins we are using for the graph, decreasing it decreases graph resolution
# [0, 255]   : the range of pixel values

plt.plot(var)
plt.show()

cv2.imshow("Pi grey image",pi_grey)
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#4. Apply histogram stretching on gray image and show the images after histogram stretching and its histogram.
import cv2
import matplotlib.pyplot as plt
import numpy as np

pi_grey = cv2.imread("orange.png", 0)
constant = (255-0)/(pi_grey.max()-pi_grey.min())

pi_grey_stretch = (pi_grey-pi_grey.min()) * constant
pi_grey_stretch = np.clip(pi_grey_stretch,0,255).astype(np.uint8)

rows = 2
columns = 2
figure_all_in_one = plt.figure(figsize=(6,8))

plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.hist(pi_grey.ravel(),256)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_stretch)
plt.title("Stretched Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.hist(pi_grey_stretch.ravel(),256)
plt.title("Stretched Pi Histogram")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#5-  Apply left and right histogram Sliding on gray image and show the images after sliding and their histogram.
import cv2
import matplotlib.pyplot as plt
import numpy as np

def slide_hist(img,shift):
    fill = np.ones(img.shape,dtype = np.uint8) * shift
    return cv2.add(img,fill)

pi_grey = cv2.imread("orange.png", 0)
pi_grey_histogram = cv2.calcHist([pi_grey],[0],None,[256],[0,255])

pi_grey_shift = slide_hist(pi_grey,20)
pi_grey_shift_histogram = cv2.calcHist([pi_grey_shift],[0],None,[256],[0,255])

rows = 2
columns = 2
plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_shift)
plt.title("Shifted Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.plot(pi_grey_shift_histogram)
plt.title("Shifted Pi Histogram")

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#6. Apply histogram equalization on gray image. Show the original image ,the histogram of the original image, the image
# after equalization and its histogram in the same figure.
import cv2
import matplotlib.pyplot as plt

pi_grey = cv2.imread("orange.png", 0)
pi_grey_histogram = cv2.calcHist([pi_grey],[0],None,[256],[0,256])
pi_grey_equalized = cv2.equalizeHist(pi_grey)
pi_grey_equalized_histogram = cv2.calcHist([pi_grey_equalized],[0],None,[256],[0,256])

rows = 2
columns = 2
plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_equalized)
plt.title("Equalized Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.plot(pi_grey_equalized_histogram)
plt.title("Equalized Pi Histogram")

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
