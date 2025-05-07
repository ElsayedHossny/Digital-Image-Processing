import numpy as np

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt



pi = Image.open("orange.png")

rows = 6
columns = 3

figure_all_in_one = plt.figure(figsize=(6,8))

# 1)BLUR
plt.subplot(rows,columns,1)
plt.imshow(pi.filter(ImageFilter.BLUR))
plt.title("Blur pi")
plt.axis("off")

# 2) DETAIL
plt.subplot(rows,columns,2)
plt.imshow(pi.filter(ImageFilter.DETAIL))
plt.title("Detail pi")
plt.axis("off")

# 3) CONTOUR
plt.subplot(rows,columns,3)
plt.imshow(pi.filter(ImageFilter.CONTOUR)) #حدود للصوره
plt.title("CONTOUR pi")
plt.axis("off")

# 4) EDGE_ENHANCE
plt.subplot(rows,columns,4)
plt.imshow(pi.filter(ImageFilter.EDGE_ENHANCE))  #حدود للصوره اكثر وضوحا
plt.title("EDGE_ENHANCE pi")
plt.axis("off")

# 5) EDGE_ENHANCE_MORE
plt.subplot(rows,columns,5)
plt.imshow(pi.filter(ImageFilter.EDGE_ENHANCE_MORE)) # بشكل أوضح وأقوى. حدود
plt.title("EDGE_ENHANCE_MORE pi")
plt.axis("off")

# 6) EMBOSS
plt.subplot(rows,columns,6)
plt.imshow(pi.filter(ImageFilter.EMBOSS)) # 3D-like effect
plt.title("EMBOSS pi")
plt.axis("off")

# 7) FIND_EDGES
plt.subplot(rows,columns,7)
plt.imshow(pi.filter(ImageFilter.FIND_EDGES)) # يكتشف فقط الحواف ويعرضها.
plt.title("FIND_EDGES pi")
plt.axis("off")

# 8) SMOOTH
plt.subplot(rows,columns,8)
plt.imshow(pi.filter(ImageFilter.SMOOTH))
plt.title("SMOOTH pi")
plt.axis("off")

# 9) SMOOTH_MORE
plt.subplot(rows,columns,9)
plt.imshow(pi.filter(ImageFilter.SMOOTH_MORE))
plt.title("SMOOTH_MORE pi")
plt.axis("off")

# 10) SHARPEN
plt.subplot(rows,columns,10)
plt.imshow(pi.filter(ImageFilter.SHARPEN)) # يزود وضوح الصورة. عكس blur
plt.title("SHARPEN pi")
plt.axis("off")

# 11) MAXFILTER
plt.subplot(rows,columns,11)
plt.imshow(pi.filter(ImageFilter.MaxFilter(3)))
plt.title("MaxFilter pi")
plt.axis("off")

# 12) MEDIANFILTER
plt.subplot(rows,columns,12)
plt.imshow(pi.filter(ImageFilter.MedianFilter(3)))
plt.title("MedianFilter pi")
plt.axis("off")

# 13) MINFILTER
plt.subplot(rows,columns,13)
plt.imshow(pi.filter(ImageFilter.MinFilter(3)))
plt.title("MinFilter pi")
plt.axis("off")

# 14) MODEFILTER
plt.subplot(rows,columns,14)
plt.imshow(pi.filter(ImageFilter.ModeFilter(3))) # أكثر قيمة متكررة (mode) من البكسلات المحيطة.
plt.title("ModeFilter pi")
plt.axis("off")

# 15) GAUSSIANBLUR
plt.subplot(rows,columns,15)
plt.imshow(pi.filter(ImageFilter.GaussianBlur(2))) # تنعيم باستخدام توزيع غاوسي (Gaussian).
plt.title("GaussianBlur pi")
plt.axis("off")

# 16) BOXBLUR
plt.subplot(rows,columns,16)
plt.imshow(pi.filter(ImageFilter.BoxBlur(2)))
plt.title("GaussianBlur pi")
plt.axis("off")

# 17) UNSHARPMASK
plt.subplot(rows,columns,17)
plt.imshow(pi.filter(ImageFilter.UnsharpMask(radius = 2,percent = 150, threshold=100)))
plt.title("GaussianBlur pi")
plt.axis("off")

#18) EMBOSS
plt.subplot(rows,columns,18)
plt.imshow(pi.filter(ImageFilter.EMBOSS))
plt.title("GaussianBlur pi")
plt.axis("off")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 2-By Using ImageFilter. Kernel and apply any kernel to the image.

pi = Image.open("orange.png")

kernel = [1,2,1,0,0,0,-1,-2,-1]

kernel_pi = pi.filter(ImageFilter.Kernel((3,3),kernel,scale = 1))

kernel_pi.show()

print(kernel)
#-----------------------------------------------------------------------------------------------------------------------
# 3-using the following kernels, numpy module and convolve function from SciPy module ,write a code to apply the
# following kernels to an image.


import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import matplotlib.pyplot as plt

pi_grey = Image.open("orange.png").convert("L")

pi_grey_array = np.array(pi_grey) # convert image to مصفوفه

kernels = {
    "Horizontal Sobel" : np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    "Vertical Sobel"   : np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),
    "Left Diagonal"    : np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]),
    "Right Diagonal"   : np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1]]),
    "Edge Detection"   : np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "Sharpen"          : np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "Box Blur"         : (1 / 9.0 ) * np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]),
    "Gaussian Blur"    : (1 / 16.0) * np.array([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
}

figure_all_in_one = plt.figure(figsize = (10,20))
rows = 4
columns = 2

for i, (kernel_name,kernel) in enumerate(kernels.items(),start = 1):
    filtered_array = convolve(pi_grey_array,kernel)
    filtered_image = np.clip(filtered_array,0,255) # cut image between min,max
    plt.subplot(rows,columns,i)
    plt.imshow(filtered_image, cmap = "gray")
    plt.title(kernel_name)
    plt.axis("off")

plt.show()