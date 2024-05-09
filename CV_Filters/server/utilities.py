# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import ndimage
from skimage import io
from scipy import ndimage
import PIL
from PIL import Image
import pandas as pd
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import histogram, cumulative_distribution
from IPython.display import display, Math, Latex
import matplotlib.image as mpimg
import random
import copy
from scipy.stats import norm



# ------------------ IMAGE FUNCTIONS -------------------------------------

#function gray scale 
def rgbtogray(image):
    r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
    # img=np.array(Image.open(image)) #Input - Color image
    # gray_img=img.copy()

    # for clr in range(img.shape[2]):
    #     gray_img[:,:,clr]=img.mean(axis=2) #Take mean of all 3 color channels of each pixel and assign it back to that pixel(in copied image)

    # plt.imshow(gray_img) #Result - Grayscale image
    # return gray_img
# -----------------------NOISE ------------------------------------------
def rgb2gray(image):
    img=np.array(Image.open(image)) #Input - Color image
    gray_img=img.copy()

    for clr in range(img.shape[2]):
        gray_img[:,:,clr]=img.mean(axis=2) #Take mean of all 3 color channels of each pixel and assign it back to that pixel(in copied image)

    plt.imshow(gray_img) #Result - Grayscale image
    return gray_img

def salt_and_pepper(img):
    row , col = img.shape
    g = np.zeros((row,col), dtype=np.float32)
    salt=0.95
    pepper=0.1
    for i in range(row):
        for j in range(col):
            rdn = np.random.random()
            if rdn < pepper:
                g[i][j] = 0
            elif rdn > salt:
                g[i][j] = 1
            else:
                g[i][j] = img[i][j]
    return g

#low : lower boundry of output interval
#high: Upper boundary of the output interval. All values generated will be less than or equal to high. The high limit may be included in the returned array of floats due to floating-point rounding in the equation low + (high-low) * random_sample(). The default value is 1.0.
def uniform_noise(img): 
    row,col=img.shape 
    uni_noise=np.zeros((row,col),dtype=np.uint8) 
    # fills array with uniformly-distributed random numbers from the range [low, high) cv2.randu(inputoutputarray,low,high) 
    cv2.randu(uni_noise,0,255) 
 
    uni_noise=(uni_noise*0.5).astype(np.uint8) 
    un_img= img + uni_noise 
    return un_img


def gaussian_noise(img):

    row,col = img.shape
    mean = 0
    var = 300
    sigma = var**0.5
    gaussion_noise = np.random.normal(loc=mean, scale=sigma, size=(row,col))
    img = img + gaussion_noise

    return img

# ------------------ LOW PASS FILTER -------------------------------------


def median_filter(noise_image):

    row,col=noise_image.shape
    filtered_image=np.zeros([row,col])
    #loop on every window 3*3 in the image
    for i in range (1,row-1):
        for j in range (1,col-1):
            image=[noise_image[i-1, j-1],
                   noise_image[i-1, j],
                   noise_image[i-1, j + 1],
                   noise_image[i, j-1],
                   noise_image[i, j],
                   noise_image[i, j + 1],
                   noise_image[i + 1, j-1],
                   noise_image[i + 1, j],
                   noise_image[i + 1, j + 1]]
            image=sorted(image)
            filtered_image[i, j]=image[4]
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image


def GaussianLowFilter(img):

    # transform the image into frequency domain, f --> F
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    
    # Create Gaussin Filter: Low Pass Filter
    M,N = img.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 10
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))
            
    # Image Filters
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    return g



def meanLowPass(img):

    # Obtain number of rows and columns 
    # of the image
    m, n = img.shape

    # Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9

    # Convolve the 3X3 mask over the image 
    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

            img_new[i, j]= temp

    img_new = img_new.astype(np.uint8)
    return img_new


# ------------------ Frequency domain FILTER -------------------------------------
def idealLowPass(img):
        # image in frequency domain
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M,N = img.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 50
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u,v] = 1
            else:
                H[u,v] = 0

    # Ideal Low Pass Filtering
    Gshift = Fshift * H

    # Inverse Fourier Transform
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    return g



def IdealHighPass(img):
    # image in frequency domain
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M,N = img.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 50
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u,v] = 1
            else:
                H[u,v] = 0             
    # Filter: High pass filter
    H = 1 - H
    # Ideal High Pass Filtering
    Gshift = Fshift * H
    # Inverse Fourier Transform
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    return g

# ------------------ HIGH PASS FILTER -------------------------------------


def sobel(img):
#     img=GaussianLowFilter(img)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    
    return G

def robert(img):

    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    return edged_img

def prewit(img):

    #define horizontal and Vertical sobel kernels
    Hx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    Hy = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
    #normalizing the vectors
    pre_x = convolve(img, Hx) / 6.0
    pre_y = convolve(img, Hy) / 6.0
    #calculate the gradient magnitude of vectors
    pre_out = np.sqrt(np.power(pre_x, 2) + np.power(pre_y, 2))
    # mapping values from 0 to 255
    pre_out = (pre_out / np.max(pre_out)) * 255
    return pre_out

# ------------------ CONVOLUTION -------------------------------------

def convolve(X, F):
    # height and width of the image
    X_height = X.shape[0]
    X_width = X.shape[1]
    
    # height and width of the filter
    F_height = F.shape[0]
    F_width = F.shape[1]
    
    H = (F_height - 1) // 2
    W = (F_width - 1) // 2
    
    #output numpy matrix with height and width
    out = np.zeros((X_height, X_width))
    #iterate over all the pixel of image X
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            #iterate over the filter
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    #get the corresponding value from image and filter
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i,j] = sum
    #return convolution  
    return out


# ------------------ PLOTTING -------------------------------------

def histogram(img):
    # convert our image into a numpy array
    img = np.asarray(img)

    # put pixels in a 1D array by flattening out img array
    flat = img.flatten()

    # show the histogram
    plt.hist(flat, bins=50)
    path = f'server/static/assests/histogram.jpg'
    plt.savefig(path)
    plt.close()



def cumm_dist(img):
    plt.hist(img.ravel(), bins = 256, cumulative = True)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count') 
    path = f'server/static/assests/cum_dist.jpg'
    plt.savefig(path)
    plt.close()


def rgbHistogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
    blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])
    data=[red_hist,green_hist,blue_hist]
    mu, std = norm.fit(data) 

    plt.subplot(3, 1, 1)

    plt.subplot(3, 1, 1)
    plt.plot(red_hist, color='r')
    plt.xlim([0, 255])

    plt.subplot(3, 1, 2)
    plt.plot(green_hist, color='g')
    plt.xlim([0, 255])

    plt.subplot(3, 1, 3)
    plt.plot(blue_hist, color='b')
    plt.xlim([0, 255])

    plt.tight_layout()
    path = f'server/static/assests/rgbhistogram.jpg'
    plt.savefig(path)
    plt.close()

    plt.plot(red_hist,color="r")
    plt.plot(green_hist,color="g")
    plt.plot(blue_hist,color="b")
    path1 = f'server/static/assests/rgbhistogram1.jpg'
    plt.savefig(path1)
    plt.close()

    plt.subplot(3, 1, 1)

    plt.subplot(3, 1, 1)
    plt.hist(red_hist,color="r")
    plt.title('red histogram')

    plt.subplot(3, 1, 2)
    plt.hist(green_hist,color="g")
    plt.title('green histogram')

    plt.subplot(3, 1, 3)
    plt.hist(blue_hist,color="b")
    plt.title('blue histogram')

    plt.tight_layout()  
    path2 = f'server/static/assests/rgbhistogram2.jpg'
    plt.savefig(path2)
    plt.close()


# ------------------ EQUALIZATION AND NORMALIZATION -------------------------------------
# create our own histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

# create our cumulative sum function
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def equalization(path):

    img = Image.open(path)

    # convert image into a numpy array
    img = np.asarray(img)
    # put pixels in a 1D array by flattening out img array
    flat = img.flatten()
    # re-normalize cumsum values to be between 0-255
    hist = get_histogram(flat, 256)
    cs = cumsum(hist)
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()



    # re-normalize the cdf
    cs = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[flat]
    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, img.shape)
    return img_new

    
def normalize(img):
    # norm_img = np.zeros((800,800))
    # final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    # return final_img
    min=float(img.min())
    max=float(img.max())
    return np.floor((img-min)/(max-min)*255.0)
    
# ------------------ THRESHOLDING -------------------------------------

def globalThresholding(img, n):
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]
    for row in range(width):
        for column in range(height):
            if img[column, row] > n:
                img[column, row] = 0
            else:
                img[column, row] = 255
    return img   


def localThresholding(img, radius):
    image = np.zeros_like(img)
    max_row, max_col = img.shape
    for i, row in enumerate(img):
        y_min = max(0, i - radius)
        y_max = min(max_row, i + radius + 1)
        for j, elem in enumerate(row):
            x_min = max(0, j - radius)
            x_max = min(max_col, j + radius + 1)
            window = img[y_min:y_max, x_min:x_max]
            if img[i, j] >= np.median(window):
                image[i, j] = 255
    return image

# ------------------ Canny -------------------------------------

def get_gaussian_kernel(kernal_size, sigma=1):
    gaussian_kernal = np.zeros((kernal_size, kernal_size), np.float32)
    size = kernal_size//2

    for x in range(-size, size+1):
        for y in range(-size, size+1):
            a = 1/(2*np.pi*(sigma**2))
            b = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_kernal[x+size, y+size] = a*b
    return gaussian_kernal/gaussian_kernal.sum()

def apply_filtering(input_image, kernal):
    
    output_image = []
    kernal_size = len(kernal)
    kernal_half = kernal_size // 2
    rows_count = len(input_image)
    columns_count = len(input_image[0])

    image_copy = copy.deepcopy(input_image)

    # wrap the image in the edge pixels
    for i in range(rows_count):
        for j in range(kernal_half): 
            image_copy[i].insert(0, input_image[i][-1-j])
            image_copy[i].append(input_image[i][j])
    for i in range(kernal_half):
        image_copy.append(image_copy[2*i])
        image_copy.insert(0, image_copy[-2-2*i].copy())

    # apply filtering
    new_rows_count = len(image_copy)
    new_columns_count = len(image_copy[0])

    for i in range(kernal_half, new_rows_count - kernal_half):
        output_row = []
        for j in range(kernal_half, new_columns_count - kernal_half):
            sum = 0
            for x in range(len(kernal)):
                for y in range(len(kernal)):
                    x1 = i + x - kernal_half
                    y1 = j + y - kernal_half
                    sum += image_copy[x1][y1] * kernal[x][y]
            output_row.append(sum)
        output_image.append(output_row)

    return output_image

# step 3 : gradient estimation

def gradient_estimate(image, gradient_estimation_filter_type):

    if (gradient_estimation_filter_type=="sobel"):
        Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    elif (gradient_estimation_filter_type=="prewitt"):
        Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        My = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
    else:
        Mx = np.array([[0, 1], [-1, 0]], np.float32)
        My = np.array([[1, 0], [0, -1]], np.float32)

    X = apply_filtering(image, Mx)
    Y = apply_filtering(image, My)

    G = np.hypot(X, Y)
    G = G / G.max() * 255
    theta = np.arctan2(Y, X)

    return (G, theta)

# step 4 : non-maxima suppression to thin out the edges

def non_maxima_suppression(image, gradient_direction):
    rows_count = len(image)
    columns_count = len(image[0])

    output_image = np.zeros((rows_count, columns_count), dtype=np.int32)
    theta = gradient_direction * 180. / np.pi
    theta[theta < 0] += 180

    
    for i in range(1, rows_count-1):
        for j in range(1, columns_count-1):
            next = 255
            previous = 255
            if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                next = image[i, j+1]
                previous = image[i, j-1]
            elif (22.5 <= theta[i,j] < 67.5):
                next = image[i+1, j-1]
                previous = image[i-1, j+1]
            elif (67.5 <= theta[i,j] < 112.5):
                next = image[i+1, j]
                previous = image[i-1, j]
            elif (112.5 <= theta[i,j] < 157.5):
                next = image[i-1, j-1]
                previous = image[i+1, j+1]

            if (image[i,j] >= next) and (image[i,j] >= previous):
                output_image[i,j] = image[i,j]
            else:
                output_image[i,j] = 0
    
    return output_image

def double_threshold(image, low_threshold_ratio, high_threshold_ratio):
    
    high_threshold = image.max() * high_threshold_ratio;
    low_threshold = high_threshold * low_threshold_ratio;
    
    rows_count = len(image)
    columns_count = len(image[0])
    output_image = np.zeros((rows_count, columns_count), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)

    strong_i = []
    strong_j = []
    weak_i = [] 
    weak_j = []
    for i in range (len(image)):
        for j in range (len(image[0])):
            if (image[i,j]>=high_threshold):
                strong_i.append(i)
                strong_j.append(j)
            if ((image[i,j] <= high_threshold) & (image[i,j] >= low_threshold)):
                weak_i.append(i)
                weak_j.append(j)
    strong_i = np.array(strong_i)
    strong_j = np.array(strong_j)
    weak_i = np.array(weak_i)
    weak_j = np.array(weak_j)
    
    output_image[strong_i, strong_j] = strong
    output_image[weak_i, weak_j] = weak


    
    return (output_image, weak, strong)

def hysteresis_edge_track(image, weak, strong=255):
    rows_count = len(image)
    columns_count = len(image[0]) 
    for i in range(1, rows_count-1):
        for j in range(1, columns_count-1):
            if (image[i,j] == weak):
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                    or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                    or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny(img):
    kernal_size = 3
    low_threshold_ratio = 0.05
    high_threshold_ratio = 0.09
    gradient_estimation_filter_type = "sobel"
   

    # step 2 : apply gaussian kernal to filter noise
    kernal = get_gaussian_kernel(kernal_size)
    image_without_noise = apply_filtering(img.tolist(), kernal)

    # step 3 : gradient estimation
    assert (gradient_estimation_filter_type in ["sobel", "prewitt", "robert"]), "gradient estimation filter type should be [\"prewitt\", \"sobel\", \"robert\"]"
    G, theta = gradient_estimate(image_without_noise, gradient_estimation_filter_type)

    # step 4 : non maxima suppression
    image_with_thin_edges = non_maxima_suppression(G, theta)

    # step 5 : double threshold
    final_image, weak, strong = double_threshold(image_with_thin_edges, low_threshold_ratio, high_threshold_ratio)

    # edge tracking with hysteresis
    img = hysteresis_edge_track(final_image, weak, strong=255)
    return img
# ------------------ Hybrid image -------------------------------------
def convolution(image, kernel):
    """ This function executes the convolution between `img` and `kernel`.
    """

    # Flip template before convolution.
    kernel = cv2.flip(kernel, -1)
    # Get size of image and kernel. 3rd value of shape is colour channel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    (pad_h, pad_w) = (kernel_h // 2, kernel_w // 2)
    # Create image to write to.
    output = np.zeros(image.shape)
    # Slide kernel across every pixel.
    for y in range(pad_h, image_h - pad_h):
        for x in range(pad_w, image_w - pad_w):
            # If coloured, loop for colours.
            for colour in range(image.shape[2]):
                # Get center pixel.
                center = image[
                    y - pad_h : y + pad_h + 1, x - pad_w : x + pad_w + 1, colour
                ]
                # Perform convolution and map value to [0, 255].
                # Write back value to output image.
                output[y, x, colour] = (center * kernel).sum() / 255

    # Return the result of the convolution.
    return output


def fourier(image, kernel):
    """ Compute convolution between `img` and `kernel` using numpy's FFT.
    """
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    # Apply padding to the kernel.
    padded_kernel = np.zeros(image.shape[:2])
    start_h = (image_h - kernel_h) // 2
    start_w = (image_w - kernel_w) // 2
    padded_kernel[start_h : start_h + kernel_h, start_w : start_w + kernel_w] = kernel
    # Create image to write to.
    output = np.zeros(image.shape)
    # Run FFT on all 3 channels.
    for colour in range(3):
        Fi = np.fft.fft2(image[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        # Inverse fourier.
        output[:, :, colour] = np.fft.fftshift(np.fft.ifft2(Fi * Fk)) / 255

    # Return the result of convolution.
    return output



def gaussian_blur(image, sigma, flag):
    """ Builds a Gaussian kernel used to perform the LPF on an image.
    """

    # Calculate size of filter.
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros((size, size))

    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    if flag:
        return fourier(image, kernel)
    else:
        return convolution(image, kernel)
    
def low_pass(image, cutoff, flag):
    """ Generate low pass filter of image.
    """
    return gaussian_blur(image, cutoff, flag)    


def high_pass(image, cutoff, flag):
    """ Generate high pass filter of image. This is simply the image minus its
    low passed result.
    """
    return ((image) / 255) - low_pass(image, cutoff, flag)



def hybrid_image(images, cutoff, flag):
    """ Create a hybrid image by summing together the low and high frequency
    images.
    """
    # Perform low pass filter and export.
    low = low_pass(images[0], cutoff[0], flag)
#     cv2.imwrite("low.jpg", low * 255)
    # Perform high pass filter and export.
    high = high_pass(images[1], cutoff[1], flag)
#     cv2.imwrite("high.jpg", (high + 0.5) * 255)
    
    result = low + high 
    print("Creating hybrid image...")
    return result