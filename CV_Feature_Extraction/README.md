[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/FSQAR7Bu)
# CV-A03

# README
Overview:
This project is an educational desktop application that performs many functions on uploaded images and visualizing the effect of each of these functions by displaying the output and input images to show difference between them all in one page .
All functions in our application are written using C++ language without using any built-in functions of the OpenCV library.

The repository contains a feature extraction and matching implementation using Harris corner detection, minimum eigenvalue operator, Scale-Invariant Feature Transform (SIFT), and image feature matching using Sum of Squared Differences (SSD) and Normalized Cross-Correlations.

## Features

- Extract unique features using Harris operator
- Extract unique features using minimum eigenvalue operator
- Generate feature descriptors using Scale-Invariant Feature Transform (SIFT)
- Match image features using Sum of Squared Differences (SSD) and Normalized Cross-Correlations

## Implemented Algorithms Steps

### Harris Function

1. Define a class named "Harris"
2. Define a function named "cornerHarris_self" that takes a double value k as a parameter
3. Create Mat objects to hold the absolute gradient in the x direction, absolute gradient in the y direction, x2, y2, xy, x2 Gaussian, y2 Gaussian, and xy Gaussian
4. Set the scale and delta values for the Sobel function
5. Use the Sobel function to calculate the x and y derivatives of the input grayscale image
6. Calculate the other three images in M (x2, y2, and xy) using the x and y derivatives
7. Apply Gaussian blur to each of the three images in M
8. Calculate the Harris response function R using the formula (x2 y2 – xy2) - k(x2 + y2)2
9. Define a function named "cornerHarris_demo" that takes a Mat object "src" and two void pointers as parameters
10. Call the "cornerHarris_self" function with a value of k
11. Normalize the Harris response function values between 0 and 255
12. Convert the normalized Harris response function values to integer format
13. Create a copy of the input image to draw the detected corners on
14. Loop through all pixels in the normalized Harris response function matrix
15. If the Harris response value at the pixel is above a specified threshold, draw a circle around the pixel on the output image
16. Return the output image with detected corners drawn on it

### Minimum Eigenvalue with Harris Operator

1. Define a function named detectCorners that takes an input image and a threshold value as parameters.
2. Convert the input image to grayscale using the cvtColor function from OpenCV
3. Compute the gradients in the x and y directions using the Sobel function from OpenCV
4. Compute the second-order partial derivatives of the image intensity using the Sobel gradients to calculate the Hessian matrix at each pixel
5. Compute the Harris response function for each pixel using the eigenvalues of the Hessian matrix
6. If the Harris response is above the specified threshold, add the pixel to the list of detected corners
7. Return a vector of Point2f objects representing the locations of the detected corners
8. In the main function, load an input image using the imread function from OpenCV
9. Call the detectCorners function to detect corners in the image with a threshold value of 100000
10. Visualize the detected corners by drawing circles at their locations using the circle function from OpenCV
11. Display the resulting image in a window using the imshow function from OpenCV

### Generate Features using Scale-Invariant Feature Transform (SIFT)

#### Feature Extraction Algorithms

##### Using Cross-Correlation

The function named "normalized_cross_correlation" calculates the mean of the two input arrays, computes the numerator and denominator of the normalized cross-correlation formula separately, checks if the denominator is equal to zero (to avoid division by zero), and returns the resulting normalized cross-correlation value. The normalized cross-correlation between two arrays always falls within the range of -1 to 1, where a value of 1 indicates a perfect match, 0 indicates no correlation, and -1 indicates a perfect anti-correlation.

##### Using Sum of Squared Differences (SSD)

The function returns the sum of squared differences between the two arrays, which is equivalent to the SSD.


## Contributers

#### Team 21

1. Mariam Mohamed Ezzat
2. Amira Mohamed Abdel-Fattah
3. Mayar Ehab Mohamed
4. Maha Medhat Fathy
5. Doha Eid
