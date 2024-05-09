#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include"Thresholding.h"
#include"Region_Growing.h"
#include"Kmeans.h"
#include"agglomerative.h"
#include"meanShift.h"
using namespace std;
using namespace cv;


//int main()
//{
//    Mat image;
//    image = imread("C:\\Users\\w\\Music\\Lenna_(test_image) (3).png", IMREAD_COLOR);
//    /*Otsu Thresholding */
////    Mat threshold_image = otsu_thresholding(image);
////    namedWindow("Otsu_Image", WINDOW_NORMAL);
////    imshow("Otsu_Image", threshold_image);
////    /*Optimal Thresholding*/
////    Mat threshold_image_Optimal = optimal_thresholding(image);
////   namedWindow("Optima_Threshold", WINDOW_NORMAL);
////   imshow("Optima_Threshold", threshold_image_Optimal);
////   /*Spectral Thresholding*/
////   Mat threshold_image_spectral = spectral_thresholding(image);
////   namedWindow("Spectral_Thresholding",WINDOW_NORMAL);
////   imshow("Spectral_Thresholding", threshold_image_spectral);

////   Mat threshold_image_spectral_local = spectral_localThresholding(image);
////   namedWindow("Spectral_local_Thresholding",WINDOW_NORMAL);
////   imshow("Spectral_local_Thresholding", threshold_image_spectral_local);
////        Mat agglomerative_image = Agglomerative::agglomarativeSegmentation(image,7);
////        namedWindow("agglomerative_show", WINDOW_NORMAL);
////        imshow("agglomerative_show", agglomerative_image);

//   /************************************K means***********************/
////    Mat kmeans_image=K_means(image);
////    namedWindow("kmeans_show", WINDOW_NORMAL);
////    imshow("kmeans_show", kmeans_image);
//    /***************************Local Thresholding**************************/

////       // Perform local thresholding on the input image
////       Mat Loacl_Image = localThresholding(image, 31, 10);

////       // Display the binary image
////       namedWindow("Local_Thresholding", WINDOW_NORMAL);
////       imshow("Local_Thresholding", Loacl_Image);

//        waitKey(0);
//        return 0;
//}




int main() {
    // Load an image
//    Mat img = imread("C:\\Users\\w\\Music\\Lenna_(test_image) (3).png", IMREAD_GRAYSCALE);

//    // Create a mask with the same size as the image, initialized with zeros
//    Mat mask(img.size(), CV_8UC1, Scalar(0));

//    // Set the seed point
//    Point seed(100, 100);

//    // Set the threshold for the difference in intensity between neighboring pixels
//    int threshold = 10;


//    // Apply the region growing algorithm
//    region_growing(img, seed, mask, threshold);

//    // Display the resulting mask
//    namedWindow("Result", WINDOW_NORMAL);
//    imshow("Result", mask);
//    waitKey(0);


    return 0;
}


