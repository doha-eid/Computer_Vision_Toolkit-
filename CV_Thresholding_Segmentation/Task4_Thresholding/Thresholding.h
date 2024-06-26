#ifndef THRESHOLDING_H
#define THRESHOLDING_H
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
/*************************************************Function Prototypes*******************************************************/

Mat otsu_thresholding (Mat image);
map<int, int> histo (Mat image);
Mat thresholding(Mat gray_image, int threshold);
Mat optimal_thresholding (Mat image);
Mat Apply_Fourier_Transform( Mat &src);
Mat Inverse_Fourier_Transform( Mat &src);
Mat spectral_thresholding (Mat image);
Mat double_thresholding(Mat gray_image, int min_threshold, int max_threshold);
Mat localThresholding(const cv::Mat& inputImage, int windowSize, int threshold);
Mat spectral_localThresholding (Mat image);
#endif // THRESHOLDING_H
