#include "harris.h"
#include <vector>
using namespace std;
using namespace cv;

Harris::Harris()
{

}

Harris::~Harris()
{

}

Mat imageMat, imageMatGray, dst;
int thresh = 128;
int max_thresh = 255;

void Harris::cornerHarris_self(double k)
{
    Mat abs_grad_x, abs_grad_y, x2y2, xy, mtrace;
    Mat x2_derivative, y2_derivative, xy_derivative, x2g_derivative, y2g_derivative, xyg_derivative;
    int scale = 1;
    int delta = 0;
    //Step one
     //to calculate x and y derivative of image we use Sobel function
     //Sobel( srcimage, dstimage, depthofimage -1 means same as input, xorder 1,yorder 0,kernelsize 3, BORDER_DEFAULT);
    Sobel( imageMatGray, abs_grad_x, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
    Sobel(imageMatGray, abs_grad_y, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);

    //calculating other three images in M
    pow(abs_grad_x, 2.0, x2_derivative);
    pow(abs_grad_y, 2.0, y2_derivative);
    multiply(abs_grad_x, abs_grad_y, xy_derivative);

    //step three apply gaussain
    GaussianBlur(x2_derivative, x2g_derivative, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(y2_derivative, y2g_derivative, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(xy_derivative, xyg_derivative, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);

    //forth step calculating R with k=0.04
    multiply(x2g_derivative, y2g_derivative, x2y2);
    multiply(xyg_derivative, xyg_derivative, xy);
    pow((x2g_derivative + y2g_derivative), 2.0, mtrace);
    dst = (x2y2 - xy) - k * mtrace;
}

Mat Harris::cornerHarris_demo(Mat src,int, void*)
{
    double k = 0.04;
    Harris::cornerHarris_self(k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    Mat src_with_corners = src.clone();
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int thresh=128 ;

            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(src_with_corners, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);
            }
        }

    }
    return src_with_corners;
}

Mat Harris::applyHarris(Mat imageMat){

    cv::cvtColor(imageMat,imageMatGray, COLOR_BGR2GRAY);
    Mat HarrisImage= Harris::cornerHarris_demo(imageMat,0, 0);
    return HarrisImage;
}

vector<Point2f> Harris::detectCorners(const Mat &image, double threshold) {
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    // Calculate the gradients in x and y directions
    Mat grad_x, grad_y;
    Sobel(grayscale, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(grayscale, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    Mat hessian_xx, hessian_xy, hessian_yy;
    // Calculate the second-order partial derivatives of the image intensity
    // using the Sobel gradients
    Sobel(grad_x, hessian_xx, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(grad_x, hessian_xy, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    Sobel(grad_y, hessian_yy, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    vector<Point2f> corners;
    Mat hessian_matrix, eigenvalues_mat;
    float k = 0.04;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Compute the Hessian matrix at each pixel
            hessian_matrix = (Mat_<float>(2, 2) <<
                hessian_xx.at<float>(y, x),
                hessian_xy.at<float>(y, x),
                hessian_xy.at<float>(y, x),
                hessian_yy.at<float>(y, x));

            // Compute the Harris response function for the pixel
            eigen(hessian_matrix, eigenvalues_mat);
            float lambda1 = eigenvalues_mat.at<float>(0, 0);
            float lambda2 = eigenvalues_mat.at<float>(1, 0);
            float harris_response = lambda1 * lambda2 - k * pow(lambda1 + lambda2, 2);

            // If the Harris response is above the threshold, add the pixel to the list of corners
            if (harris_response > threshold) {
                corners.push_back(Point2f(x, y));
            }
        }
    }

    return corners;
}

Mat Harris::applyEigen(Mat image){

    vector<Point2f> corners = detectCorners(image, 100000);

    // Draw circles at the locations of the detected corners
    for (int i = 0; i < corners.size(); ++i) {
        circle(image, corners[i], 5, Scalar(0, 0, 255), FILLED);
    }

    return image;
}
