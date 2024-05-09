#ifndef REGION_GROWING_H
#define REGION_GROWING_H

#include <opencv2/opencv.hpp>
#include <queue>

#include <mutex>

using namespace cv;
using namespace std;

void region_growing(Mat& img, Point seed, Mat& mask, int threshold);
#endif // REGION_GROWING_H
