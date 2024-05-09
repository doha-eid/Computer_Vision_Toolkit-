#ifndef KMEANS_H
#define KMEANS_H

#include <QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include <mutex>

using namespace cv;
using namespace std;



Mat K_means(Mat image);

#endif // KMEANS_H
