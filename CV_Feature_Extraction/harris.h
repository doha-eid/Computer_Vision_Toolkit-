#ifndef HARRIS_H
#define HARRIS_H


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QImage>

using namespace cv;
using namespace std;

class Harris
{
public:
  Harris();
  ~Harris();




  static void cornerHarris_self(double k);
  static Mat cornerHarris_demo(Mat src,int, void*);
  static Mat applyHarris(Mat imageMat);
  static vector<Point2f> detectCorners(const Mat &image, double threshold);
  static Mat applyEigen(Mat image);

};
#endif // HARRIS_H
