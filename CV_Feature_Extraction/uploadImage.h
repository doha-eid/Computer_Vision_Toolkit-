#ifndef UPLOADIMAGE_H
#define UPLOADIMAGE_H

#include <QByteArray>
#include <QPixmap>
#include <QFileDialog>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class uploadImage
{
public:
  uploadImage();
  ~uploadImage();

  void setFirstImage(Mat image);
  void setSecondImage(Mat image);
  Mat getFirstImage();
  Mat getSecondImage();
  static QPixmap convertMatToPixmap(Mat imageMat);
  static Mat readImage_Mat(QString filePath);
  static QString readImage_Path();

private:
  Mat first;
  Mat second;
};
#endif // UPLOADIMAGE_H
