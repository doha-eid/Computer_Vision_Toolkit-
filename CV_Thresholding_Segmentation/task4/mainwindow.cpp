#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "uploadImage.h"
#include"Thresholding.h"
#include"Region_Growing.h"
#include"Kmeans.h"
#include"agglomerative.h"
#include"meanShift.h"
#include <QMessageBox>
#include <QDebug>

#include <cmath>
#include <vector>

#include <QByteArray>
#include <QPixmap>
#include <QFileDialog>

//#include <QtCharts/QChartView>
//#include <QtCharts/QBarSeries>

using namespace std;
using namespace cv;

uploadImage image = {};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_uploadimage_clicked()
{
    QString path=uploadImage::readImage_Path();
    Mat Image = uploadImage::readImage_Mat(path);
    if (Image.cols != 1 && Image.rows != 1){
        image.setImage(Image);
        QPixmap pixmap = uploadImage::convertMatToPixmap(Image);
        QPixmap scaledpixmap = pixmap.scaled(ui->uploaded_image->size(), Qt::IgnoreAspectRatio);
        ui->uploaded_image->setPixmap(scaledpixmap);
    }
}


void MainWindow::on_optimal_thresh_clicked()
{
    Mat threshold_image_Optimal = optimal_thresholding(image.getImage());
    QPixmap threshold_image_Optimal_pixmap = uploadImage::convertMatToPixmap(threshold_image_Optimal);
    QPixmap threshold_image_Optimal_scaledpixmap = threshold_image_Optimal_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
    ui->output_image->setPixmap(threshold_image_Optimal_scaledpixmap);

}


void MainWindow::on_otsu_clicked()
{
    Mat otsu_image = otsu_thresholding(image.getImage());
    QPixmap otsu_image_pixmap = uploadImage::convertMatToPixmap(otsu_image);
    QPixmap otsu_image_scaledpixmap = otsu_image_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
    ui->output_image->setPixmap(otsu_image_scaledpixmap);
}


void MainWindow::on_spect_thresh_clicked()
{
    Mat threshold_image_spectral = spectral_thresholding(image.getImage());
    QPixmap threshold_image_spectral_pixmap = uploadImage::convertMatToPixmap(threshold_image_spectral);
    QPixmap threshold_image_spectral_scaledpixmap = threshold_image_spectral_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
    ui->output_image->setPixmap(threshold_image_spectral_scaledpixmap);

}


void MainWindow::on_spect_local_thresh_clicked()
{
    Mat threshold_image_spectral_local = spectral_localThresholding(image.getImage());
      QPixmap threshold_image_spectral_local_pixmap = uploadImage::convertMatToPixmap(threshold_image_spectral_local);
      QPixmap threshold_image_spectral_local_scaledpixmap = threshold_image_spectral_local_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
      ui->output_image->setPixmap(threshold_image_spectral_local_scaledpixmap);
}


void MainWindow::on_kmean_clicked()
{
    Mat kmeans_image=K_means(image.getImage());
    QPixmap kmeans_image_pixmap = uploadImage::convertMatToPixmap(kmeans_image);
    QPixmap kmeans_image_scaledpixmap = kmeans_image_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
    ui->output_image->setPixmap(kmeans_image_scaledpixmap);
}


void MainWindow::on_regiongrowing_clicked()
{
    Mat img=image.getImage();
    Mat gray_image;
    cvtColor(img, gray_image, COLOR_BGR2GRAY);
//   int size=img.size();
    // Create a mask with the same size as the image, initialized with zeros
        Mat mask(gray_image.size(), CV_8UC1, Scalar(0));

        // Set the seed point
        Point seed(100, 100);

        // Set the threshold for the difference in intensity between neighboring pixels
        int threshold = 10;

        // Apply the region growing algorithm
        region_growing(gray_image, seed, mask, threshold);
        QPixmap regiongrowing_image_pixmap = uploadImage::convertMatToPixmap(mask);
        QPixmap regiongrowing_image_scaledpixmap = regiongrowing_image_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
        ui->output_image->setPixmap(regiongrowing_image_scaledpixmap);
}


void MainWindow::on_agglo_clicked()
{
    Mat agglomerative_image = Agglomerative::agglomarativeSegmentation(image.getImage(),7);
        QPixmap agglomerative_image_pixmap = uploadImage::convertMatToPixmap(agglomerative_image);
        QPixmap agglomerative_image_scaledpixmap = agglomerative_image_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
        ui->output_image->setPixmap(agglomerative_image_scaledpixmap);
}


void MainWindow::on_meanshift_clicked()
{
    Mat segementedImage = MeanShift::MeanShiftSegmentation(image.getImage(),20,45);
        QPixmap meanshift_pixmap = uploadImage::convertMatToPixmap(segementedImage);
        QPixmap meanshift_scaledpixmap = meanshift_pixmap.scaled(ui->output_image->size(), Qt::IgnoreAspectRatio);
        ui->output_image->setPixmap(meanshift_scaledpixmap);
}


//void MainWindow::on_uploadimage_clicked()
//{

//}

