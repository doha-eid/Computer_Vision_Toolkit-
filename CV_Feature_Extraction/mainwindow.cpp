#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "uploadImage.h"
#include "harris.h"
#include <QMessageBox>
#include <QDebug>

#include <iostream>
#include <cmath>
#include <vector>
#include "image.h"
#include "sift.h"

#include <QByteArray>
#include <QPixmap>
#include <QFileDialog>

//#include <QtCharts/QChartView>
//#include <QtCharts/QBarSeries>
using namespace cv;

uploadImage siftImage1 = {};
uploadImage siftImage2 = {};
QString path1;
QString path2;

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

void MainWindow::on_uploadButton_clicked()
{
    QString path=uploadImage::readImage_Path();
    Mat Image = uploadImage::readImage_Mat(path);
    if (Image.cols != 1 && Image.rows != 1){
        QPixmap pixmap = uploadImage::convertMatToPixmap(Image);
        QPixmap scaledpixmap = pixmap.scaled(ui->uploadImage->size(), Qt::IgnoreAspectRatio);
        ui->uploadImage->setPixmap(scaledpixmap);

        Mat HarrisImage= Harris::applyHarris(Image);
        QPixmap harrisPixmap = uploadImage::convertMatToPixmap(HarrisImage);
        QPixmap harrisscaledpixmap = harrisPixmap.scaled(ui->HarrisImage->size(), Qt::IgnoreAspectRatio);
        ui->HarrisImage->setPixmap(harrisscaledpixmap);


        Mat EigenImage= Harris::applyEigen(Image);
        QPixmap eigenPixmap = uploadImage::convertMatToPixmap(EigenImage);
        QPixmap eigenscaledpixmap = eigenPixmap.scaled(ui->EigenImage->size(), Qt::IgnoreAspectRatio);
        ui->EigenImage->setPixmap(eigenscaledpixmap);

        sift::siftFeatures(path);
        Mat resultSift = uploadImage::readImage_Mat("../a03-team-21-1/siftFeatures.png");
        QPixmap siftPixmap = uploadImage::convertMatToPixmap(resultSift);

        QPixmap siftscaledpixmap = siftPixmap.scaled(ui->SiftImage->size(), Qt::IgnoreAspectRatio);
        ui->SiftImage->setPixmap(siftscaledpixmap);
    }
}


void MainWindow::on_sift1Button_clicked()
{
    path1=uploadImage::readImage_Path();
    Mat Image = uploadImage::readImage_Mat(path1);
    if (Image.cols != 1 && Image.rows != 1){
        QPixmap pixmap = uploadImage::convertMatToPixmap(Image);
        QPixmap scaledpixmap1 = pixmap.scaled(ui->uploadsift1->size(), Qt::IgnoreAspectRatio);
        ui->uploadsift1->setPixmap(scaledpixmap1);

}
}




void MainWindow::on_sift2Button_clicked()
{
    path2=uploadImage::readImage_Path();
    Mat Image = uploadImage::readImage_Mat(path2);
    if (Image.cols != 1 && Image.rows != 1){
        QPixmap pixmap = uploadImage::convertMatToPixmap(Image);
        QPixmap scaledpixmap = pixmap.scaled(ui->uploadsift2->size(), Qt::IgnoreAspectRatio);
        ui->uploadsift2->setPixmap(scaledpixmap);

        sift::applySift(path1,path2);

        Mat ssd = uploadImage::readImage_Mat("../a03-team-21-1/ssd.png");
        Mat ncc = uploadImage::readImage_Mat("../a03-team-21-1/ncc.png");
        QPixmap ssdPixmap = uploadImage::convertMatToPixmap(ssd);
        QPixmap nccPixmap = uploadImage::convertMatToPixmap(ncc);


        QPixmap ssdscaledpixmap = ssdPixmap.scaled(ui->ssd->size(), Qt::IgnoreAspectRatio);
        ui->ssd->setPixmap(ssdscaledpixmap);

        QPixmap nssscaledpixmap = nccPixmap.scaled(ui->ncc->size(), Qt::IgnoreAspectRatio);
        ui->ncc->setPixmap(nssscaledpixmap);



}
}




