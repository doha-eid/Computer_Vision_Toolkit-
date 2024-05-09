#include <QMainWindow>
//#include <vector>
//#include <QtCharts>

//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>
//#include "src/Helpers/image.h"

#ifndef MAINWINDOW_H
#define MAINWINDOW_H



QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:

  void on_uploadButton_clicked();
  void on_sift1Button_clicked();
  void on_sift2Button_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
