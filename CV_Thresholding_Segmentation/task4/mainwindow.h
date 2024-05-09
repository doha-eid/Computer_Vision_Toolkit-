#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

    void on_optimal_thresh_clicked();

    void on_otsu_clicked();

    void on_spect_thresh_clicked();

    void on_spect_local_thresh_clicked();

    void on_kmean_clicked();

    void on_regiongrowing_clicked();

    void on_agglo_clicked();

    void on_meanshift_clicked();

    void on_uploadimage_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
