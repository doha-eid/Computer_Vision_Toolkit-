/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.4.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *output_image;
    QPushButton *kmean;
    QLabel *uploaded_image;
    QLabel *upload_label;
    QLabel *filter_label;
    QPushButton *optimal_thresh;
    QPushButton *otsu;
    QPushButton *spect_thresh;
    QPushButton *spect_local_thresh;
    QPushButton *regiongrowing;
    QPushButton *agglo;
    QPushButton *meanshift;
    QPushButton *uploadimage;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1520, 990);
        MainWindow->setStyleSheet(QString::fromUtf8("QMainWindow{\n"
"  background-color: rgb(238, 238, 238);\n"
"}"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        centralwidget->setStyleSheet(QString::fromUtf8("QWidget{\n"
"  background-color: rgb(238, 238, 238);\n"
"}"));
        output_image = new QLabel(centralwidget);
        output_image->setObjectName("output_image");
        output_image->setGeometry(QRect(990, 190, 460, 420));
        output_image->setStyleSheet(QString::fromUtf8("border-left: 2px solid rgb(0, 0, 127);\n"
""));
        kmean = new QPushButton(centralwidget);
        kmean->setObjectName("kmean");
        kmean->setGeometry(QRect(60, 410, 251, 41));
        kmean->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        uploaded_image = new QLabel(centralwidget);
        uploaded_image->setObjectName("uploaded_image");
        uploaded_image->setGeometry(QRect(440, 190, 460, 420));
        uploaded_image->setStyleSheet(QString::fromUtf8("border-left: 2px solid rgb(0, 0, 127);\n"
""));
        upload_label = new QLabel(centralwidget);
        upload_label->setObjectName("upload_label");
        upload_label->setGeometry(QRect(530, 60, 271, 51));
        QFont font;
        font.setFamilies({QString::fromUtf8("Times New Roman")});
        font.setPointSize(21);
        upload_label->setFont(font);
        upload_label->setStyleSheet(QString::fromUtf8("border-bottom: 2px solid rgb(0, 0, 127);\n"
"\n"
""));
        upload_label->setAlignment(Qt::AlignCenter);
        filter_label = new QLabel(centralwidget);
        filter_label->setObjectName("filter_label");
        filter_label->setGeometry(QRect(1090, 60, 271, 51));
        filter_label->setFont(font);
        filter_label->setStyleSheet(QString::fromUtf8("border-bottom: 2px solid rgb(0, 0, 127);\n"
"\n"
""));
        filter_label->setAlignment(Qt::AlignCenter);
        optimal_thresh = new QPushButton(centralwidget);
        optimal_thresh->setObjectName("optimal_thresh");
        optimal_thresh->setGeometry(QRect(60, 170, 251, 41));
        optimal_thresh->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"QPushButton::pressed {\n"
" background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        otsu = new QPushButton(centralwidget);
        otsu->setObjectName("otsu");
        otsu->setGeometry(QRect(60, 230, 251, 41));
        otsu->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        spect_thresh = new QPushButton(centralwidget);
        spect_thresh->setObjectName("spect_thresh");
        spect_thresh->setGeometry(QRect(60, 290, 251, 41));
        spect_thresh->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        spect_local_thresh = new QPushButton(centralwidget);
        spect_local_thresh->setObjectName("spect_local_thresh");
        spect_local_thresh->setGeometry(QRect(60, 350, 251, 41));
        spect_local_thresh->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        regiongrowing = new QPushButton(centralwidget);
        regiongrowing->setObjectName("regiongrowing");
        regiongrowing->setGeometry(QRect(60, 470, 251, 41));
        regiongrowing->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        agglo = new QPushButton(centralwidget);
        agglo->setObjectName("agglo");
        agglo->setGeometry(QRect(60, 530, 251, 41));
        agglo->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        meanshift = new QPushButton(centralwidget);
        meanshift->setObjectName("meanshift");
        meanshift->setGeometry(QRect(60, 590, 251, 41));
        meanshift->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        uploadimage = new QPushButton(centralwidget);
        uploadimage->setObjectName("uploadimage");
        uploadimage->setGeometry(QRect(50, 40, 251, 41));
        uploadimage->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	border-radius: 20px;\n"
"    background-color: rgb(0, 0, 90);\n"
"    border: none;\n"
"    color: white;\n"
"    font-size: 20px;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"background-color: white;\n"
"color: rgb(0, 0, 90);\n"
"}"));
        MainWindow->setCentralWidget(centralwidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        output_image->setText(QString());
        kmean->setText(QCoreApplication::translate("MainWindow", "K-means", nullptr));
        uploaded_image->setText(QString());
        upload_label->setText(QCoreApplication::translate("MainWindow", "Uploaded Image", nullptr));
        filter_label->setText(QCoreApplication::translate("MainWindow", "Output Image", nullptr));
        optimal_thresh->setText(QCoreApplication::translate("MainWindow", "Optimal Thresholding", nullptr));
        otsu->setText(QCoreApplication::translate("MainWindow", "Otsu", nullptr));
        spect_thresh->setText(QCoreApplication::translate("MainWindow", "Spectral Thresholding", nullptr));
        spect_local_thresh->setText(QCoreApplication::translate("MainWindow", "Spectral Local Thresholding", nullptr));
        regiongrowing->setText(QCoreApplication::translate("MainWindow", "Region Growing", nullptr));
        agglo->setText(QCoreApplication::translate("MainWindow", "Agglomerative", nullptr));
        meanshift->setText(QCoreApplication::translate("MainWindow", "Mean Shift", nullptr));
        uploadimage->setText(QCoreApplication::translate("MainWindow", "Upload Image", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
