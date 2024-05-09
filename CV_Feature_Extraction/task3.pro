QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    harris.cpp \
    image.cpp \
    main.cpp \
    mainwindow.cpp \
    sift.cpp \
    uploadImage.cpp

HEADERS += \
    harris.h \
    image.h \
    mainwindow.h \
    sift.h \
    stb_image.h \
    stb_image_write.h \
    uploadImage.h

FORMS += \
    mainwindow.ui



INCLUDEPATH +=D:\Apps\opencv\release\install\include

LIBS += -LD:\Apps\opencv\release\lib \
    -lopencv_calib3d470 \
    -lopencv_core470 \
    -lopencv_features2d470 \
    -lopencv_flann470 \
    -lopencv_highgui470 \
    -lopencv_imgproc470 \
    -lopencv_imgcodecs470 \
    -lopencv_photo470 \
    -lopencv_stitching470 \
    -lopencv_ts470 \
    -lopencv_video470 \
    -lopencv_videoio470 \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
