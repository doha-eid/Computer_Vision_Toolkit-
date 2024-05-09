QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    agglomerative.cpp \
    kmeans.cpp \
    main.cpp \
    mainwindow.cpp \
    meanShift.cpp \
    region_growing.cpp \
    thresholding.cpp \
    uploadimage.cpp

HEADERS += \
    agglomerative.h \
    kmeans.h \
    mainwindow.h \
    meanShift.h \
    region_growing.h \
    thresholding.h \
    uploadimage.h

FORMS += \
    mainwindow.ui


INCLUDEPATH += C:\opencv\release\install\include

LIBS += C:\opencv\release\bin\libopencv_core470.dll
LIBS += C:\opencv\release\bin\libopencv_highgui470.dll
LIBS += C:\opencv\release\bin\libopencv_imgcodecs470.dll
LIBS += C:\opencv\release\bin\libopencv_imgproc470.dll
LIBS += C:\opencv\release\bin\libopencv_calib3d470.dll

#LIBS += -LD:\Apps\opencv\release\bin \
#    -lopencv_calib3d470 \
#    -lopencv_core470 \
#    -lopencv_features2d470 \
#    -lopencv_flann470 \
#    -lopencv_highgui470 \
#    -lopencv_imgproc470 \
#    -lopencv_imgcodecs470 \
#    -lopencv_photo470 \
#    -lopencv_stitching470 \
#    -lopencv_ts470 \
#    -lopencv_video470 \
#    -lopencv_videoio470 \


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
