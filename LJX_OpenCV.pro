QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
greaterThan (QT_MAJOR_VERSION, 4): QT += widgets printsupport

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    qcustomplot.cpp

HEADERS += \
    mainwindow.h \
    qcustomplot.h

FORMS += \
    mainwindow.ui

#INCLUDEPATH += -L/home/vtie/opencv-4.5.4/build/lib

#LIBS += -L/usr/local/lib \
#        -lopencv_gapi \
#        -lopencv_highgui \
#        -lopencv_ml \
#        -lopencv_objdetect \
#        -lopencv_photo \
#        -lopencv_stitching \
#        -lopencv_video \
#        -lopencv_calib3d \
#        -lopencv_features2d \
#        -lopencv_dnn \
#        -lopencv_flann \
#        -lopencv_videoio \
#        -lopencv_imgcodecs \
#        -lopencv_imgproc \
#        -lopencv_core \
#LIBS += -L /usr/local/lib/libopencv_*

INCLUDEPATH += /usr/local/include

LIBS += -L /usr/local/lib/libopencv_*.so
LIBS += -L /usr/local/lib/libopencv_calib3d.so.4.5


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
