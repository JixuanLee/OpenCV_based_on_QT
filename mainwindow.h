#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QFileDialog>
#include <QDebug>
#include <QImage>
#include "qcustomplot.h"
#include <QGraphicsItem>
#include <QGraphicsView>
#include <QWheelEvent>
#include <QMessageBox>
#include <qdesktopwidget.h>
#include <qtablewidget.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:


private slots:
    // 模块0：基础模块
    bool isThereImage(bool DoUNeedPrintError=1);
    QImage ImageWithScaled(QImage  qimage);
    void show_image_now(QImage &need_show_image);
    void draw_2D_plot(const QVector<double>& x, const QVector<double>& y,
                      const QString& title,
                      const double xRange[2], const QString& xLabel,
                      const double yRange[2], const QString& yLabel);
    void on_undo_pb_clicked();
    void on_redo_pb_clicked();
    void reset_base();
    cv::Mat My_QImage2Mat(QImage& src_QImage);
    QImage My_Mat2QImage(cv::Mat& src_Mat);
    void polyfit(int n, std::vector <double> x, std::vector <double> y, int poly_n, double a[]);
    void gauss_solve(int n, double A[], double x[], double b[]);

    void on_load_image_pb_clicked();
    void on_label_linkActivated();
    void on_gray_image_pb_clicked();
    void on_gray_draw_pb_clicked(bool isNeedEqualizeHist=false);
    void on_equalizeHist_pb_clicked();

    void on_save_pb_clicked();

    void on_brightness_hs_valueChanged(int value);
    void on_brightness_reset_pb_clicked();

    void on_colorpic_pb_clicked();

    void on_easybinary_pb_clicked();
    void on_otsubinary_pb_clicked();

    void on_contrast_hs_valueChanged(int value);
    void on_contrast_reset_pb_clicked();

    void on_saturate_hs_valueChanged(int value);
    void on_saturate_reset_pb_clicked();

    void on_warmcold_hs_valueChanged(int value);
    void on_warmcold_reset_pb_clicked();

    void getLaplaceKernel(int level, int kernel[3][3], int &sumKernel);
    void on_laplacesharpen_hs_valueChanged(int value);
    void on_laplacesharpen_reset_pb_clicked();

    void on_brightness_hs_sliderReleased();
    void on_contrast_hs_sliderReleased();
    void on_saturate_hs_sliderReleased();
    void on_warmcold_hs_sliderReleased();
    void on_laplacesharpen_hs_sliderReleased();

    void set_table_all_init(std::string item_str, QTableWidget* table);
    void set_table_lock(std::vector<std::vector<int>> unlock_ij, QTableWidget* table);
    void on_filtermode_vsb_valueChanged(int value);
    void on_filter_pb_clicked();
    int  Table_Int_Get(int i, int j, QTableWidget* table, int min=-0x7fffffff, int max=0x7fffffff);
    void FilterMode_1_Mean_Run();
    void FilterMode_2_Median_Run();
    void FilterMode_3_Gaussian_Run();
    void FilterMode_4_Bilateral_Run();

    static void My_OnMouse(int event, int x, int y, int flags, void* userdata);
    void on_image_windows_pb_clicked();

    void on_gradientsharpen_pb_clicked();

    void on_edgeDetect_cbox_currentIndexChanged(int index);
    void on_edgeDetect_pb_clicked();
    void on_edgeDetect_reset_pb_clicked();
    void EdgeDetectMode_1_Roberts(QImage& src, QImage& dst);
    void EdgeDetectMode_2_Sobel(QImage& src, QImage& dst);
    void EdgeDetectMode_3_Laplace(QImage& src, QImage& dst);
    void EdgeDetectMode_4_Prewitt(QImage& src, QImage& dst);
    void EdgeDetectMode_5_Canny(QImage& src, QImage& dst, double thresholdLow, double thresholdHigh, int WeakEdgeThreshold=10);
    void EdgeDetectMode_5_Canny_OpenCV_Office(QImage& src, QImage& dst, double thresholdLow, double thresholdHigh);

    void on_sideWindow_meanFilter_pb_clicked();

    void on_ErodeDilate_vhb_valueChanged(int value);
    void on_erodedilate_pb_clicked();

    void on_warp_affine_pb_3points_clicked();
    void on_warp_affine_pb_rotate_clicked();
    void warpPerspectiveBase(Point2f srcDynamicQuad[4], Point2f dstDynamicQuad[4], Mat src ,Mat& dst);
    void on_warp_perspective_pb_4points_clicked();

    double generateGaussianNoise(double mu, double sigma);
    Mat addGaussianNoise(Mat &srcImag, double mu=2, double sigma=0.8);
    Mat addSaltNoise(const Mat srcImage, int num=3000);
    void on_gaussnoise_dsb1_valueChanged(double arg1);
    void on_gaussnoise_dsb2_valueChanged(double arg1);
    void on_saltnoise_sb_valueChanged(int arg1);
    void on_gaussNoise_pb_clicked();
    void on_saltnoise_pb_clicked();

    void on_monoCalibration_pb_clicked();
    void calRealPoint(std::vector<std::vector<Point3f>>& obj, int boardWidth, int boardHeight, int imgNumber, int squareSize);
    void outputCameraParam(void);
    void on_stereoCalibration_pb_clicked();

    void on_HSmerge_pb_clicked();

    void on_ORB_pb_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat image;
    QSize show_size;
    QImage img_raw;
    int history_num = 10;
    int history_index;
    std::vector<QImage> history_image;

    QImage brightness_image_now;
    QImage brightness_image_init;

    QImage easy_binary_image_now;
    QImage easy_binary_image_init;

    QImage otsu_binary_image;

    QImage contrast_image_now;
    QImage contrast_image_init;

    QImage saturate_image_now;
    QImage saturate_image_init;

    QImage warmcold_image_now;
    QImage warmcold_image_init;

    QImage laplacesharpen_image_now;
    QImage laplacesharpen_image_init;
    QImage laplacesharpen_image_out;

    QImage gradientsharpen_image;

    QImage meanfilter_image;
    QImage medianfilter_image;
    QImage gaussianfilter_image;
    QImage bilateralfilter_image;
    QImage windows_image;

    QImage edgeDetect_image;

    QImage sidewindow_mean_image;

    QImage erodedilate_image_now;
    QImage erodedilate_image_init;

    QImage gaussnoise_image;
    QImage saltnoise_image;

    bool canOtherFunctionWork = false;
    bool isUndoing = false;
    bool isRedoing = false;
    bool forcedPermission = false;

    bool isFirstBrightnessComeIn = true;
    bool isFirstEasybinaryComeIn = true;
    bool isFirstContrastComeIn = true;
    bool isFirstSaturateComeIn = true;
    bool isFirstWarmcoldComeIn = true;
    bool isFirstLaplacesharpenComeIn = true;
    bool isFirstErodeDilateComeIn = true;

    bool isGrayNow = false;
    bool isEasyBinaryNow = false;
    bool isOtsuBinaryNow = false;

    bool isBrightnessReleased = true;
    bool isContrastReleased = true;
    bool isSaturateReleased = true;
    bool isWarmcoldReleased = true;
    bool isLaplacesharpenReleased = true;

    int FilterMode = 0;
    std::vector<std::vector<int>> unlock_ij;

    int EdgeDetectMode = 0;

    int ErodeDilateMode = 0;
    int ErodeDilateKernelSize = -1;

    double GaussNoiseMu = -1;
    double GaussNoiseSigma = -1;
    int SaltNoiseNum = -1;


    enum FILTERMODE
    {
        NOFILTER = 0,
        MEANFILTER,
        MEDIANFILTER,
        GAUSSIANFILTER,
        BILATERALFILTER
    };

    // 定义边缘检测算法的枚举类型
    enum EdgeDetectionAlgorithm {
        DEFAULT = 0,
        Roberts,
        Sobel,
        Laplace,
        Prewitt,
        Canny,
        Canny_OpenCV
    };

    enum ERODEDILATEMODE {
        DEFAULT_EDM = 0,
        ERODE,
        DILATE,
        OPENING,
        CLOSING
    };

};

#endif // MAINWINDOW_H
