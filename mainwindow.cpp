#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;

double mouse_scaleFactor = 1.0;// 用于存储图像的缩放比例
double MouseZoomRes = 0.5;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    show_size = ui->label_image->size();

    ui->label_image->setText(tr("请加载图片..."));
    ui->label_image->setAlignment(Qt::AlignCenter);
    ui->label_image->setStyleSheet("font-weight: bold; font-size: 20px");

    ui->easybinary_sb->setRange(0, 255);
    ui->easybinary_sb->clear(); // 保证文本框中无数字
    ui->easybinary_sb->setEnabled(false); // 开局不准改！

    ui->ErodeDilate_sb->setRange(1, 25);
    ui->ErodeDilate_sb->clear(); // 保证文本框中无数字
    ui->ErodeDilate_sb->setEnabled(false); // 开局不准改！

    ui->gaussnoise_dsb1->setRange(0.0, 255.0);
    ui->gaussnoise_dsb1->clear(); // 保证文本框中无数字
    ui->gaussnoise_dsb1->setEnabled(false); // 开局不准改！

    ui->gaussnoise_dsb2->setRange(0.01, 9999.0);
    ui->gaussnoise_dsb2->clear(); // 保证文本框中无数字
    ui->gaussnoise_dsb2->setEnabled(false); // 开局不准改！

    ui->saltnoise_sb->setRange(1, INT_MAX);
    ui->saltnoise_sb->clear(); // 保证文本框中无数字
    ui->saltnoise_sb->setEnabled(false); // 开局不准改！

    ui->brightness_hs->setRange(-50, 49);
    ui->contrast_hs->setRange(-128, 127);
    ui->saturate_hs->setRange(-100, 99);
    ui->warmcold_hs->setRange(-100, 99);
    ui->laplacesharpen_hs->setRange(0, 15);

    ui->filtermode_vsb->setRange(0, 4);
    ui->filtermode_vsb->setValue(0);
    set_table_all_init("/", ui->filter_table);
    ui->filter_lb_mode->setText("缺 省");
    ui->filter_lb_ksize->setText("缺 省");

    ui->ErodeDilate_vhb->setRange(0, 4);
    ui->ErodeDilate_vhb->setValue(0);
    ui->E_D_lb2->setText("缺 省");

}

MainWindow::~MainWindow()
{
    delete ui;
}


// 模块0.0：当前图像存在检测
bool MainWindow::isThereImage(bool DoUNeedPrintError)
{
    const QPixmap *pixmap = ui->label_image->pixmap();

    if (pixmap && !pixmap->isNull())
        canOtherFunctionWork = true;
    else
        canOtherFunctionWork = false;

    if(canOtherFunctionWork)
    {
        ui->easybinary_sb->setEnabled(true); // 有图啦？改吧改吧-.-
        ui->ErodeDilate_sb->setEnabled(true);
        ui->gaussnoise_dsb1->setEnabled(true);
        ui->gaussnoise_dsb2->setEnabled(true);
        ui->saltnoise_sb->setEnabled(true);
        return true;
    }
    else
    {
        if (DoUNeedPrintError)
            QMessageBox::information(this, tr("嘿！"), tr("你还没加载图片呢！"), QMessageBox::Ok | QMessageBox::Yes);
        return false;
    }
}

// 模块0.1：图片大小与label大小相适应
QImage MainWindow::ImageWithScaled(QImage qimage)
{
    QImage image;
    QSize imageSize = qimage.size();

    double dWidthRatio = 1.0*imageSize.width() / show_size.width();
    double dHeightRatio = 1.0*imageSize.height() / show_size.height();
    if (dWidthRatio > dHeightRatio)
        image = qimage.scaledToWidth(show_size.width());
    else
        image = qimage.scaledToHeight(show_size.height());

    return image;
}

// 模块0.2：可视化&保存（基础模块）
void MainWindow::show_image_now(QImage &need_show_image)
{
    need_show_image = ImageWithScaled(need_show_image);
    need_show_image = need_show_image.scaled(show_size,Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_image->setPixmap(QPixmap::fromImage(need_show_image));
    ui->label_image->setAlignment(Qt::AlignCenter);

    static QImage last_need_show_image;

    // 本if原本用于预防重复录入相同图像导致undo redo缓存区出现重复叠加，加上强制允许flag用以hs类滑动条ui程序最后录入图像
    if (need_show_image != last_need_show_image  ||  forcedPermission == true)
    {
        if (forcedPermission == true) // 确保来一次forcedpremission才执行一次，执行后立刻清空
            forcedPermission = false;

        if ( !isUndoing &&
             !isRedoing &&
             isBrightnessReleased &&
             isContrastReleased &&
             isSaturateReleased &&
             isWarmcoldReleased &&
             isLaplacesharpenReleased )
        {
            if (static_cast<int>(history_image.size()) == history_num)
                history_image.erase(history_image.begin());
             history_image.push_back(need_show_image.copy()); // .copy() is really nssy.
             history_index = history_image.size();

        //qDebug()<<"push: history_image["<<history_index-1<<"] = "<<need_show_image;
        }
    }
    last_need_show_image = need_show_image.copy();
}

// 模块0.3：绘制2D图像（基础模块）
void MainWindow::draw_2D_plot(const QVector<double>& x, const QVector<double>& y,
                              const QString& title,
                              const double xRange[2], const QString& xLabel,
                              const double yRange[2], const QString& yLabel)
{
    QCustomPlot *plot = new QCustomPlot();
    QCPGraph *graph = plot->addGraph(plot->xAxis, plot->yAxis);
    graph->setData(x, y);

    plot->plotLayout()->insertRow(0);
    plot->plotLayout()->addElement(0, 0, new QCPTextElement(plot, title, QFont("sans", 12, QFont::Bold)));

    plot->xAxis->setRange(xRange[0], xRange[1]);
    plot->xAxis->setLabel(xLabel);
    plot->yAxis->setRange(yRange[0], yRange[1]);
    plot->yAxis->setLabel(yLabel);

    plot->resize(QSize(800, 600));
    plot->move(QGuiApplication::primaryScreen()->geometry().center() - plot->rect().center());
    plot->show();
}

// 模块0.4：撤销（基础模块）
void MainWindow::on_undo_pb_clicked()
{
    if(!isThereImage())
        return;

    isUndoing = true;
    if (history_index > 1)
    {
        history_index--;
        QImage undo_image = history_image[history_index-1];

        show_image_now(undo_image);
        ui->statusbar->showMessage("剩余撤销次数：" + QString::number(history_index-1));
    }
    else
    {
        ui->statusbar->showMessage("Error3：已到达[撤销]尽头！");
    }
    isUndoing = false;
}

// 模块0.5：重做（基础模块）
void MainWindow::on_redo_pb_clicked()
{
    if(!isThereImage())
        return;

    isRedoing = true;
    if (history_index < static_cast<int>(history_image.size()))
    {
        history_index++;
        QImage redo_image = history_image[history_index-1];

        show_image_now(redo_image);
        ui->statusbar->showMessage("剩余重做次数：" + QString::number(history_image.size()-history_index));
    }
    else
    {
        ui->statusbar->showMessage("Error4：已到达[重做]尽头！");
    }
    isRedoing = false;
}

// 模块0.6：全局复位（基础模块）
void MainWindow::reset_base()
{
    isGrayNow = false;
    isEasyBinaryNow = false;
    isOtsuBinaryNow = false;
    on_contrast_reset_pb_clicked();
    on_brightness_reset_pb_clicked();
    on_saturate_reset_pb_clicked();
    on_warmcold_reset_pb_clicked();
    on_laplacesharpen_reset_pb_clicked();
    isFirstEasybinaryComeIn = true; // is equal to ...reset_pb_clicked()
    isFirstErodeDilateComeIn = true;

    ui->brightness_hs->setValue(0);
    ui->contrast_hs->setValue(0);
    ui->saturate_hs->setValue(0);
    ui->warmcold_hs->setValue(0);
    ui->laplacesharpen_hs->setValue(0);
    ui->otsubinary_lb2->setText(tr("缺 省"));
    ui->easybinary_sb->clear();
    ui->filtermode_vsb->setValue(0);
    ui->filter_lb_mode->setText("缺 省");
    ui->filter_lb_ksize->setText("缺 省");
    ui->edgeDetect_cbox->setCurrentIndex(0);
    ui->ErodeDilate_vhb->setValue(0);
    ui->E_D_lb2->setText("缺 省");
    ui->ErodeDilate_sb->clear();
    ui->gaussnoise_dsb1->clear();
    ui->gaussnoise_dsb2->clear();
    ui->saltnoise_sb->clear();

    isBrightnessReleased = true;
    isContrastReleased = true;
    isSaturateReleased = true;
    isWarmcoldReleased = true;
    isLaplacesharpenReleased = true;
    isRedoing = false;
    isUndoing = false;
    forcedPermission = false;
    ErodeDilateMode = 0;
    ErodeDilateKernelSize = -1;
    GaussNoiseMu = -1;
    GaussNoiseSigma = -1;
    SaltNoiseNum = -1;

}

// 模块0.7：图像格式转换I（基础模块）
cv::Mat MainWindow::My_QImage2Mat(QImage& src_QImage)
{
    Mat dst_Mat;
    switch (src_QImage.format())
    {
        case QImage::Format_RGB32: // 彩32位，无Alpha通道
        case QImage::Format_ARGB32: // 彩32位，有Alpha通道
        case QImage::Format_ARGB32_Premultiplied:
        {
            dst_Mat = Mat(src_QImage.height(), src_QImage.width(), CV_8UC4, (void*)src_QImage.constBits(), src_QImage.bytesPerLine());
            cvtColor(dst_Mat, dst_Mat, COLOR_BGRA2BGR);
            break;
        }
        case QImage::Format_RGB888: // 彩24位，无Alpha通道
        {
            dst_Mat = Mat(src_QImage.height(), src_QImage.width(), CV_8UC3, (void*)src_QImage.constBits(), src_QImage.bytesPerLine());
            cvtColor(dst_Mat, dst_Mat, COLOR_RGB2BGR);
            break;
        }
        case QImage::Format_Indexed8: // 灰8位，无Alpha通道
        {
            dst_Mat = Mat(src_QImage.height(), src_QImage.width(), CV_8UC1, (void*)src_QImage.constBits(), src_QImage.bytesPerLine());
            break;
        }
        case QImage::Format_Mono: // 二值化1位，无Alpha通道
        {
            src_QImage = src_QImage.convertToFormat(QImage::Format_Grayscale8);
            dst_Mat = Mat(src_QImage.height(), src_QImage.width(), CV_8UC1, (void*)src_QImage.constBits(), src_QImage.bytesPerLine());
            break;
        }
        case QImage::Format_Grayscale8: // 灰8位，OTSU指定格式
        {
            dst_Mat = Mat(src_QImage.height(), src_QImage.width(), CV_8UC1, src_QImage.bits(), src_QImage.bytesPerLine());
            break;
        }
        default: // 其他
        {
            qDebug() << "Unsupported QImage format: " << src_QImage.format();
            break;
        }
    }
    return dst_Mat;
}

// 模块0.8：图像格式转换II（基础模块）
QImage MainWindow::My_Mat2QImage(cv::Mat& src_Mat)
{
    QImage dst_QImage;
    switch (src_Mat.type())
    {
        case CV_8UC1: // 灰8位，无Alpha通道
        {
            QVector<QRgb> colorTable;
            for (int i = 0; i < 256; i++)
                colorTable.push_back(qRgb(i, i, i));
            dst_QImage = QImage(src_Mat.data, src_Mat.cols, src_Mat.rows, src_Mat.step, QImage::Format_Grayscale8);
//            dst_QImage = QImage(src_Mat.data, src_Mat.cols, src_Mat.rows, src_Mat.step, QImage::Format_Mono);
            dst_QImage.setColorTable(colorTable);
            break;
        }
        case CV_8UC3: // 彩24位，无Alpha通道
        {
            dst_QImage = QImage(src_Mat.data, src_Mat.cols, src_Mat.rows, src_Mat.step, QImage::Format_RGB888);
            dst_QImage = dst_QImage.rgbSwapped();
            break;
        }
        case CV_8UC4: // 彩32位，有Alpha通道
        {
            dst_QImage = QImage(src_Mat.data, src_Mat.cols, src_Mat.rows, src_Mat.step, QImage::Format_ARGB32);
            break;
        }
        default: // 其他
        {
            qDebug() << "Unsupported Mat type: " << src_Mat.type();
            break;
        }
    }
    return dst_QImage;
}

void MainWindow::polyfit(int n, vector <double> x, vector <double> y, int poly_n, double a[])
{
    int i, j;
    double *tempx, *tempy, *sumxx, *sumxy, *ata;

    tempx = new double[n];
    sumxx = new double[poly_n * 2 + 1];
    tempy = new double[n];
    sumxy = new double[poly_n + 1];
    ata = new double[(poly_n + 1)*(poly_n + 1)];

    for (i = 0; i < n; i++)
    {
        tempx[i] = 1;
        tempy[i] = y[i];
    }
    for (i = 0; i < 2 * poly_n + 1; i++)
        for (sumxx[i] = 0, j = 0; j < n; j++)
        {
            sumxx[i] += tempx[j];
            tempx[j] *= x[j];
        }
    for (i = 0; i < poly_n + 1; i++)
        for (sumxy[i] = 0, j = 0;j<n;j++)
        {
            sumxy[i] += tempy[j];
            tempy[j] *= x[j];
        }
    for (i = 0;i<poly_n + 1;i++)
        for (j = 0;j<poly_n + 1;j++)
            ata[i*(poly_n + 1) + j] = sumxx[i + j];
    gauss_solve(poly_n + 1, ata, a, sumxy);

    delete[] tempx;
    tempx = NULL;
    delete[] sumxx;
    sumxx = NULL;
    delete[] tempy;
    tempy = NULL;
    delete[] sumxy;
    sumxy = NULL;
    delete[] ata;
    ata = NULL;
}


void MainWindow::gauss_solve(int n, double A[], double x[], double b[])
{
    int i, j, k, r;
    double max;
    for (k = 0;k<n - 1;k++)
    {
        max = fabs(A[k*n + k]); /*find maxmum*/
        r = k;
        for (i = k + 1;i<n - 1;i++)
            if (max<fabs(A[i*n + i]))
            {
                max = fabs(A[i*n + i]);
                r = i;
            }
        if (r != k)
            for (i = 0;i<n;i++) /*change array:A[k]&A[r] */
            {
                max = A[k*n + i];
                A[k*n + i] = A[r*n + i];
                A[r*n + i] = max;
            }
        max = b[k]; /*change array:b[k]&b[r] */
        b[k] = b[r];
        b[r] = max;
        for (i = k + 1;i<n;i++)
        {
            for (j = k + 1;j<n;j++)
                A[i*n + j] -= A[i*n + k] * A[k*n + j] / A[k*n + k];
            b[i] -= A[i*n + k] * b[k] / A[k*n + k];
        }
    }

    for (i = n - 1;i >= 0;x[i] /= A[i*n + i], i--)
        for (j = i + 1, x[i] = b[i];j<n;j++)
            x[i] -= A[i*n + j] * x[j];
}


// 模块1：加载图片
void MainWindow::on_load_image_pb_clicked()
{
    QString load_filename = QFileDialog::getOpenFileName(this,tr("打开图片"),".",tr("Image File(*.png *.jpg *.bmp"));
    qDebug()<<"Load Filename: "<<load_filename;
    image = cv::imread(load_filename.toLatin1().data());

    img_raw = My_Mat2QImage(image);

    on_label_linkActivated();
    if (isThereImage()) //为了防止点击加载图片，但是没有实际加载进来，触发3连报警
        reset_base();
}

// 模块2：展示图片
void MainWindow::on_label_linkActivated()
{
    QImage ratio_img = img_raw;
    show_image_now(ratio_img);
}

// 模块3.1：灰度化
void MainWindow::on_gray_image_pb_clicked()
{
    if(!isThereImage())
        return;

    QImage grayimg;
    //    grayimg = img_raw.convertToFormat(QImage::Format_ARGB32);
    grayimg = ui->label_image->pixmap()->toImage().convertToFormat(QImage::Format_ARGB32).scaled(img_raw.size());
    QColor rgb_color;
    for(int y = 0; y < grayimg.height(); y++)
    {
        for(int x =0; x < grayimg.width(); x++)
        {
            rgb_color = QColor(img_raw.pixel(x,y));
            // 模式1：平均灰度
//            int rgb_avg = (rgb_color.red()+rgb_color.green()+rgb_color.blue())/3;
            // 模式2：权重灰度
            int rgb_avg = 0.3*rgb_color.red()+0.59*rgb_color.green()+0.11*rgb_color.blue();
            grayimg.setPixel(x, y, qRgb(rgb_avg, rgb_avg, rgb_avg));
        }
    }

    show_image_now(grayimg);
    isGrayNow = true;
}

// 模块3.2：灰度直方图绘制
void MainWindow::on_gray_draw_pb_clicked(bool isNeedEqualizeHist)
{
    if(!isThereImage())
        return;

//    if (isNeedEqualizeHist == false)
//        on_gray_image_pb_clicked();

    QImage gray_draw_img = ui->label_image->pixmap()->toImage();

    QVector<double> graynum(256, 0);
    int graynow;
    double allnum = gray_draw_img.width() * gray_draw_img.height();

    for (int i=0; i<gray_draw_img.width(); i++){
        for (int j=0; j<gray_draw_img.height(); j++){
            if ( (QColor(gray_draw_img.pixel(i,j)).red() != QColor(gray_draw_img.pixel(i,j)).blue()) ||
                 (QColor(gray_draw_img.pixel(i,j)).red() != QColor(gray_draw_img.pixel(i,j)).green()) ||
                 (QColor(gray_draw_img.pixel(i,j)).green() != QColor(gray_draw_img.pixel(i,j)).blue())  )
            {
                QMessageBox::information(this, tr("嘿！"), tr("当前尚未灰度化，无法进行灰度直方图绘制或者灰度均衡化"), QMessageBox::Ok | QMessageBox::Yes);
                goto end_of_loop;
            }
            graynow = QColor(gray_draw_img.pixel(i,j)).red();
            graynum[graynow] = graynum[graynow] + 100/allnum;
        }
    }

    if (isNeedEqualizeHist == false)
    {
        QVector<double> x(256);
        for (int k=0; k<256; k++)
            x[k] = k;

        double ymax= 0;
        for (int k=0; k<256; k++)
        {
            if (graynum[k] > ymax)
                ymax = graynum[k];
        }

        double xRange[2] = {0, 255};
        double yRange[2] = {0, ymax * 1.20};

        draw_2D_plot(x, graynum, "灰度直方图", xRange, "灰度值", yRange, "概率分布  %");
    }
    else
    {
        QVector<double> equalize_pixel(256, 0);
        int pixelIndex;
        int newRGB;

        for (int index=0; index<256; index++){
            for(int m=0; m<=index; m++){
                equalize_pixel[index] += graynum[m];
            }
            equalize_pixel[index] *= 2.55;
        }

        for (int w=0; w<gray_draw_img.width(); w++){
            for (int h=0; h<gray_draw_img.height(); h++){
                pixelIndex = QColor(gray_draw_img.pixel(w,h)).red();
                newRGB = int( equalize_pixel[pixelIndex] );
                newRGB = qBound(0, newRGB, 255);
                gray_draw_img.setPixel(w, h, qRgb(newRGB, newRGB, newRGB));
            }
        }
        show_image_now(gray_draw_img);
    }

    end_of_loop:
     return;
}

// 模块3.3：灰度直方图均衡化
void MainWindow::on_equalizeHist_pb_clicked()
{
    on_gray_draw_pb_clicked(true);
}

// 模块4：保存图片
void MainWindow::on_save_pb_clicked()
{
    if(!isThereImage())
        return;
    if(ui->label_image->pixmap()!=nullptr)
    {
        QString save_filename = QFileDialog::getSaveFileName(this, tr("保存图片(格式后缀需指定)"), "/home/vtie/Desktop",tr("*.png;; *.jpg;; *.bmp;; *.tif;; *.GIF"));
        if (save_filename.isEmpty())
            return;
        else
        {
            if (!(ui->label_image->pixmap()->toImage().save(save_filename)))
            {
                QMessageBox::information(this, tr("Error！"), tr("图片保存失败！"));
                return;
            }
            ui->statusbar->showMessage("图片保存成功！");
            QMessageBox::information(this, tr("Wow！"), tr("图片保存成功！"));
        }
    }
    else
       QMessageBox::warning(nullptr, "提示", "请先打开图片！", QMessageBox::Yes |  QMessageBox::Yes);
}


// 模块5.1：亮度调节
void MainWindow::on_brightness_hs_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->brightness_hs->setValue(0);
            return;
        }
    }

    isBrightnessReleased = false;

    if (isFirstBrightnessComeIn)
    {
        isFirstBrightnessComeIn = false;
        brightness_image_init = ui->label_image->pixmap()->toImage();
    }
    brightness_image_now = brightness_image_init;

    int red, green, blue;
    int pixels = brightness_image_now.width() * brightness_image_now.height();
    unsigned int *data = (unsigned int *)brightness_image_now.bits();

    for (int i = 0; i < pixels; ++i)
    {
        red= qRed(data[i])+ value;
        red = (red < 0x00) ? 0x00 : (red > 0xff) ? 0xff : red;
        green= qGreen(data[i]) + value;
        green = (green < 0x00) ? 0x00 : (green > 0xff) ? 0xff : green;
        blue= qBlue(data[i]) + value;
        blue =  (blue  < 0x00) ? 0x00 : (blue  > 0xff) ? 0xff : blue ;
        data[i] = qRgba(red, green, blue, qAlpha(data[i]));
    }

    show_image_now(brightness_image_now);
    ui->brightness_lb->setText(QString::number(value));
}


// 模块5.2： 亮度调节初始化
void MainWindow::on_brightness_reset_pb_clicked()
{
    if(!isThereImage())
        return;
    isFirstBrightnessComeIn = true;
}


// 模块5.3：亮度调节结束保存（滑动条专属）
void MainWindow::on_brightness_hs_sliderReleased()
{
    isBrightnessReleased = true;
    forcedPermission = true;
    show_image_now(brightness_image_now);// 更新最后一次图像，并保存在undo缓存区
}

// 模块6： 全局复位(彩色化)
void MainWindow::on_colorpic_pb_clicked()
{
    if(!isThereImage())
        return;
    QImage reset_image = img_raw;
    show_image_now(reset_image);

    reset_base();
}

// 模块7： 简单二值化
void MainWindow::on_easybinary_pb_clicked()
{
    if(!isThereImage())
        return;
    if (isOtsuBinaryNow)
        ui->statusbar->showMessage("Error 1: 已经OTSU二值化，请先全局复位！");
    else
    {
        int value = ui->easybinary_sb->value();

        if (isGrayNow == false)
            on_gray_image_pb_clicked();

        if (isFirstEasybinaryComeIn)
        {
            isFirstEasybinaryComeIn = false;
            easy_binary_image_init = ui->label_image->pixmap()->toImage();
        }
        easy_binary_image_now = easy_binary_image_init;

        int gray_value;
        QColor easybinary_color;
        for (int i=0; i<easy_binary_image_now.height(); ++i)
        {
            for (int j=0; j<easy_binary_image_now.width(); ++j)
            {
                easybinary_color = QColor(easy_binary_image_now.pixel(j,i));
                gray_value = easybinary_color.red();
                if (gray_value < value)
                    gray_value = 0;
                else
                    gray_value = 255;
                easy_binary_image_now.setPixel(j,i, qRgb(gray_value, gray_value, gray_value));
            }
        }

        show_image_now(easy_binary_image_now);
        isEasyBinaryNow = true;
        ui->statusbar->showMessage(""); // （若有则）消除报错信息
    }
}

// 模块8： OTSU二值化
void MainWindow::on_otsubinary_pb_clicked()
{
    if(!isThereImage())
        return;

    if (isEasyBinaryNow)
        ui->statusbar->showMessage("Error 2: 已经Easy二值化，请先全局复位！");
    else
    {
        if (isGrayNow == false)
            on_gray_image_pb_clicked();
        otsu_binary_image = ui->label_image->pixmap()->toImage();

        // 首先将otsu_binary_image转换为灰度格式, 然后创建一个Mat对象，用otsu_binary_image的数据和属性来初始化
        otsu_binary_image = otsu_binary_image.convertToFormat(QImage::Format_Grayscale8);
        Mat otsu_binary_mat_image_in = My_QImage2Mat(otsu_binary_image);
        Mat otsu_binary_mat_image_out;

        // 一参为输入的Mat格式图像，二参为输出的otsu最优二值化后的Mat格式图像, 返回值为otsu最优二值化阈值
        double otsu_threshold = cv::threshold(otsu_binary_mat_image_in, otsu_binary_mat_image_out, 0, 255, THRESH_BINARY + THRESH_OTSU);

        // 将得到的二值化结果直接转为QImage格式并show
        otsu_binary_image = My_Mat2QImage(otsu_binary_mat_image_out);

        show_image_now(otsu_binary_image);

        ui->otsubinary_lb2->setText(QString::number(otsu_threshold));
        isOtsuBinaryNow = true;

        ui->statusbar->showMessage(" "); // （若有则）消除报错信息
    }
}

// 模块9.1： 对比度调节
void MainWindow::on_contrast_hs_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->contrast_hs->setValue(0);
            return;
        }
    }

    isContrastReleased = false;

    if (isFirstContrastComeIn)
    {
        isFirstContrastComeIn = false;
        contrast_image_init = ui->label_image->pixmap()->toImage();
    }
    contrast_image_now = contrast_image_init;

    int pixels = contrast_image_now.width() * contrast_image_now.height();
    unsigned int *data = (unsigned int *)contrast_image_now.bits();
    int red, green, blue, nowred, nowgreen, nowblue;

    float nonlinear_param = 256.0/(256-value) - 1;

    for (int i=0; i<pixels; ++i)
    {
        nowred = qRed(data[i]);
        nowblue = qBlue(data[i]);
        nowgreen = qGreen(data[i]);

        red = nowred + (nowred - 127) * nonlinear_param;
        blue = nowblue + (nowblue - 127) * nonlinear_param;
        green = nowgreen + (nowgreen - 127) * nonlinear_param;

        red = (red < 0x00) ? 0x00 : (red > 0xff) ? 0xff : red;
        blue = (blue < 0x00) ? 0x00 : (blue > 0xff) ? 0xff : blue;
        green = (green < 0x00) ? 0x00 : (green > 0xff) ? 0xff : green;

        data[i]=qRgba(red, green, blue,qAlpha(data[i]));
    }

    show_image_now(contrast_image_now);
    ui->contrast_lb->setText(QString::number(value));
}

// 模块9.2： 对比度调节初始化
void MainWindow::on_contrast_reset_pb_clicked()
{
    if(!isThereImage())
        return;
    isFirstContrastComeIn = true;
}

// 模块9.3：对比度调节结束保存（滑动条专属）
void MainWindow::on_contrast_hs_sliderReleased()
{
    isContrastReleased = true;
    forcedPermission = true;
    show_image_now(contrast_image_now);// 更新最后一次图像，并保存在undo缓存区
}


// 模块10.1： 饱和度调节
void MainWindow::on_saturate_hs_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->saturate_hs->setValue(0);
            return;
        }
    }

    isSaturateReleased = false;

    if (isFirstSaturateComeIn)
    {
        isFirstSaturateComeIn = false;
        saturate_image_init = ui->label_image->pixmap()->toImage();
    }
    saturate_image_now = saturate_image_init;

    int pixels = saturate_image_now.width() * saturate_image_now.height();
    unsigned int *data = (unsigned int *)saturate_image_now.bits();
    int red, green, blue, nowred, nowgreen, nowblue;
    float delta = 0;
    float min_color, max_color;
    float Luminance, Saturate;
    float increment_param = value/100.0;
    float saturate_param;

    for (int i=0; i<pixels; ++i)
    {
        nowred = qRed(data[i]);
        nowgreen = qGreen(data[i]);
        nowblue = qBlue(data[i]);
        min_color = std::min(std::min(nowred, nowblue), nowgreen);
        max_color = std::max(std::max(nowred, nowblue), nowgreen);

        delta = (max_color - min_color)/255.0;
        if (delta == 0)
            continue;
        Luminance = (max_color + min_color)/2/255.0;
        Saturate = std::max(delta/Luminance, delta/(1-Luminance)) / 2;

        if (increment_param >= 0)
        {
            if ((increment_param + Saturate) >=1)
                saturate_param = Saturate;
            else
                saturate_param = 1 - increment_param;

            saturate_param = 1.0/saturate_param - 1;

            red = nowred + (nowred - Luminance * 255.0) * saturate_param;
            green = nowgreen + (nowgreen - Luminance * 255.0) * saturate_param;
            blue = nowblue + (nowblue - Luminance * 255.0) * saturate_param;

            red = (red < 0x00) ? 0x00 : (red > 0xff) ? 0xff : red;
            green = (green < 0x00) ? 0x00 : (green > 0xff) ? 0xff : green;
            blue = (blue < 0x00) ? 0x00 : (blue > 0xff) ? 0xff : blue;
        }
        else
        {
            saturate_param = increment_param;

            red = Luminance*255.0 + (nowred - Luminance * 255) * (1 + saturate_param);
            green = Luminance*255.0 + (nowgreen - Luminance * 255) * (1 + saturate_param);
            blue = Luminance*255.0 + (nowblue - Luminance * 255) * (1 + saturate_param);

            red = (red < 0x00) ? 0x00 : (red > 0xff) ? 0xff : red;
            green = (green < 0x00) ? 0x00 : (green > 0xff) ? 0xff : green;
            blue = (blue < 0x00) ? 0x00 : (blue > 0xff) ? 0xff : blue;
        }

        data[i] = qRgba(red, green, blue, qAlpha(data[i]));
    }

    show_image_now(saturate_image_now);
    ui->saturate_lb->setText(QString::number(value));
}

// 模块10.2： 饱和度调节初始化
void MainWindow::on_saturate_reset_pb_clicked()
{
    if(!isThereImage())
        return;
    isFirstSaturateComeIn = true;
}

// 模块10.3：饱和度调节结束保存（滑动条专属）
void MainWindow::on_saturate_hs_sliderReleased()
{
    isSaturateReleased = true;
    forcedPermission = true;
    show_image_now(saturate_image_now);// 更新最后一次图像，并保存在undo缓存区
}


// 模块11.0.a： 图像滤波表格初始化设置
void MainWindow::set_table_all_init(std::string item_str, QTableWidget* table)
{
    QTableWidgetItem* table_items = new QTableWidgetItem(item_str.data());

    for (int i=0; i<table->rowCount(); i++){
        for (int j=0; j<table->columnCount(); j++){
            QTableWidgetItem* item_now = table_items->clone();
            table->setItem(i, j, item_now);
        }
    }
}

// 模块11.0.b： 设置图像滤波表格无关元素为“不可编辑”(需要指定豁免元素坐标)
void MainWindow::set_table_lock(std::vector<std::vector<int>> unlock_ij, QTableWidget* table)
{
    int rows = unlock_ij.size();

    // 将表格的所有单元格都设置为不可编辑
    for (int i=0; i < table->rowCount(); i++)
        for (int j=0; j < table->columnCount(); j++)
            table->item(i, j)->setFlags(table->item(i, j)->flags() & ~Qt::ItemIsEditable);

    // 根据unlock_ij中的索引来将对应的单元格设置为可编辑
    for (int unlock_i=0; unlock_i < rows; unlock_i++){
        int i = unlock_ij[unlock_i][0]; // 获取行索引
        int j = unlock_ij[unlock_i][1]; // 获取列索引
        table->item(i, j)->setFlags(table->item(i, j)->flags() | Qt::ItemIsEditable); // 设置单元格的可编辑标志
    }
}

// 模块11.0.c： 图像滤波表格Int提取
int MainWindow::Table_Int_Get(int i, int j, QTableWidget* table, int min, int max)
{
    QTableWidgetItem* item = table->item(i, j);
    item->text() = item->text().remove(tr(" "));

    bool int_ok;
    int item_int = item->text().toInt(&int_ok);
    if (int_ok)
    {
        if (item_int>=min && item_int<=max)
            return item_int;
        else
        {
            QMessageBox::warning(nullptr, "Error!", "请在滤波表格输入符合范围要求的整数！", QMessageBox::Yes |  QMessageBox::Ok);
            return -1;
        }
    }
    else
    {
        QMessageBox::warning(nullptr, "Error!", "请在滤波表格输入整数！", QMessageBox::Yes |  QMessageBox::Ok);
        return -1;
    }
}

// 模块11.1.a： 均值滤波计算
// ----适用于去除随机噪声，但会使图像边缘模糊----
void MainWindow::FilterMode_1_Mean_Run()
{
    int level, ksize;
    if ( ( level = Table_Int_Get(0,0,ui->filter_table,1,5) ) == -1)
        return;
    ksize = level * 2 - 1;

    meanfilter_image = ui->label_image->pixmap()->toImage();
    cv::Mat meanfilter_mat = My_QImage2Mat(meanfilter_image);
    blur(meanfilter_mat, meanfilter_mat, cv::Size(ksize,ksize), cv::Point(-1,-1));
    meanfilter_image = My_Mat2QImage(meanfilter_mat);

    show_image_now(meanfilter_image);
    ui->filter_lb_ksize->setText(QString::number(ksize) + "*" + QString::number(ksize));
}

// 模块11.1.b： 中值滤波计算
// ----适用于去除椒盐噪声，且保留边缘信息相对较好----
void MainWindow::FilterMode_2_Median_Run()
{
    int level, ksize;
    if ( ( level = Table_Int_Get(1,0,ui->filter_table,1,5) ) == -1)
        return;
    ksize = level * 2 - 1;

    medianfilter_image = ui->label_image->pixmap()->toImage();
    cv::Mat medianfilter_mat = My_QImage2Mat(medianfilter_image);
    medianBlur(medianfilter_mat, medianfilter_mat, ksize);
    medianfilter_image = My_Mat2QImage(medianfilter_mat);

    show_image_now(medianfilter_image);
    ui->filter_lb_ksize->setText(QString::number(ksize) + "*" + QString::number(ksize));
}

// 模块11.1.c： 高斯滤波计算
// ----适用于去除高斯噪声，且保留边缘信息相对较好----
void MainWindow::FilterMode_3_Gaussian_Run()
{
    int level, ksize, sigma;
    if ( ( level = Table_Int_Get(2,0,ui->filter_table,1,5) ) == -1  ||
         ( sigma = Table_Int_Get(2,1,ui->filter_table,1,100) ) == -1    )
        return;
    ksize = level * 2 - 1;

    gaussianfilter_image = ui->label_image->pixmap()->toImage();
    cv::Mat gaussianfilter_mat = My_QImage2Mat(gaussianfilter_image);
    GaussianBlur(gaussianfilter_mat, gaussianfilter_mat, cv::Size(ksize,ksize), sigma);

    // just for zuoye
//    double zuoye_init[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
//    cv::Mat zuoye_mat = Mat(3, 3, CV_64F, zuoye_init);
//    cv::filter2D(gaussianfilter_mat, gaussianfilter_mat, -1, zuoye_mat);

    gaussianfilter_image = My_Mat2QImage(gaussianfilter_mat);

    show_image_now(gaussianfilter_image);
    ui->filter_lb_ksize->setText(QString::number(ksize) + "*" + QString::number(ksize));
}

// 模块11.1.d： 双边滤波计算
// ----考虑空间、色彩两因素，且保留边缘信息非常好，但计算量较大----
void MainWindow::FilterMode_4_Bilateral_Run()
{
    int level, ksize, sigma_color, sigma_space;
    if ( ( level = Table_Int_Get(3,0,ui->filter_table,1,5) ) == -1  ||
         ( sigma_color = Table_Int_Get(3,1,ui->filter_table,1,100) ) == -1 ||
         ( sigma_space = Table_Int_Get(3,2,ui->filter_table,1,100) ) == -1    )
        return;
    ksize = level * 2 - 1;

    bilateralfilter_image = ui->label_image->pixmap()->toImage();
    cv::Mat bilateralfilter_mat = My_QImage2Mat(bilateralfilter_image);
    bilateralFilter(bilateralfilter_mat, bilateralfilter_mat, ksize, sigma_color, sigma_space);
    bilateralfilter_image = My_Mat2QImage(bilateralfilter_mat);

    show_image_now(bilateralfilter_image);
    ui->filter_lb_ksize->setText(QString::number(ksize) + "*" + QString::number(ksize));
}

// 模块11.2： 图像滤波模式滑选
void MainWindow::on_filtermode_vsb_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->filtermode_vsb->setValue(0);
            return;
        }
    }

    FilterMode = value;

    switch (FilterMode)
    {
        case NOFILTER:
        {
            set_table_all_init("/", ui->filter_table);
            ui->statusbar->showMessage(tr("当前[尚未选择]滤波模式！"));
            ui->filter_lb_mode->setText(tr("缺 省"));
            break;
        }
        case MEANFILTER:
        {
            set_table_all_init("/", ui->filter_table);
            ui->filter_table->setItem(0, 0, new QTableWidgetItem(" "));

            unlock_ij = {{0, 0}};
            set_table_lock(unlock_ij, ui->filter_table);

            ui->statusbar->showMessage(tr("当前为[均值滤波]，适用于去除随机噪声，但会使图像边缘模糊--请在表格空白处输入整数, 其中滤波程度范围1~5"));
            ui->filter_lb_mode->setText(tr("均值"));
            break;
        }
        case MEDIANFILTER:
        {
            set_table_all_init("/", ui->filter_table);
            ui->filter_table->setItem(1, 0, new QTableWidgetItem(" "));

            unlock_ij = {{1, 0}};
            set_table_lock(unlock_ij, ui->filter_table);

            ui->statusbar->showMessage(tr("当前为[中值滤波]，适用于去除椒盐噪声，且保留边缘信息相对较好--请在表格空白处输入整数, 其中滤波程度范围1~5"));
            ui->filter_lb_mode->setText(tr("中值"));
            break;
        }
        case GAUSSIANFILTER:
        {
            set_table_all_init("/", ui->filter_table);
            ui->filter_table->setItem(2, 0, new QTableWidgetItem(" "));
            ui->filter_table->setItem(2, 1, new QTableWidgetItem(" "));

            unlock_ij = {{2, 0}, {2, 1}};
            set_table_lock(unlock_ij, ui->filter_table);

            ui->statusbar->showMessage(tr("当前为[高斯滤波]，适用于去除高斯噪声，且保留边缘信息相对较好--请在表格空白处输入整数, 其中滤波程度范围1~5, σ范围1-100"));
            ui->filter_lb_mode->setText(tr("高斯"));
            break;
        }
        case BILATERALFILTER:
        {
            set_table_all_init("/", ui->filter_table);
            ui->filter_table->setItem(3, 0, new QTableWidgetItem(" "));
            ui->filter_table->setItem(3, 1, new QTableWidgetItem(" "));
            ui->filter_table->setItem(3, 2, new QTableWidgetItem(" "));

            unlock_ij= {{3, 0}, {3, 1}, {3, 2}};
            set_table_lock(unlock_ij, ui->filter_table);

            ui->statusbar->showMessage(tr("当前为[双边滤波]，考虑空间、色彩，保留边缘信息很好，但计算量大--请在表格空白处输入整数, 其中滤波程度范围1~5, σ范围均为1-100"));
            ui->filter_lb_mode->setText(tr("双边"));
            break;
        }
        default:
        {
            QMessageBox::information(this, tr("诶！"), tr("图像滤波模式选择错误！"), QMessageBox::Ok | QMessageBox::Yes);
            break;
        }
    }
}

// 模块11.3： 图像滤波(按钮)整体功能实现
void MainWindow::on_filter_pb_clicked()
{
    if(!isThereImage())
        return;
    if (FilterMode != NOFILTER && FilterMode != MEANFILTER && FilterMode!=MEDIANFILTER && FilterMode!=GAUSSIANFILTER && FilterMode!=BILATERALFILTER)
    {
        QMessageBox::information(this, tr("诶！"), tr("图像滤波模式选择错误！"), QMessageBox::Ok | QMessageBox::Yes);
        return;
    }

    switch (FilterMode)
    {
        case NOFILTER:
        {
            QMessageBox::information(this, tr("诶！"), tr("请先选择滤波模式！"), QMessageBox::Ok | QMessageBox::Yes);
            break;
        }
        case MEANFILTER:
        {
            FilterMode_1_Mean_Run();
            break;
        }
        case MEDIANFILTER:
        {
            FilterMode_2_Median_Run();
            break;
        }
        case GAUSSIANFILTER:
        {
            FilterMode_3_Gaussian_Run();
            break;
        }
        case BILATERALFILTER:
        {
            FilterMode_4_Bilateral_Run();
            break;
        }
        default:
            break;
    }
}


// 模块12.1： 图像窗口化（鼠标事件）
void MainWindow::My_OnMouse(int event, int x, int y, int flags, void* userdata)// 原本利用滚轮实现，但是可能滚轮事件被其他不知名程序占用了，因此弃用。
{
//    cv::Mat* mouse_mat = (cv::Mat*)userdata;
    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN: // 鼠标左键点击放大
        {
            mouse_scaleFactor += MouseZoomRes;
            if (mouse_scaleFactor > 10) mouse_scaleFactor = 10;
            break;
        }
        case cv::EVENT_RBUTTONDOWN: // 鼠标右键点击缩小
        {
            mouse_scaleFactor -= MouseZoomRes;
            if (mouse_scaleFactor < 0.1) mouse_scaleFactor = 0.1;
            break;
        }
        default:
            break;
    }

    Q_UNUSED(userdata);
    Q_UNUSED(x);
    Q_UNUSED(y);
    Q_UNUSED(flags);

}

// 模块12.2： 图像窗口化（主程序）
void MainWindow::on_image_windows_pb_clicked()
{
    if(!isThereImage())
        return;

    windows_image = ui->label_image->pixmap()->toImage();

    cv::Mat windows_image_in = My_QImage2Mat(windows_image);
    cv::Mat windows_image_out;

    cv::namedWindow("Image Display and Zoom");// 创建一个窗口，用于显示图片
    cv::imshow("Image Display and Zoom", windows_image_in);

    cv::setMouseCallback("Image Display and Zoom", MainWindow::My_OnMouse, &windows_image_in);// 设置鼠标滚轮的回调函数

    while (true)
    {
        cv::resize(windows_image_in, windows_image_out, cv::Size(), mouse_scaleFactor, mouse_scaleFactor, INTER_LANCZOS4);
        cv::imshow("Image Display and Zoom", windows_image_out);

        int key = cv::waitKey(10);
        if (key == 27)
            break; // 等待用户按键，如果按下ESC键，则退出循环
        if (cv::getWindowProperty("Image Display and Zoom", cv::WND_PROP_AUTOSIZE)!= 1 )
            break; // 或者点击窗口的关闭按钮，利用AUTOSIZE是否存在（为1）检测窗口是否存在,也可以退出循环
    }

    cv::destroyAllWindows();
}

// 模块13.1： 冷暖度调节
void MainWindow::on_warmcold_hs_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->warmcold_hs->setValue(0);
            return;
        }
    }

    isWarmcoldReleased = false;

    if (isFirstWarmcoldComeIn)
    {
        isFirstWarmcoldComeIn = false;
        warmcold_image_init = ui->label_image->pixmap()->toImage();
    }
    warmcold_image_now = warmcold_image_init;

    QColor oldcolor;
    int r,g,b;

    if (value >= 0)
    {
        for (int x=0; x<warmcold_image_now.width(); x++){
            for (int y=0; y<warmcold_image_now.height(); y++){
                oldcolor = QColor(warmcold_image_now.pixel(x,y));
                r = oldcolor.red() + value;
                g = oldcolor.green() + value;
                b = oldcolor.blue();

                r = qBound(0,r,255);
                g = qBound(0,g,255);

                warmcold_image_now.setPixel(x, y, qRgb(r,g,b));
            }
        }
    }
    else
    {
        for (int x=0; x<warmcold_image_now.width(); x++){
            for (int y=0; y<warmcold_image_now.height(); y++){
                oldcolor = QColor(warmcold_image_now.pixel(x,y));
                r = oldcolor.red();
                g = oldcolor.green();
                b = oldcolor.blue() + (-value);

                b = qBound(0,b,255);

                warmcold_image_now.setPixel(x, y, qRgb(r,g,b));
            }
        }
    }

    show_image_now(warmcold_image_now);
    ui->warmcold_lb_2->setText(QString::number(value));
}

// 模块13.2： 冷暖度调节初始化
void MainWindow::on_warmcold_reset_pb_clicked()
{
    if(!isThereImage())
        return;
    isFirstWarmcoldComeIn = true;
}

// 模块13.3： 冷暖调节结束保存（滑动条专属）
void MainWindow::on_warmcold_hs_sliderReleased()
{
    isWarmcoldReleased = true;
    forcedPermission = true;
    show_image_now(warmcold_image_now);// 更新最后一次图像，并保存在undo缓存区
}


// 模块14.0： Laplace锐化算子计算
void MainWindow::getLaplaceKernel(int level, int kernel[3][3], int &sumKernel)
{
    switch (level) { // 根据锐化等级选择不同的算子
    case 0:
        kernel[0][0] = 0; kernel[0][1] = 0; kernel[0][2] = 0;
        kernel[1][0] = 0; kernel[1][1] = 1; kernel[1][2] = 0;
        kernel[2][0] = 0; kernel[2][1] = 0; kernel[2][2] = 0;
        sumKernel = 1;
        ui->statusbar->showMessage(tr("当前[尚未选择]锐化等级！"));
        break;
    case 1: // 锐化等级1
        kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
        kernel[1][0] = -1; kernel[1][1] = 20; kernel[1][2] = -1;
        kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;
        sumKernel = 16;
        break;
    case 2: // 锐化等级2
        kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
        kernel[1][0] = -1; kernel[1][1] = 15; kernel[1][2] = -1;
        kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;
        sumKernel = 11;
        break;
    case 3: // 锐化等级3
        kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
        kernel[1][0] = -1; kernel[1][1] = 12; kernel[1][2] = -1;
        kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;
        sumKernel = 8;
        break;
    case 4: // 锐化等级4
    case 5: // 锐化等级5
    case 6: // 锐化等级6
    case 7: // 锐化等级7
    case 8: // 锐化等级8
    case 9: // 锐化等级9
        kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
        kernel[1][0] = -1; kernel[1][1] = 14-level; kernel[1][2] = -1;
        kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;
        sumKernel = 10-level;
        break;
    case 10: // 锐化等级10
    case 11: // 锐化等级11
        kernel[0][0] = -1; kernel[0][1] = -1; kernel[0][2] = -1;
        kernel[1][0] = -1; kernel[1][1] = 20-level; kernel[1][2] = -1;
        kernel[2][0] = -1; kernel[2][1] = -1; kernel[2][2] = -1;
        sumKernel = 12-level;
        break;
    case 12: // 锐化等级12
        kernel[0][0] = -1; kernel[0][1] = -2; kernel[0][2] = -1;
        kernel[1][0] = -2; kernel[1][1] = 13; kernel[1][2] = -2;
        kernel[2][0] = -1; kernel[2][1] = -2; kernel[2][2] = -1;
        sumKernel = 1;
        break;
    case 13: // 锐化等级13
        kernel[0][0] = -2; kernel[0][1] = -2; kernel[0][2] = -2;
        kernel[1][0] = -2; kernel[1][1] = 17; kernel[1][2] = -2;
        kernel[2][0] = -2; kernel[2][1] = -2; kernel[2][2] = -2;
        sumKernel = 1;
        break;
    case 14: // 锐化等级14
        kernel[0][0] = -2; kernel[0][1] = -3; kernel[0][2] = -2;
        kernel[1][0] = -3; kernel[1][1] = 21; kernel[1][2] = -3;
        kernel[2][0] = -2; kernel[2][1] = -3; kernel[2][2] = -2;
        sumKernel = 1;
        break;
    case 15: // 锐化等级15
        kernel[0][0] = -3; kernel[0][1] = -3; kernel[0][2] = -3;
        kernel[1][0] = -3; kernel[1][1] = 25; kernel[1][2] = -3;
        kernel[2][0] = -3; kernel[2][1] = -3; kernel[2][2] = -3;
        sumKernel = 1;
        break;
    default: // 其他情况，返回空的算子
        kernel = {0};
        ui->statusbar->showMessage(tr("当前[锐化等级]错误！"));
        break;
    }
}

// 模块14.1： Laplace锐化调节
void MainWindow::on_laplacesharpen_hs_valueChanged(int value)
{   
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->laplacesharpen_hs->setValue(0);
            return;
        }
    }


    isLaplacesharpenReleased = false;

    if (isFirstLaplacesharpenComeIn)
    {
        isFirstLaplacesharpenComeIn = false;
        laplacesharpen_image_init = ui->label_image->pixmap()->toImage();
    }
    laplacesharpen_image_now = laplacesharpen_image_init;
    laplacesharpen_image_out = laplacesharpen_image_now.copy();

    int kernelSize = 3;
    int sumKernel;
    int kernel[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

    getLaplaceKernel(value,kernel,sumKernel);

    int r,g,b;
    QColor color;

    for(int x=kernelSize/2; x<laplacesharpen_image_now.width()-(kernelSize/2); x++){
        for(int y=kernelSize/2; y<laplacesharpen_image_now.height()-(kernelSize/2); y++){
            r = 0;
            g = 0;
            b = 0;

            for(int i = -kernelSize/2; i<= kernelSize/2; i++){
                for(int j = -kernelSize/2; j<= kernelSize/2; j++){
                    color = QColor(laplacesharpen_image_now.pixel(x+i, y+j));
                    r += color.red()*kernel[kernelSize/2+i][kernelSize/2+j];
                    g += color.green()*kernel[kernelSize/2+i][kernelSize/2+j];
                    b += color.blue()*kernel[kernelSize/2+i][kernelSize/2+j];
                }
            }

            r = qBound(0, r/sumKernel, 255);
            g = qBound(0, g/sumKernel, 255);
            b = qBound(0, b/sumKernel, 255);
            laplacesharpen_image_out.setPixel(x,y, qRgb(r,g,b));

        }
    }

    show_image_now(laplacesharpen_image_out);
    ui->laplacesharpen_lb_2->setText(QString::number(value));
}

// 模块14.2： Laplace锐化调节初始化
void MainWindow::on_laplacesharpen_reset_pb_clicked()
{
    if(!isThereImage())
        return;
    isFirstLaplacesharpenComeIn = true;
}

// 模块14.3： Laplace锐化调节结束保存（滑动条专属）
void MainWindow::on_laplacesharpen_hs_sliderReleased()
{
    isLaplacesharpenReleased = true;
    forcedPermission = true;
    show_image_now(laplacesharpen_image_out);// 更新最后一次图像，并保存在undo缓存区
}


// 模块15： 简单梯度锐化
void MainWindow::on_gradientsharpen_pb_clicked()
{
    if (!isThereImage())
        return;

    gradientsharpen_image = ui->label_image->pixmap()->toImage();

    // 常用Sobel算子的水平和垂直方向的卷积核
//    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
//    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    // 课程课件对应的简单算子的水平和垂直方向的卷积核
    int kernelX[3][3] = {{0, 0, 0}, {0, -1, 1}, {0, 0, 0}};
    int kernelY[3][3] = {{0, 0, 0}, {0, -1, 0}, {0, 1, 0}};
    int T_g_min = 10; // 梯度过小阈值
    int T_g_max = 120; // 梯度过大阈值

    for (int x = 1; x < gradientsharpen_image.width() - 1; x++) {
        for (int y = 1; y < gradientsharpen_image.height() - 1; y++) {

            // 使用灰度计算梯度, 也可以选用彩色3通道（即2种方法，但后者暂不表）
            int gray[3][3];
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gray[i + 1][j + 1] = qGray(gradientsharpen_image.pixel(x + i, y + j));
                }
            }

            // 计算水平/垂直梯度
            int gx = 0;
            int gy = 0;
            int r,g,b,grad;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    gx += kernelX[i][j] * gray[i][j];
                    gy += kernelY[i][j] * gray[i][j];
                }
            }

            // 计算总梯度
            grad = sqrt(gx*gx + gy*gy);
            grad = qBound(0, grad, 255);

            // 计算锐化像素值
            if (grad > T_g_max)
            {
                r = 220;
                g = 220;
                b = 220;
            }
            else if (T_g_max>grad && grad>T_g_min)
            {
                r = (grad + qRed(gradientsharpen_image.pixel(x, y))) / 1; // 方法1：加权平均方法
                g = (grad + qGreen(gradientsharpen_image.pixel(x, y))) / 1;
                b = (grad + qBlue(gradientsharpen_image.pixel(x, y))) / 1;
            }
            else
            {
                r = qRed(gradientsharpen_image.pixel(x, y));
                g = qGreen(gradientsharpen_image.pixel(x, y));
                b = qBlue(gradientsharpen_image.pixel(x, y));
            }

            r = qBound(0, r, 255);
            g = qBound(0, g, 255);
            b = qBound(0, b, 255);

//            gradientsharpen_image.setPixel(x, y, qRgb(grad,grad,grad)); // 边缘检测结果显示
            gradientsharpen_image.setPixel(x, y, qRgb(r,g,b)); // 梯度锐化结果显示
        }
    }
    show_image_now(gradientsharpen_image);
}


// 模块16.0： 边缘检测模式提取
void MainWindow::on_edgeDetect_cbox_currentIndexChanged(int index)
{
    EdgeDetectMode = index;
}

// 模块16.1.a： 边缘检测算法Roberts实现
void MainWindow::EdgeDetectMode_1_Roberts(QImage& src, QImage& dst)
{
    int kernelX[2][2] = {{1, 0}, {0, -1}};
    int kernelY[2][2] = {{0, 1}, {-1, 0}};

    for (int x = 0; x < src.width() - 1; x++) {
        for (int y = 0; y < src.height() - 1; y++) {
            int gray[2][2];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    gray[i][j] = qGray(src.pixel(x + i, y + j));
                }
            }
            // 计算水平和垂直方向的梯度值
            int gx = 0;
            int gy = 0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    gx += kernelX[i][j] * gray[i][j];
                    gy += kernelY[i][j] * gray[i][j];
                }
            }
            // 计算总的梯度值
            // int g = abs(gx) + abs(gy); // 绝对值方法
            int g = sqrt(gx * gx + gy * gy); // 平方和开根号方法

            g = qBound(0, g, 255);
            dst.setPixel(x, y, qRgb(g, g, g));
        }
    }

    show_image_now(dst);
}

// 模块16.1.b： 边缘检测算法Sobel实现
void MainWindow::EdgeDetectMode_2_Sobel(QImage& src, QImage& dst)
{
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int x = 1; x < src.width() - 1; x++) {
        for (int y = 1; y < src.height() - 1; y++) {
            int gray[3][3];
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gray[i + 1][j + 1] = qGray(src.pixel(x + i, y + j));
                }
            }

            int gx = 0;
            int gy = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    gx += kernelX[i][j] * gray[i][j];
                    gy += kernelY[i][j] * gray[i][j];
                }
            }

            // int g = abs(gx) + abs(gy); // 绝对值方法
            int g = sqrt(gx * gx + gy * gy); // 平方和开根号方法

            g = qBound(0, g, 255);
            dst.setPixel(x, y, qRgb(g, g, g));
        }
    }

    show_image_now(dst);
}

// 模块16.1.c： 边缘检测算法Laplace实现
void MainWindow::EdgeDetectMode_3_Laplace(QImage& src, QImage& dst)
{
    int kernel[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};

    for (int x = 1; x < src.width() - 1; x++) {
        for (int y = 1; y < src.height() - 1; y++) {

        int gray[3][3];
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                gray[i + 1][j + 1] = qGray(src.pixel(x + i, y + j));
            }
        }

        int g = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                g += kernel[i][j] * gray[i][j];
            }
        }

        g = qBound(0, g, 255);
        dst.setPixel(x, y, qRgb(g, g, g));
        }
    }

    show_image_now(dst);
}

// 模块16.1.d： 边缘检测算法Prewitt实现
void MainWindow::EdgeDetectMode_4_Prewitt(QImage& src, QImage& dst)
{
    int kernelX[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    for (int x = 1; x < src.width() - 1; x++) {
        for (int y = 1; y < src.height() - 1; y++) {

            int gray[3][3];
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gray[i + 1][j + 1] = qGray(src.pixel(x + i, y + j));
                }
            }

            int gx = 0;
            int gy = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    gx += kernelX[i][j] * gray[i][j];
                    gy += kernelY[i][j] * gray[i][j];
                }
            }

            // int g = abs(gx) + abs(gy); // 绝对值方法
            int g = sqrt(gx * gx + gy * gy); // 平方和开根号方法

            g = qBound(0, g, 255);
            dst.setPixel(x, y, qRgb(g, g, g));
        }
    }

    show_image_now(dst);
}

// 模块16.1.e： 边缘检测算法Canny实现
void MainWindow::EdgeDetectMode_5_Canny(QImage& src, QImage& dst, double thresholdLow, double thresholdHigh, int WeakEdgeThreshold)
{
    int edge_width = src.width();
    int edge_height = dst.height();

    if (edge_width<5 || edge_height<5)
    {
        QMessageBox::information(this, tr("嘿！"), tr("当前图片过小无法进行Canny边缘检测！"), QMessageBox::Ok | QMessageBox::Yes);
        return;
    }

    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int lowThreshold = thresholdLow;
    int highThreshold = thresholdHigh;

    int LjxWeakEdgeThreshold = WeakEdgeThreshold; // ljx modify, 越大，边缘信息与噪点将一同增多, 1~9, default=10( Don't use Ljx's modify )
    double kernelGauss[5][5] =  {{2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0},
                            {4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0,9.0 /159.0 , 4.0 /159.0 },
                            {5.0 /159.0 , 12.0 /159.0 ,15.0 /159.0 ,12.0 /159.0 ,5.0 /159.0 },
                            {4.0 /159.0 , 9.0 /159.0 , 12.0 /159.0 ,9.0 /159.0 , 4.0 /159.0 },
                            {2.0 /159.0 , 4.0 /159.0 , 5.0 /159.0 , 4.0 /159.0 , 2.0 /159.0 } };

    int smoothedGray[edge_width][edge_height];
    int gradientValue[edge_width][edge_height];
    double gradientDirection[edge_width][edge_height];
    int gradientValueNonMaxSuppress[edge_width][edge_height];
    int edgeClasses[edge_width][edge_height];

    // 1.进行高斯滤波，平滑图像
    for (int x = 2; x < edge_width - 2; x++) {
        for (int y = 2; y < edge_height - 2; y++) {

            int gray[5][5];
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    gray[i + 2][j + 2] = qGray(edgeDetect_image.pixel(x + i, y + j));
                }
            }

            double guass_smoothed_now = 0;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    guass_smoothed_now += kernelGauss[i][j] * gray[i][j];
                }
            }

            smoothedGray[x][y] = qBound(0, int(guass_smoothed_now), 255);
        }
    }

    // 2.计算梯度值和方向
    for (int x = 1; x < edge_width - 1; x++) {
        for (int y = 1; y < edge_height - 1; y++) {

            int gray[3][3];
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if ((x+i)<2 || (y+j)<2 || (x+i)>=edge_width-2 || (y+j)>=edge_height)
                        gray[i + 1][j + 1] = qGray(edgeDetect_image.pixel(x + i, y + j));
//                        gray[i + 1][j + 1] = 0;
                    else
                        gray[i + 1][j + 1] = smoothedGray[x + i][y + j];
                }
            }

            int gx = 0;
            int gy = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    gx += kernelX[i][j] * gray[i][j];
                    gy += kernelY[i][j] * gray[i][j];
                }
            }

//             int gValue = abs(gx) + abs(gy); // 绝对值方法
            int gValue = sqrt(gx * gx + gy * gy); // 平方和开根号方法
            double gTheta = atan2(gy, gx) * 180.0 / M_PI; // 计算梯度的方向

            gValue = qBound(0, gValue, 255);

            gradientValue[x][y] = gValue;
            gradientDirection[x][y] = gTheta;
        }
    }

    // 3.非极大值抑制
    for (int x = 1; x < edge_width - 1; x++) {
        for (int y = 1; y < edge_height - 1; y++) {

            int gValue = gradientValue[x][y];
            double gTheta = gradientDirection[x][y];

            // 根据梯度方向，将其量化为四个主要方向
            int direction;
            if ((gTheta >= -22.5 && gTheta <= 22.5) || (gTheta >= 157.5 && gTheta <= 180.0) || (gTheta >= -180.0 && gTheta <= -157.5)) // 水平方向
                direction = 0;
            else if ((gTheta > 22.5 && gTheta <= 67.5) || (gTheta < -112.5 && gTheta >= -157.5))// 正对角线方向
                direction = 1;
            else if ((gTheta > 67.5 && gTheta <= 112.5) || (gTheta < -67.5 && gTheta >= -112.5)) // 垂直方向
                direction = 2;
            else if ((gTheta > 112.5 && gTheta < 157.5) || (gTheta < -22.5 && gTheta >= -67.5)) // 负对角线方向
                direction = 3;
            else
                direction = -1;

            // 根据梯度方向，比较当前像素与其两个相邻像素的梯度值，如果当前像素不是局部最大，则将其设为0
            switch (direction) {
                case 0: // 水平方向
                {
                    if (gValue < gradientValue[x - 1][y] || gValue < gradientValue[x + 1][y])
                        gValue = 0;
                    break;
                }
                case 1: // 正对角线方向
                {
                    if (gValue < gradientValue[x - 1][y - 1] || gValue < gradientValue[x + 1][y + 1])
                        gValue = 0;
                    break;
                }
                case 2: // 垂直方向
                {
                    if (gValue < gradientValue[x][y - 1] || gValue < gradientValue[x][y + 1])
                        gValue = 0;
                    break;
                }
                case 3: // 负对角线方向
                {
                    if (gValue < gradientValue[x + 1][y - 1] || gValue < gradientValue[x - 1][y + 1])
                        gValue = 0;
                    break;
                }
                default:
                {
                    ui->statusbar->showMessage(tr("Error 4：Canny算法梯度方向错误！"));
                    break;
                }
            }
            gradientValueNonMaxSuppress[x][y] = gValue;
        }
    }

    // 4.双阈值检测
    for (int x = 1; x < edge_width - 1; x++) {
        for (int y = 1; y < edge_height - 1; y++) {

            int gValue = gradientValueNonMaxSuppress[x][y];
            // 根据两个阈值，将梯度值分为三类
            if (gValue >= highThreshold) // 强边缘，设为255
                gValue = 255;
            else if (gValue < lowThreshold) // 非边缘，设为0
                gValue = 0;
            else // 弱边缘，设为128
                gValue = 128;

            edgeClasses[x][y] = gValue;
        }
    }

    // 5.边缘连接
    for (int x = 1; x < edge_width - 1; x++) {
        for (int y = 1; y < edge_height - 1; y++) {

            int edgeClass = edgeClasses[x][y];
            // 如果当前像素是弱边缘，检查其周围8个像素是否有强边缘，如果有，则将其设为255，否则设为0
            if (edgeClass == 128) {
                bool hasStrongEdge = false;
                int numWeakEdge = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (edgeClasses[x + i][y + j] == 255) // ljx modify
                        {
                            hasStrongEdge = true;
                            break;
                        }
                        if (edgeClasses[x + i][y + j] == 128)
                            numWeakEdge += 1;
                    }
                    if (hasStrongEdge)
                        break;
                }

                if (hasStrongEdge)  // 边缘连接，设为255
                    edgeClass = 255;
                else if (numWeakEdge >= LjxWeakEdgeThreshold) // ljx modify
                    edgeClass = 255;
                else  // 边缘断开，设为0
                    edgeClass = 0;
            }
            edgeClasses[x][y] = edgeClass;

            dst.setPixel(x, y, qRgb(edgeClass, edgeClass, edgeClass));
        }
    }

    // 6.显示图像
    show_image_now(dst);
}

// 模块16.1.f： 边缘检测算法Canny-OpenCV官方实现
void MainWindow::EdgeDetectMode_5_Canny_OpenCV_Office(QImage& src, QImage& dst, double thresholdLow, double thresholdHigh)
{
    cv::Mat CannyInputMat = My_QImage2Mat(src);
    cv::Mat CannyOutputMat;
    cv::Canny(CannyInputMat, CannyOutputMat, thresholdLow, thresholdHigh);
    dst = My_Mat2QImage(CannyOutputMat).copy();
    show_image_now(dst);
}

// 模块16.2： 边缘检测入口
void MainWindow::on_edgeDetect_pb_clicked()
{
    if (EdgeDetectMode==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->edgeDetect_cbox->setCurrentIndex(0);
            return;
        }
    }

    edgeDetect_image = ui->label_image->pixmap()->toImage();
    QImage edgeDetect_result_image = QImage(edgeDetect_image.size(), QImage::Format_ARGB32);

    switch (EdgeDetectMode) {
        case Roberts:
        {
            EdgeDetectMode_1_Roberts(edgeDetect_image, edgeDetect_result_image);
            break;
        }

        case Sobel:
        {
            EdgeDetectMode_2_Sobel(edgeDetect_image, edgeDetect_result_image);
            break;
        }

        case Laplace:
        {
            EdgeDetectMode_3_Laplace(edgeDetect_image, edgeDetect_result_image);
            break;
        }

        case Prewitt:
        {
            EdgeDetectMode_4_Prewitt(edgeDetect_image, edgeDetect_result_image);
            break;
        }

        case Canny:
        {
            EdgeDetectMode_5_Canny(edgeDetect_image, edgeDetect_result_image, 10, 50, 10); // nice: 10-40/50-10;
            break;
        }
        case Canny_OpenCV:
        {
            EdgeDetectMode_5_Canny_OpenCV_Office(edgeDetect_image, edgeDetect_result_image, 50, 150);
            break;
        }

        default:
        {
            ui->statusbar->showMessage(tr("当前[算法模式]尚未选择！"));
            break;
        }
    }

}

// 模块16.3： 边缘检测刷新
void MainWindow::on_edgeDetect_reset_pb_clicked()
{
    if (!isThereImage())
        return;
    show_image_now(img_raw);
}

// 模块17： 边窗均值滤波
void MainWindow::on_sideWindow_meanFilter_pb_clicked()
{
    if (!isThereImage())
        return;

    QImage sidewindow_mean_image = ui->label_image->pixmap()->toImage();
    cv::Mat inputMat = My_QImage2Mat(sidewindow_mean_image);

    int W = inputMat.cols;
    int H = inputMat.rows;
    int SideWindow_size = 3;

    double ZS_sw[9] = {1.0/4, 1.0/4, 0, 1.0/4, 1.0/4, 0, 0, 0, 0};
    double S_sw[9] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 0, 0, 0};
    double YS_sw[9] = {0, 0, 0, 1.0/4, 1.0/4, 0, 1.0/4, 1.0/4, 0};
    double Z_sw[9] = {1.0/6, 1.0/6, 0, 1.0/6, 1.0/6, 0, 1.0/6, 1.0/6, 0};
    double Y_sw[9] = {0, 1.0/6, 1.0/6, 0, 1.0/6, 1.0/6, 0, 1.0/6, 1.0/6};
    double ZX_sw[9] = {0, 0, 0, 1.0/4, 1.0/4, 0, 1.0/4, 1.0/4, 0};
    double X_sw[9] = {0, 0, 0, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
    double YX_sw[9] = {0, 0, 0, 0, 1.0/4, 1.0/4, 0, 1.0/4, 1.0/4};

    cv::Mat ZS_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,ZS_sw);
    cv::Mat S_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,S_sw);
    cv::Mat YS_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,YS_sw);
    cv::Mat Z_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,Z_sw);
    cv::Mat Y_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,Y_sw);
    cv::Mat ZX_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,ZX_sw);
    cv::Mat X_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,X_sw);
    cv::Mat YX_mat = Mat(SideWindow_size, SideWindow_size,CV_64F,YX_sw);

    cv::Mat inputChannels[3];
    cv::split(inputMat, inputChannels);

    QVector<cv::Mat> outputChannels(3);

    for (int c = 0; c < 3; c++)
    {
        cv::Mat outputChannel(H, W, CV_8UC1);

        for (int i = SideWindow_size/2; i < W - SideWindow_size/2; i++){
            for (int j = SideWindow_size/2; j < H - SideWindow_size/2; j++){

                cv::Mat neighborhood = inputChannels[c](cv::Rect(i - 1, j - 1, 3, 3));
                cv::Mat neighborhoodDouble;
                neighborhood.convertTo(neighborhoodDouble, CV_64F);

                QVector<cv::Mat> result_sw(8);
                filter2D(neighborhoodDouble, result_sw[0], -1, ZS_mat);
                filter2D(neighborhoodDouble, result_sw[1], -1, S_mat);
                filter2D(neighborhoodDouble, result_sw[2], -1, YS_mat);
                filter2D(neighborhoodDouble, result_sw[3], -1, Z_mat);
                filter2D(neighborhoodDouble, result_sw[4], -1, Y_mat);
                filter2D(neighborhoodDouble, result_sw[5], -1, ZX_mat);
                filter2D(neighborhoodDouble, result_sw[6], -1, X_mat);
                filter2D(neighborhoodDouble, result_sw[7], -1, YX_mat);

                int index_result_min;
                for (int k = 0; k < 8; k++)
                {
                    if (k == 0)
                    index_result_min = k;
                    else
                    {
                        if (norm(result_sw[k], NORM_L2) < norm(result_sw[k-1], NORM_L2))
                        index_result_min = k;
                    }
                }

                uchar outputPixelValue = static_cast<uchar>(result_sw[index_result_min].at<double>(1, 1));
                outputChannel.at<uchar>(j, i) = outputPixelValue;
            }
        }

        outputChannels[c] = outputChannel;
    }

    cv::Mat outputMat;
    cv::merge(outputChannels.toStdVector(), outputMat);
    QImage outputImage = My_Mat2QImage(outputMat);

    show_image_now(outputImage);
}


// 模块18.1： 腐蚀膨胀模式获取
void MainWindow::on_ErodeDilate_vhb_valueChanged(int value)
{
    if (value==0)
    {
        if (!isThereImage(false))
            return;
    }
    else
    {
        if (!isThereImage())
        {
            ui->ErodeDilate_vhb->setValue(0);
            return;
        }
    }

    ErodeDilateMode = value;

    switch (ErodeDilateMode)
    {
        case DEFAULT_EDM:
        {
            ui->statusbar->showMessage(tr("当前[尚未选择]腐蚀膨胀模式！"));
            ui->E_D_lb2->setText(tr("缺 省"));
            break;
        }
        case ERODE:
        {
            ui->statusbar->showMessage(tr("当前为[腐蚀]模式"));
            ui->E_D_lb2->setText(tr("腐蚀"));
            break;
        }
        case DILATE:
        {
            ui->statusbar->showMessage(tr("当前为[膨胀]模式"));
            ui->E_D_lb2->setText(tr("膨胀"));
            break;
        }
        case OPENING:
        {
            ui->statusbar->showMessage(tr("当前为[开操作]模式，腐蚀->膨胀"));
            ui->E_D_lb2->setText(tr("开操作"));
            break;
        }
        case CLOSING:
        {
            ui->statusbar->showMessage(tr("当前为[闭操作]模式，膨胀->腐蚀"));
            ui->E_D_lb2->setText(tr("闭操作"));
            break;
        }
        default:
        {
            QMessageBox::information(this, tr("诶！"), tr("图像腐蚀膨胀模式选择错误！"), QMessageBox::Ok | QMessageBox::Yes);
            break;
        }
    }
}

// 模块18.2： 腐蚀膨胀具体实现
void MainWindow::on_erodedilate_pb_clicked()
{
    if (!isThereImage())
        return;

    ErodeDilateKernelSize = ui->ErodeDilate_sb->value();
    if (ErodeDilateKernelSize<1 || ErodeDilateKernelSize>25)
    {
        ErodeDilateKernelSize = -1;
        return;
    }
    else if (ErodeDilateKernelSize%2==0)
    {
        ErodeDilateKernelSize = -1;
        QMessageBox::information(this, tr("嘿！"), tr("请输入1-25内的奇数！"), QMessageBox::Ok | QMessageBox::Yes);
        return;
    }

    if (isFirstErodeDilateComeIn)
    {
        isFirstErodeDilateComeIn = false;
        erodedilate_image_init = ui->label_image->pixmap()->toImage();
    }
    erodedilate_image_now = erodedilate_image_init;

    cv::Mat ED_input_mat = My_QImage2Mat(erodedilate_image_now);
    cv::Mat ED_output_mat= ED_input_mat.clone();

    //MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE, 分别为矩形、椭圆和交叉形,分别为0,1,2
    cv::Mat ED_kernel = getStructuringElement(MORPH_RECT, Size(ErodeDilateKernelSize, ErodeDilateKernelSize));

    switch (ErodeDilateMode) {
        case DEFAULT_EDM:
        {
            QMessageBox::information(this, tr("哎！"), tr("尚未选择:图像腐蚀膨胀模式！"), QMessageBox::Ok | QMessageBox::Yes);
            break;
        }
        case ERODE:
        {
            cv::erode(ED_input_mat, ED_output_mat, ED_kernel);
            break;
        }
        case DILATE:
        {
            cv::dilate(ED_input_mat, ED_output_mat, ED_kernel);
            break;
        }
        case OPENING:
        {
            cv::morphologyEx(ED_input_mat, ED_output_mat, MORPH_OPEN, ED_kernel);
            break;
        }
        case CLOSING:
        {
            cv::morphologyEx(ED_input_mat, ED_output_mat, MORPH_CLOSE, ED_kernel);
            break;
        }
        default:
        {
            QMessageBox::information(this, tr("诶！"), tr("图像腐蚀膨胀模式选择错误！"), QMessageBox::Ok | QMessageBox::Yes);
            break;
        }
    }

    erodedilate_image_now = My_Mat2QImage(ED_output_mat);
    show_image_now(erodedilate_image_now);
}


void MainWindow::on_warp_affine_pb_3points_clicked()
{
    if (!isThereImage())
        return;

    QImage warp_1 = ui->label_image->pixmap()->toImage();
    cv::Mat src_1 = My_QImage2Mat(warp_1);
    cv::Mat dst_1 = Mat::zeros(src_1.rows, src_1.cols, src_1.type());
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Mat warp_mat(2, 3, CV_32FC1);

    Point2f srcTri[3], dstTri[3];

    srcTri[0] = Point2f(0, 0);
    srcTri[1] = Point2f(src_1.cols - 1,0);
    srcTri[2]= Point2f(0,src_1.rows - 1);

    dstTri[0] =Point2f(src_1.cols * 0.0,src_1.rows * 0.33);
    dstTri[1] =Point2f(src_1.cols * 0.85,src_1.rows * 0.25);
    dstTri[2] =Point2f(src_1.cols* 0.15,src_1.rows* 0.7);

    warp_mat=getAffineTransform( srcTri, dstTri);
    warpAffine( src_1, dst_1, warp_mat,src_1.size());

    warp_1 = My_Mat2QImage(dst_1);
    show_image_now(warp_1);
}

void MainWindow::on_warp_affine_pb_rotate_clicked()
{
    if (!isThereImage())
        return;

    QImage warp_2 = ui->label_image->pixmap()->toImage();
    cv::Mat src_2 = My_QImage2Mat(warp_2);
    cv::Mat dst_2 = Mat::zeros(src_2.rows, src_2.cols, src_2.type());
    cv::Mat rot_mat(2, 3, CV_32FC1);

    Point2f center = Point2f( src_2.cols/2, src_2.rows/2);
    double angle = -50.0;
    double scale = 0.6;

    rot_mat=getRotationMatrix2D( center, angle, scale );
    warpAffine( src_2, dst_2, rot_mat, src_2.size());

    warp_2 = My_Mat2QImage(dst_2);
    show_image_now(warp_2);
}

void MainWindow::warpPerspectiveBase(Point2f srcDynamicQuad[4], Point2f dstDynamicQuad[4], Mat src, Mat& dst)
{
    if (!isThereImage())
        return;

    Mat warp_matrix(3,3,CV_32FC1);

    warp_matrix=getPerspectiveTransform(srcDynamicQuad,dstDynamicQuad);
    warpPerspective(src,dst,warp_matrix,src.size());
}

void MainWindow::on_warp_perspective_pb_4points_clicked()
{
    if (!isThereImage())
        return;

    QImage warp_3 = ui->label_image->pixmap()->toImage();
    cv::Mat src_3 = My_QImage2Mat(warp_3);

    Point2f srcQuad[4], dstQuad[4];

    srcQuad[0]=Point2f(0,0);
    srcQuad[1]=Point2f(src_3.cols -1,0);
    srcQuad[2]=Point2f(0, src_3.rows-1);
    srcQuad[3]=Point2f(src_3.cols -1, src_3.rows-1);

    dstQuad[0]=Point2f(src_3.cols*0.05,src_3.rows*0.33);
    dstQuad[1]=Point2f(src_3.cols*0.9,src_3.rows*0.25);
    dstQuad[2]=Point2f(src_3.cols*0.2,src_3.rows*0.7);
    dstQuad[3]=Point2f(src_3.cols*0.8,src_3.rows*0.9);

    cv::Mat dst_3 = Mat::zeros(src_3.rows, src_3.cols, src_3.type());
    warpPerspectiveBase(srcQuad, dstQuad, src_3, dst_3);
    warp_3 = My_Mat2QImage(dst_3);
    show_image_now(warp_3);

}



double MainWindow::generateGaussianNoise(double mu, double sigma)
{
    static bool isFirstGenGaussNoiseComeIn = true;
    if (isFirstGenGaussNoiseComeIn)
    {
        srand(time(NULL));// 设置随机数种子
        isFirstGenGaussNoiseComeIn = false;
    }

    //定义小值
    const double epsilon = 1.0e-8; // Need Try
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);

    //flag为真构造高斯随机Box-Muller变量
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
    return z0*sigma + mu;
}

Mat MainWindow::addGaussianNoise(Mat &srcImag, double mu, double sigma)
{
    Mat dstImage = srcImag.clone();
    for (int i = 0; i < dstImage.rows; i++)
    {
        for (int j = 0; j < dstImage.cols; j++)
        {
            //添加高斯噪声
            dstImage.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[0] + generateGaussianNoise(mu, sigma) * 32);
            dstImage.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[1] + generateGaussianNoise(mu, sigma) * 32);
            dstImage.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[2] + generateGaussianNoise(mu, sigma) * 32);
        }
    }
    return dstImage;
}

Mat MainWindow::addSaltNoise(const Mat srcImage, int num)
{
    Mat dstImage = srcImage.clone();
    for (int k = 0; k < num; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 255;		//盐噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 255;
            dstImage.at<Vec3b>(i, j)[1] = 255;
            dstImage.at<Vec3b>(i, j)[2] = 255;
        }
    }

    for (int k = 0; k < num; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 0;		//椒噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 0;
            dstImage.at<Vec3b>(i, j)[1] = 0;
            dstImage.at<Vec3b>(i, j)[2] = 0;
        }
    }
    return dstImage;
}

void MainWindow::on_gaussnoise_dsb1_valueChanged(double arg1)
{
    GaussNoiseMu = arg1;
}

void MainWindow::on_gaussnoise_dsb2_valueChanged(double arg1)
{
    GaussNoiseSigma = arg1;
}

void MainWindow::on_saltnoise_sb_valueChanged(int arg1)
{
    SaltNoiseNum = arg1;
}

void MainWindow::on_gaussNoise_pb_clicked()
{
    if (!isThereImage() || GaussNoiseMu<0 || GaussNoiseSigma<0)
        return;

    gaussnoise_image = ui->label_image->pixmap()->toImage();
    cv::Mat gaussnoise_mat_in = My_QImage2Mat(gaussnoise_image);
    cv::Mat gaussnoise_mat_out;

    gaussnoise_mat_out = addGaussianNoise(gaussnoise_mat_in, GaussNoiseMu, GaussNoiseSigma);
    gaussnoise_image = My_Mat2QImage(gaussnoise_mat_out);

    show_image_now(gaussnoise_image);
}

void MainWindow::on_saltnoise_pb_clicked()
{
    if (!isThereImage() || SaltNoiseNum<0)
        return;

    saltnoise_image = ui->label_image->pixmap()->toImage();
    cv::Mat saltnoise_mat_in = My_QImage2Mat(saltnoise_image);
    cv::Mat saltnoise_mat_out;

    saltnoise_mat_out = addGaussianNoise(saltnoise_mat_in, GaussNoiseMu, GaussNoiseSigma);
    saltnoise_image = My_Mat2QImage(saltnoise_mat_out);

    show_image_now(saltnoise_image);
}





//单目摄像机标定
void MainWindow::on_monoCalibration_pb_clicked()
{
    ifstream fin("/home/vtie/LJX_OpenCV/calibration/imageDatalist_left.txt"); /* 标定所用图像文件的路径 */
    ofstream fout("/home/vtie/LJX_OpenCV/calibration/caliberation_result_left.txt");  /* 保存标定结果的文件 */
    string distFilePathInBase = "/home/vtie/LJX_OpenCV/calibration/left";
    string distFilePathOutBase = "/home/vtie/LJX_OpenCV/calibration/chess";

    int image_count = 0;  /* 图像数量 */
    Size image_size;  /* 图像的尺寸 */
    Size board_size = Size(9, 6);    /* 标定板上每行、列的角点数 */
    int CornerNum = board_size.width * board_size.height;  //每张图片上总的角点数
    vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
    vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
    string filename;

    while (getline(fin, filename))
    {
        image_count++;
        filename.erase(filename.size()-1);
        cout<<"Now is No."<<image_count<<" image."<<endl;

        Mat imageInput = imread(filename);

        if (image_count == 1)  //读入第一张图片时获取图像宽高信息
        {
            image_size.width = imageInput.cols;
            image_size.height = imageInput.rows;
            cout<<"image_size.width = "<<image_size.width<<endl;
            cout<<"image_size.height = "<<image_size.height<<"\n"<<endl;
        }

        if (imageInput.rows<=0 || imageInput.cols<=0)
        {
            cout<<"Error: Can not get image!"<<endl; //找不到image
            waitKey(0);
            exit(1);
        }

        // 如果寻找到所有的角点, 则返回1, 并得到角点在图像坐标系下的像素坐标, 否则返回0
        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
        {
            cout<<"Error: Can not find chessboard corners!"<<endl; //找不到角点
            waitKey(0);
            exit(1);
        }
        else
        {
            Mat view_gray;
            cvtColor(imageInput, view_gray, COLOR_RGB2GRAY);

            cv::find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化, Need try to get the param of Size().

            image_points_seq.push_back(image_points_buf);  //保存亚像素角点
                                                           /* 在图像上显示角点位置 */
            cv::drawChessboardCorners(view_gray, board_size, image_points_buf, false); //用于在图片中标记角点
            imshow("Camera Calibration", view_gray);//显示图片
            waitKey(50);//暂停0.5S
        }
    }

    int total = image_points_seq.size();
    cout<<"We totally SUCCESSFULLY get "<<total<<"calibration images."<<endl;
    cout<<"Start Calibration Now……"<<endl;

    Size square_size = Size(10, 10);  /* 实际测量得到的标定板上每个棋盘格的大小, mm */
    vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
    /*内外参数*/
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
    vector<int> point_counts;  // 每幅图像中角点的数量
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
    vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */

    /* 初始化标定板上角点的三维坐标 */
    int i, j, t;
    for (t = 0; t<image_count; t++)
    {
        vector<Point3f> tempPointSet;
        for (i = 0; i<board_size.height; i++)
        {
            for (j = 0; j<board_size.width; j++)
            {
                Point3f realPoint;
                /* 假设标定板放在世界坐标系中z=0的平面上 */
                realPoint.x = i*square_size.width;
                realPoint.y = j*square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);
    }

    /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
    for (i = 0; i<image_count; i++)
        point_counts.push_back(CornerNum);

    /* 开始标定 */
    cv::calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

    //对标定结果进行评价
    cout<<"Start evaluate the calibration results………"<<endl;
    double total_err = 0.0; /* 所有图像的平均误差的总和 */
    double err = 0.0; /* 每幅图像的平均误差 */
    vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */

    for (i = 0; i<image_count; i++)
    {
        vector<Point3f> tempPointSet = object_points[i];
        /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
        cv::projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

        /* 计算新的投影点和旧的投影点之间的误差*/
        vector<Point2f> tempImagePoint = image_points_seq[i];
        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
        Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
        for (int j = 0; j < static_cast<int>(tempImagePoint.size()); j++)
        {
            image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
            tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err /= point_counts[i];

        cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
    }

    fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;

    //保存定标结果
    cout<<"Start save calibration results………"<<endl;
    Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
    fout << "相机内参数矩阵：" << endl;
    fout << cameraMatrix << endl << endl;
    fout << "畸变系数：\n";
    fout << distCoeffs << endl << endl << endl;
    for (int i = 0; i<image_count; i++)
    {
        fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
        fout << tvecsMat[i] << endl;
        /* 将旋转向量转换为相对应的旋转矩阵 */
        cv::Rodrigues(tvecsMat[i], rotation_matrix);
        fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
        fout << rotation_matrix << endl;
        fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
        fout << rvecsMat[i] << endl << endl;
    }

    fout << endl;

    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);

    string imageFileName;
    stringstream StrStm;
    string distFilePathIn;
    string distFilePathOut;
    for (int i = 0; i != 5; i++)
    {cout<<"ceshi1"<<endl;
        // 使用相同的cameraMatrix作为输入和输出，表示不改变摄像机内参矩阵。使用单位矩阵作为R参数，表示不进行旋转校正
        initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);

        imageFileName.clear();
        StrStm.clear();
        StrStm << i + 1;
        StrStm >> imageFileName;

        distFilePathIn.clear();
        distFilePathIn = distFilePathInBase + imageFileName + ".jpg";

        Mat imageSource = imread(distFilePathIn);
        Mat newimage = imageSource.clone();

        remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
        distFilePathOut.clear();
        distFilePathOut = distFilePathOutBase + imageFileName + "_d.jpg";
        imwrite(distFilePathOut, newimage);
    }

    waitKey(0);
}

/*计算标定板上模块的实际物理坐标*/
void MainWindow::calRealPoint(vector<vector<Point3f>>& obj, int boardWidth, int boardHeight, int imgNumber, int squareSize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++)
    {
        for (int colIndex = 0; colIndex < boardWidth; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}

Mat R, T, E, F;
Mat Rl, Rr, Pl, Pr, Q;
Mat mapLx, mapLy, mapRx, mapRy;
Mat cameraMatrixL = (Mat_<double>(3, 3) << 530.1397548683084, 0, 338.2680507680664,
                                            0, 530.2291152852337, 232.4902023212199,
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.266294943795012, -0.0450330886310585, 0.0003024821418382528, -0.001243865371699451, 0.2973605735168139);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 530.1397548683084, 0, 338.2680507680664,
                                            0, 530.2291152852337, 232.4902023212199,
                                            0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.266294943795012, -0.0450330886310585, 0.0003024821418382528, -0.001243865371699451, 0.2973605735168139);

void MainWindow::outputCameraParam(void)
{
    /*保存数据*/
    /*输出数据*/
    FileStorage fs("/home/vtie/LJX_OpenCV/calibration/stereo_intrisics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!" << endl;
    }

    fs.open("/home/vtie/LJX_OpenCV/calibration/stereo_extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr" << Rr << endl << "Pl" << Pl << endl << "Pr" << Pr << endl << "Q" << Q << endl;
        fs.release();
    }
    else
    {
        cout << "Error: can not save the extrinsic parameters\n";
    }
}

//双目摄像机标定
void MainWindow::on_stereoCalibration_pb_clicked()
{
    ifstream finL("/home/vtie/LJX_OpenCV/calibration/imageDatalist_left.txt"); /* 标定所用图像文件的路径 */
    ifstream finR("/home/vtie/LJX_OpenCV/calibration/imageDatalist_right.txt"); /* 标定所用图像文件的路径 */

    //摄像头的分辨率
    const int imageWidth = 640;
    const int imageHeight = 480;
    //横向的角点数目
    const int boardWidth = 9;
    //纵向的角点数目
    const int boardHeight = 6;
    //总的角点数目

    //相机标定时需要采用的图像帧数
    const int totalImageNum = 14;
    //标定板黑白格子的大小 单位是mm
    const int squareSize = 10;
    //标定板的总内角点尺寸
    const Size boardSize = Size(boardWidth, boardHeight);
    //像素尺寸
    Size imageSize = Size(imageWidth, imageHeight);


    //R旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
    vector<Mat> rvecs; //R
    vector<Mat> tvecs; //T
    //左边摄像机所有照片角点的坐标集合
    vector<vector<Point2f>> imagePointL;
    //右边摄像机所有照片角点的坐标集合
    vector<vector<Point2f>> imagePointR;
    //各图像的角点的实际的物理坐标集合
    vector<vector<Point3f>> objRealPoint;
    //左边摄像机某一照片角点坐标集合
    vector<Point2f> cornerL;
    //右边摄像机某一照片角点坐标集合
    vector<Point2f> cornerR;

    Mat rgbImageL, grayImageL;
    Mat rgbImageR, grayImageR;
    Mat intrinsic;
    Mat distortion_coeff;
    //校正旋转矩阵R，投影矩阵P，重投影矩阵Q
    //映射表
    Mat mapLx, mapLy, mapRx, mapRy;
    Rect validROIL, validROIR;
    //图像校正之后，会对图像进行裁剪，其中，validROI裁剪之后的区域

    Mat img;
    int goodFrameCount = 1;
    while (goodFrameCount <= totalImageNum)
    {
        string filenameL;
        string filenameR;

        /*读取左边的图像*/
        getline(finL,filenameL);
        getline(finR,filenameR);
        filenameL.erase(filenameL.size()-1);
        filenameR.erase(filenameR.size()-1);
        rgbImageL = imread(filenameL);
        rgbImageR = imread(filenameR);
        cout<<"Now is No."<<goodFrameCount+1<<" Left & Right images."<<endl;

        if (rgbImageL.rows<=0 || rgbImageL.cols<=0 || rgbImageR.rows<=0 || rgbImageR.cols<=0)
        {
            cout<<"Error: Can not get image!"<<endl; //找不到image
            waitKey(0);
            exit(1);
        }

        imshow("chessboardL", rgbImageL);
        imshow("chessboardR", rgbImageR);
        cvtColor(rgbImageR, grayImageR, COLOR_RGB2GRAY);
        cvtColor(rgbImageL, grayImageL, COLOR_RGB2GRAY);

        bool isFindL, isFindR;
        isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
        isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);

        if (isFindL == true && isFindR == true)
        {
            cv::find4QuadCornerSubpix(grayImageL, cornerL, Size(5, 5)); //对粗提取的角点进行精确化, Need try to get the param of Size().
//            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
            imshow("chessboardL", rgbImageL);
            imagePointL.push_back(cornerL);

            cv::find4QuadCornerSubpix(grayImageR, cornerR, Size(5, 5)); //对粗提取的角点进行精确化, Need try to get the param of Size().
//            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
            imshow("chessboardR", rgbImageR);
            imagePointR.push_back(cornerR);

            goodFrameCount++;
        }
        else
        {
            cout<<"Error: Can not find chessboard corners!"<<endl; //找不到角点
            waitKey(0);
            exit(1);
        }
    }

    //计算实际的校正点的三维坐标，根据实际标定格子的大小来设置

    calRealPoint(objRealPoint, boardWidth, boardHeight, totalImageNum, squareSize);

    //标定摄像头
    //R – 输出第一和第二相机坐标系之间的旋转矩阵
    //T – 输出第一和第二相机坐标系之间的平移向量
    //E – 输出本征矩阵
    //F – 输出基础矩阵
    double rms = stereoCalibrate(   objRealPoint, imagePointL, imagePointR,
                                    cameraMatrixL, distCoeffL,
                                    cameraMatrixR, distCoeffR,
                                    Size(imageWidth, imageHeight), R, T, E, F, CALIB_USE_INTRINSIC_GUESS,
                                    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    cout<<"Stereo Calibration done with RMS error = "<<rms<<endl;

    //R - 左摄像头和右摄像头之间的旋转矩阵
    //T - 左摄像头和右摄像头之间的平移向量
    //Rl Rr - 分别表示左摄像头和右摄像头的旋转矩阵，这些矩阵将各自的摄像头坐标系对齐到世界坐标系。
    //Pl Pr - 分别表示左摄像头和右摄像头的投影矩阵，这些矩阵描述了从世界坐标系到图像坐标系的投影。
    //Q - 一个用于重投影的附加矩阵，它结合了旋转和平移，并且可以用于执行附加的仿射变换
    //validROIL validROIR - 输出参数，表示经过校正后，左/右相机的有效视场（即可以用于立体匹配的区域）
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize,
                  R, T, Rl, Rr, Pl, Pr, Q,
                  CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);

    //摄像机校正映射
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    Mat rectifyImageL, rectifyImageR;
    cvtColor(grayImageL, rectifyImageL, COLOR_GRAY2RGB);
    cvtColor(grayImageR, rectifyImageR, COLOR_GRAY2RGB);

    imshow("RecitifyL Before", rectifyImageL);
    imshow("RecitifyR Before", rectifyImageR);

    //经过remap之后，左右相机的图像已经共面并且行对准了
    Mat rectifyImageL2, rectifyImageR2;
    remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
    remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);

    imshow("rectifyImageL Rectified", rectifyImageL2);
    imshow("rectifyImageR Rectified", rectifyImageR2);

    outputCameraParam();

    //显示校正结果
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(0, 0, w, h));
    cv::resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA); //缩小图像
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf), cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));
    cv::resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR); //缩小图像
    Rect vroiR(cvRound(validROIR.x*sf), cvRound(validROIR.y*sf), cvRound(validROIR.width*sf), cvRound(validROIR.height*sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);


    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

    imshow("rectified", canvas);



    // BM-Disparity
    Mat leftImage = imread("/home/vtie/LJX_OpenCV/calibration/left06.jpg", IMREAD_GRAYSCALE);
    Mat rightImage = imread("/home/vtie/LJX_OpenCV/calibration/right06.jpg", IMREAD_GRAYSCALE);

    if (!leftImage.data || !rightImage.data)
    {
        std::cerr << "Error: Could not load image files." << std::endl;
        return;
    }

    // 设置BM算法状态
    int numDisparities = 32;
    int blockSize = 15;
    Ptr<StereoBM> bm =StereoBM::create(numDisparities, blockSize);
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numDisparities * 7 + 16);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(10);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);

    // 计算视差图
    Mat disparity;
    bm->compute(leftImage, rightImage, disparity);
    disparity.convertTo(disparity, CV_32F, 1.0 / 16);

    // 显示视差图
    namedWindow("Disparity", WINDOW_NORMAL);
    imshow("Disparity", disparity);

    waitKey(0);
}

void MainWindow::on_HSmerge_pb_clicked()
{
//    if (!isThereImage())
//        return;

    // 1. 读取RGB图像
//    QImage HSmerge_img = ui->label_image->pixmap()->toImage();
    cv::Mat rgb_image = cv::imread("/home/vtie/LJX_OpenCV/images/straight_lines2.jpg");
//    cv::Mat rgb_image = cv::imread("/home/vtie/LJX_OpenCV/images/test6.jpg");
//    cv::Mat rgb_image = My_QImage2Mat(HSmerge_img);
    if (rgb_image.empty()) {
        std::cout << "无法读取图像" << std::endl;
        return;
    }

    // 2. 转换为HLS图像
    cv::Mat hls_image;
    cv::cvtColor(rgb_image, hls_image, cv::COLOR_BGR2HLS);

    // 3. 提取H通道
    cv::Mat h_channel = cv::Mat::zeros(hls_image.rows, hls_image.cols, CV_8UC1);
    for (int i = 0; i < hls_image.rows; i++) {
        for (int j = 0; j < hls_image.cols; j++) {
            cv::Vec3b pixel = hls_image.at<cv::Vec3b>(i, j);
            h_channel.at<uchar>(i, j) = pixel[0];
        }
    }

    // 4. 对H通道进行Sobel边缘检测
    cv::Mat sobel_h_channel;
    cv::Sobel(h_channel, sobel_h_channel, CV_8U, 0, 1, 3);

    // 5. 提取S通道
    cv::Mat s_channel = cv::Mat::zeros(hls_image.rows, hls_image.cols, CV_8UC1);
    for (int i = 0; i < hls_image.rows; i++) {
        for (int j = 0; j < hls_image.cols; j++) {
            cv::Vec3b pixel = hls_image.at<cv::Vec3b>(i, j);
            s_channel.at<uchar>(i, j) = pixel[1];
        }
    }

    // 6. 对S通道进行阈值分割
    cv::Mat binary_s_channel;
    cv::threshold(s_channel, binary_s_channel, 220, 255, cv::THRESH_BINARY);

    // 7. 将两个通道融合
    cv::Mat HS_result = cv::Mat::zeros(hls_image.rows, hls_image.cols, CV_8UC1);
//    cv::bitwise_and(sobel_h_channel, binary_s_channel, HS_result);
    HS_result = sobel_h_channel + binary_s_channel;

    // 8. 显示HS-merge结果
    cv::imshow("HSmerge_Result", HS_result);
//    cv::imshow("HSmerge_Results", sobel_h_channel);
//    cv::imshow("HSmerge_Resultb", binary_s_channel);

    // 10.29: 9.透视变换
    Point2f srcQuad[4], dstQuad[4];
    Mat warp_mat(3, 3, CV_32FC1);

    srcQuad[0]=Point2f(HS_result.cols/2-63, HS_result.rows/2+100);
    srcQuad[1]=Point2f(HS_result.cols/6-20, HS_result.rows);
    srcQuad[2]=Point2f(HS_result.cols*5/6+60, HS_result.rows);
    srcQuad[3]=Point2f(HS_result.cols/2+65, HS_result.rows/2+100);

    dstQuad[0]=Point2f(HS_result.cols/4, 0);
    dstQuad[1]=Point2f(HS_result.cols/4, HS_result.rows);
    dstQuad[2]=Point2f(HS_result.cols*3/4, HS_result.rows);
    dstQuad[3]=Point2f(HS_result.cols*3/4, 0);

    Mat aftWarpPerspective = Mat::zeros(HS_result.rows, HS_result.cols, HS_result.type());
//    warpPerspectiveBase(srcQuad, dstQuad, HS_result, aftWarpPerspective);

    warp_mat = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(HS_result, aftWarpPerspective, warp_mat, HS_result.size());

    for (int r=0; r<aftWarpPerspective.rows; r++){
        for (int c=0; c<aftWarpPerspective.cols; c++){
            if(aftWarpPerspective.at<uchar>(r,c)>=200)
                aftWarpPerspective.at<uchar>(r,c) = 255;
            else
                aftWarpPerspective.at<uchar>(r,c) = 0;
        }
    }
    cv::imshow("aftWarpPerspective_Result", aftWarpPerspective);

    //10.
    vector<int> num_of_cols(aftWarpPerspective.cols, 0);

    for (int r=0; r<aftWarpPerspective.rows; r++){
        for (int c=0; c<aftWarpPerspective.cols; c++){
            if (aftWarpPerspective.at<uchar>(r,c) == 255) // 255=white
                num_of_cols[c]++;
        }
    }

    vector<int> max_num_of_cols(2, -1);
    vector<int> temp_max(2, 0);
    int DoNotFindNextMaxColThrd = 200;

    for (int index_of_max=0; index_of_max<2; index_of_max++){
        for (int i=0; i<aftWarpPerspective.cols; i++){
            if (index_of_max==1 &&
                (i>=max_num_of_cols[0]-DoNotFindNextMaxColThrd && i<=max_num_of_cols[0]+DoNotFindNextMaxColThrd)
                )
                continue;
            if (num_of_cols[i] > temp_max[index_of_max])
            {
                temp_max[index_of_max] = num_of_cols[i];
                max_num_of_cols[index_of_max] = i;
            }
        }
    }
//    qDebug()<<"col: "<<max_num_of_cols[0]<<" "<<max_num_of_cols[1];
//    qDebug()<<"is: "<<temp_max[0]<<" "<<temp_max[1];

    int window_width = 9;
    int window_height = 21;
    int window_num = int(aftWarpPerspective.rows/window_height);
    int useful_points_num_thrd = int(window_width*window_height*0.5);

    #define poly_n 3 // jieci^
    double a[2][poly_n+1] = {{0.0}};  // dai qiu xi shu

    for (int now_col=0; now_col<=1; now_col++){
        int poly_num = 0;
        vector<double> poly_x(window_num, 0.0); // x---row
        vector<double> poly_y(window_num, 0.0); // y---col

        for (int now_row=window_height/2; now_row+window_height/2<aftWarpPerspective.rows; now_row+=window_height){
            int useful_points_num = 0;
            int sum_num=0;
            int sum_x = 0;
            int sum_y = 0;
            int arv_x = 0;
            int arv_y = 0;
            for (int i=now_row-window_height/2; i<=now_row+window_height/2; i++){
                for (int j=max_num_of_cols[now_col]-window_width/2; j<=max_num_of_cols[now_col]+window_width/2; j++){
                    if(aftWarpPerspective.at<uchar>(i,j) != 0){
                        useful_points_num++;
                        sum_num++;
                        sum_x += i;
                        sum_y += j;
                    }
                }
            }
            if (useful_points_num <= useful_points_num_thrd)
                continue;

            arv_x = sum_x/sum_num;
            arv_y = sum_y/sum_num;
            poly_x[poly_num] = arv_x;
            poly_y[poly_num] = arv_y;
            poly_num++;
        }
        poly_x.erase(poly_x.begin()+poly_num, poly_x.end());
        poly_y.erase(poly_y.begin()+poly_num, poly_y.end());
        polyfit(poly_num, poly_x, poly_y, poly_n, a[now_col]);
    }

    Mat aftWarpPerspectiveWithLaneLines = aftWarpPerspective.clone(); // CV_8UC1
    cvtColor(aftWarpPerspectiveWithLaneLines, aftWarpPerspectiveWithLaneLines, COLOR_GRAY2BGR);
    Mat laneLines = Mat::zeros(aftWarpPerspective.rows, aftWarpPerspective.cols, CV_8UC3);
    cv::Vec3b green(0, 255, 0);

    Mat laneArea = Mat::zeros(laneLines.rows, laneLines.cols, CV_8UC3);

    for (int now_row=0; now_row<aftWarpPerspectiveWithLaneLines.rows; now_row++){
        int real_col_1 = 0;
        int real_col_2 = 0;
        for(int i=0; i<=poly_n; i++) {
            real_col_1 = real_col_1 + a[0][i] * std::pow(now_row, i);
            real_col_2 = real_col_2 + a[1][i] * std::pow(now_row, i);
        }
        cv::Point pixel_position1(real_col_1, now_row);
        cv::Point pixel_position2(real_col_2, now_row);
        aftWarpPerspectiveWithLaneLines.at<cv::Vec3b>(pixel_position1) = green;
        aftWarpPerspectiveWithLaneLines.at<cv::Vec3b>(pixel_position2) = green;
        laneLines.at<cv::Vec3b>(pixel_position1) = green;
        laneLines.at<cv::Vec3b>(pixel_position2) = green;
        for (int i=min(real_col_1,real_col_2); i<=max(real_col_1,real_col_2); i++)
            laneArea.at<cv::Vec3b>(cv::Point(i, now_row)) = green;
    }
    imshow("aftWarpPerspectiveWithLaneLines", aftWarpPerspectiveWithLaneLines);
    imshow("laneLines", laneLines);

    //11.
    Mat laneLinesAftWarpPerspective = Mat::zeros(rgb_image.size(), rgb_image.type());
    Mat laneAreaAftWP =  Mat::zeros(rgb_image.size(), rgb_image.type());
    Mat rgbImageWithLaneLines = rgb_image.clone();

    warp_mat = getPerspectiveTransform(dstQuad, srcQuad);
    warpPerspective(laneLines, laneLinesAftWarpPerspective, warp_mat, laneLines.size());
    warpPerspective(laneArea, laneAreaAftWP, warp_mat, laneArea.size());

    imshow("laneLinesAftWarpPerspective", laneLinesAftWarpPerspective);
//    imshow("laneAreaAftWP", laneAreaAftWP);

    addWeighted(laneLinesAftWarpPerspective, 0.5, rgbImageWithLaneLines, 1, 0, rgbImageWithLaneLines);
    imshow("rgbImageWithLaneLines", rgbImageWithLaneLines);

    Mat rgbImageWithLaneLinesAndArea = rgbImageWithLaneLines.clone();
    addWeighted(laneAreaAftWP, 0.5, rgbImageWithLaneLines, 1, 0, rgbImageWithLaneLinesAndArea);
    imshow("rgbImageWithLaneLinesAndArea", rgbImageWithLaneLinesAndArea);


    //12. hough:
    Mat gray_hough;
    Mat rgb_hough = rgb_image.clone();
    vector<Vec2f> lines;

    cvtColor(rgb_image, gray_hough, COLOR_BGR2GRAY);
    cv::Canny(gray_hough, gray_hough, 50, 100, 3);
    HoughLines(gray_hough, lines, 1, CV_PI/180, 260, 0, 0, 0.3*CV_PI, 0.7*CV_PI); // 250 is better
    for (size_t i=0; i<lines.size(); i++){
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;

        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        double k;
        double k_max = 0.50, k_min = -0.50;
        k = double(pt1.y-pt2.y)/double(pt1.x-pt2.x);

        if (k<=k_min || k>=k_max){
            line(rgb_hough, pt1, pt2, Scalar(0,255,0), 1, LINE_AA);
//             qDebug()<<":: "<<pt1.y-pt2.y<<" "<<pt1.x-pt2.x<<" "<<double(pt1.y-pt2.y)/double(pt1.x-pt2.x)<<" "<<k;
        }
    }

    imshow("rgb_hough", rgb_hough);
    cv::waitKey(0);
    return;
}




void MainWindow::on_ORB_pb_clicked()
{
    cv::Mat obj = cv::imread("/home/vtie/LJX_OpenCV/images/obj.jpg");
    cv::Mat scene = cv::imread("/home/vtie/LJX_OpenCV/images/scene.jpg");

    if (obj.empty() || scene.empty()) {
        std::cout << "无法读取图像" << std::endl;
        return;
    }

    vector<KeyPoint> obj_keypoints, scene_keypoints;
    Mat obj_des, scene_des;
    Ptr<ORB> detector = ORB::create();

    detector->detect(obj, obj_keypoints);
    detector->detect(scene, scene_keypoints);
    detector->compute(obj, obj_keypoints, obj_des);
    detector->compute(scene, scene_keypoints, scene_des);

    BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch> matches;
    matcher.match(obj_des, scene_des, matches);

    Mat matchImgBadly;
    drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, matchImgBadly);
    imshow("matchImgBadly", matchImgBadly);

    vector<int> queryIdex(matches.size()), trainIdex(matches.size());
    for (size_t i=0; i<matches.size(); i++){
        queryIdex[i] = matches[i].queryIdx;
        trainIdex[i] = matches[i].trainIdx;
    }

    Mat H;

    vector<Point2f> points1, points2;
    KeyPoint::convert(obj_keypoints, points1, queryIdex);
    KeyPoint::convert(scene_keypoints, points2, trainIdex);
    int ransacReprojThrd = 5;

    H = findHomography(Mat(points1), Mat(points2), RANSAC, ransacReprojThrd);

    vector<char> matchesMask(matches.size(), 0);
    Mat points1_trans;

    perspectiveTransform(Mat(points1), points1_trans, H);
    for (size_t i=0; i<points1.size(); i++){
        if (norm(points2[i] - points1_trans.at<Point2f>((int)i,0)) <= ransacReprojThrd)
            matchesMask[i] = 1;
    }

    Mat matchImgGoodly;
    drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, matchImgGoodly, Scalar::all(-1),Scalar::all(-1), matchesMask);
    imshow("matchImgGoodly", matchImgGoodly);

    vector<Point2f> obj_corners(4);
    vector<Point2f> scene_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(obj.cols, 0);
    obj_corners[2] = Point(obj.cols, obj.rows);
    obj_corners[3] = Point(0, obj.rows);
    perspectiveTransform(obj_corners, scene_corners, H);

    Point2f src_point[4], dst_point[4];
    src_point[0].x = scene_corners[0].x;
    src_point[0].y = scene_corners[0].y;
    src_point[1].x = scene_corners[1].x;
    src_point[1].y = scene_corners[1].y;
    src_point[2].x = scene_corners[2].x;
    src_point[2].y = scene_corners[2].y;
    src_point[3].x = scene_corners[3].x;
    src_point[3].y = scene_corners[3].y;
    dst_point[0].x = 0;
    dst_point[0].y = 0;
    dst_point[1].x = obj.cols;
    dst_point[1].y = 0;
    dst_point[2].x = obj.cols;
    dst_point[2].y = obj.rows;
    dst_point[3].x = 0;
    dst_point[3].y = obj.rows;

    Mat newM(3, 3, CV_32FC1);
    newM = getPerspectiveTransform(src_point, dst_point);

    Mat sceneAftWP = scene.clone();
    warpPerspective(scene, sceneAftWP, newM, obj.size());
    imshow("sceneAftWP", sceneAftWP);

    Mat result = sceneAftWP.clone();
    absdiff(obj, sceneAftWP, result);
    imshow("result", result);

    return;
}
