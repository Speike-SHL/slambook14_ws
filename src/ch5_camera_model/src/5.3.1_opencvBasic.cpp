/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       5.3.1_opencvBasic.cpp
 * @author     Speike
 * @date       2023/06/07 14:46:11
 * @brief      OpenCV基本使用方法
**/

#include <ctime>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

string pic_path = "./src/ch5_camera_model/data/5.3.1_ubuntu.png";

class TicToc
{
public:
    TicToc()
    {
        tic();
    }
    inline void tic()
    {
        start = std::chrono::system_clock::now();
    }
    // 返回值的单位是毫秒
    inline double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

int main(void)
{
    // 读取图像
    cv::Mat image;
    image = cv::imread(pic_path);

    // 判断文件是否读取成功
    if(image.data == nullptr)
    {
        cerr << "文件" << pic_path << "读取失败，当前路径为" << filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }

    // 文件顺利读取, 首先输出一些基本信息
    cout << "图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << endl;
    cv::imshow("image_ori", image);      // 用cv::imshow显示图像

    // 判断image的类型
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) 
    {
        // 图像类型不符合要求
        cout << "请输入一张彩色图或灰度图." << endl;
        return EXIT_FAILURE;
    }

    /********************************** 多种方法遍历图像 ************************************/
    // 方法1 image.at
    TicToc t_1;
    for (int v = 0; v < image.rows; v++)
        for (int u = 0; u < image.cols; u++)
            for (int c = 0; c < image.channels(); c++)
                uchar data = image.at<cv::Vec3b>(v, u)[c];
    cout << "方法1遍历图像耗时：" << t_1.toc() << "ms" << endl;
    // 方法2 使用指针
    TicToc t_2;
    for (int v = 0; v < image.rows; v++)
    {
        // 用cv::Mat::ptr获得图像的行指针
        uchar *row_ptr = image.ptr<uchar>(v);   // row_ptr是第y行的头指针
        for (int u = 0; u < image.cols; u++)
        {
            // 访问位于 x,y 处的像素
            uchar *data_ptr = &row_ptr[u * image.channels()];   // data_ptr 指向待访问的像素数据
            // 输出该像素的每个通道,如果是灰度图就只有一个通道
            for (int c = 0; c != image.channels(); c++)
                uchar data = data_ptr[c];  // data为I(x,y)第c个通道的值
        }
    }
    cout << "方法2遍历图像耗时：" << t_2.toc() << "ms" << endl;
    // 方法3 简便版指针,但会一定程度增加耗时
    TicToc t_3;
    for (int v = 0; v < image.rows; v++)
        for (int u = 0; u < image.cols; u++)
            for (int c = 0; c < image.channels(); c++)
                uchar data = image.ptr<uchar>(v)[u * image.channels() + c];
    cout << "方法3遍历图像耗时：" << t_3.toc() << "ms" << endl;

    /*********************************** 关于图像拷贝 *************************************/
    cv::Mat image_another = image;  //浅拷贝
    image_another(cv::Rect(0, 0, 500, 500)).setTo(0);
    cout << "浅拷贝: 修改image_another也会导致image发生改变" << endl;
    cv::imshow("image", image);
    cv::imshow("image_another", image_another);
    cv::waitKey(0);

    cv::Mat image_clone = image.clone();    //深拷贝
    image_clone(cv::Rect(0, 0, 500, 500)).setTo(255);
    cout << "深拷贝: 修改image_clone不会导致image发生改变" << endl;
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}

