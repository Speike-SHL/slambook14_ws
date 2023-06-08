/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       5.3.2_undistortImage.cpp
 * @author     Speike
 * @date       2023/06/07 16:27:20
 * @brief      5.3.2节鱼眼相机图片去畸变
**/

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
using namespace std;

string pic_path = "./src/ch5_camera_model/data/5.3.2_distorted.png";
// 相机内参
double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
// 畸变参数
double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;

void myUndistort(cv::Mat &image_distorted)
{
    // 创建一个去畸变空白图,去畸变图中找对应像素
    cv::Mat image_undistort = cv::Mat(image_distorted.rows, image_distorted.cols, CV_8UC1); 
    for (int v = 0; v < image_undistort.rows; v++)
        for (int u = 0; u < image_undistort.cols; u++)
        {
            // 去畸变图像像素坐标(u,v)-->归一化坐标(x,y)
            double x = (u - cx) / fx, y = (v - cy) / fy;
            // 归一化坐标(x,y)-->计算极坐标r
            double r = sqrt(x * x + y * y);
            // 去畸变图像(x,y,r)-->课本公式5.12-->畸变图中归一化坐标(x_distorted,y_distorted)
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            // 畸变图归一化坐标(x_distorted,y_distorted)-->畸变图像素坐标(u_distorted,v_distorted)
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;
            // 在畸变图中查找(u_distorted,v_distorted)像素赋值到去畸变图(u,v)上
            if(v_distorted >=0 && u_distorted >=0 && v_distorted < image_distorted.rows && u_distorted < image_distorted.cols)
            {
                image_undistort.at<uchar>(v, u) = image_distorted.at<uchar>(int(v_distorted), int(u_distorted));
            }
            else
            {
                image_undistort.at<uchar>(v, u) = 255;
            }
        }

    // 放大两倍显示图像
    cv::Mat image_distorted_show, image_undistort_show;
    cv::resize(image_distorted, image_distorted_show, cv::Size(), 2, 2);
    cv::resize(image_undistort, image_undistort_show, cv::Size(), 2, 2);
    cv::imshow("distortedImage", image_distorted_show);
    cv::imshow("myUndistortImage", image_undistort_show);
    cv::waitKey(0);
}

void myUndistortFull(cv::Mat &image_distorted)
{
    cv::Mat image_undistort = cv::Mat(image_distorted.rows*2, image_distorted.cols*2, CV_8UC1); 
    for (int v = 0-image_undistort.rows/4; v < image_undistort.rows/4*3; v++)
        for (int u = 0-image_undistort.cols/4; u < image_undistort.cols/4*3; u++)
        {
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;
            if(v_distorted >=0 && u_distorted >=0 && v_distorted < image_distorted.rows && u_distorted < image_distorted.cols)
            {
                image_undistort.at<uchar>(v+image_undistort.rows/4, u+image_undistort.cols/4) = image_distorted.at<uchar>(int(v_distorted), int(u_distorted));
            }
            else
            {
                image_undistort.at<uchar>(v+image_undistort.rows/4, u+image_undistort.cols/4) = 255;
            }
        }

    // 放大两倍显示图像
    cv::Mat image_distorted_show, image_undistort_show;
    cv::resize(image_distorted, image_distorted_show, cv::Size(), 2, 2);
    cv::resize(image_undistort, image_undistort_show, cv::Size(), 2, 2);
    cv::imshow("distortedImage", image_distorted_show);
    cv::imshow("myUndistortImage", image_undistort_show);
    cv::waitKey(0);
}

void opencvUndistort(cv::Mat &image_distorted)
{
    cv::Mat image_undistort;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1,4) << k1, k2, p1, p2);
    cv::undistort(image_distorted, image_undistort, cameraMatrix, distCoeffs);
    // 放大两倍显示图像
    cv::Mat image_distorted_show, image_undistort_show;
    cv::resize(image_distorted, image_distorted_show, cv::Size(), 2, 2);
    cv::resize(image_undistort, image_undistort_show, cv::Size(), 2, 2);
    cv::imshow("distortedImage", image_distorted_show);
    cv::imshow("myUndistortImage", image_undistort_show);
    cv::waitKey(0);
}

int main(void)
{
    cv::Mat image = cv::imread(pic_path, CV_8UC1);  //按灰度图读取图像
    if(image.data == nullptr)
    {
        cerr << "图片" << pic_path << "读取失败，当前路径为: " << filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }

    /************************** 自己的去畸变函数 ******************************/
    // 去畸变：前后尺寸一致
    cout << "自己实现去畸变：前后尺寸一致" << endl;
    myUndistort(image);
    // 去畸变：畸变图中所有像素完全展开
    cout << "自己实现去畸变：畸变图中所有像素完全展开" << endl;
    myUndistortFull(image);
    // 使用OpenCV进行去畸变
    cout << "使用OpenCV进行去畸变" << endl;
    opencvUndistort(image);

    return 0;
}
