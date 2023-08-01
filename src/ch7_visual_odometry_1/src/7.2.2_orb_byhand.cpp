/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       7.2.2_orb_byhand.cpp
 * @author     Speike
 * @date       2023/08/01 20:43:57
 * @brief      手写orb特征点的提取与匹配
**/

#include <iostream>
#include <filesystem>
#include "tic_toc.h"
#include <nmmintrin.h>
#include <opencv2/opencv.hpp>
using namespace std;

void OrientedFAST();
void RotatedBRIEF();
void ComputeORB();
void BruteForceMatch();

int main()
{
    //-- 读取图像
    cv::Mat img_1 = cv::imread("./src/ch7_visual_odometry_1/data/1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread("./src/ch7_visual_odometry_1/data/2.png", CV_LOAD_IMAGE_COLOR);
    if(img_1.data == nullptr || img_2.data == nullptr)
    {
        cerr << "文件" << "./src/ch7_visual_odometry_1/data/(1或2).png" << "读取失败，当前路径为" << filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }
}
