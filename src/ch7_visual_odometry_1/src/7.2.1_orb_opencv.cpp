/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       7.2.1_orb_opencv.cpp
 * @author     Speike
 * @date       2023/08/01 17:17:42
 * @brief      使用Opencv实现orb特征点提取匹配
 **/

#include "tic_toc.h"
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

int main()
{
    //-- 读取图像
    cv::Mat img_1 = cv::imread("./src/ch7_visual_odometry_1/data/1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread("./src/ch7_visual_odometry_1/data/2.png", CV_LOAD_IMAGE_COLOR);
    if (img_1.data == nullptr || img_2.data == nullptr) {
        cerr << "文件"
             << "./src/ch7_visual_odometry_1/data/(1或2).png"
             << "读取失败，当前路径为" << experimental::filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }

    //-- 初始化
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    vector<cv::DMatch> matches;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步：检测Oriented FAST角点
    TicToc t_orb;
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步：根据角点计算Rotated BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    cout << "提取ORB特征耗时：" << t_orb.toc() << "ms." << endl;

    cv::Mat outimg1, outimg2, outimg;
    // 输入图像，输入图像的关键点，输出图像，关键点颜色(-1表示随机)，关键点标志
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(img_2, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::hconcat(outimg1, outimg2, outimg);
    cv::namedWindow("ORB特征", cv::WINDOW_KEEPRATIO);
    cv::imshow("ORB特征", outimg);

    //-- 第三步：对Rotated BRIEF描述子进行匹配
    TicToc t_match;
    matcher->match(descriptors_1, descriptors_2, matches);
    cout << "特征点匹配耗时：" << t_match.toc() << "ms." << endl;

    //-- 第四步：匹配点对的筛选
    auto min_max = minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }

    //-- 第五步：绘制匹配结果
    cv::Mat img_match, img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::namedWindow("未筛选的匹配", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("筛选后的匹配", cv::WINDOW_KEEPRATIO);
    cv::imshow("未筛选的匹配", img_match);
    cv::imshow("筛选后的匹配", img_goodmatch);
    cv::waitKey(0);
    return EXIT_SUCCESS;
}
