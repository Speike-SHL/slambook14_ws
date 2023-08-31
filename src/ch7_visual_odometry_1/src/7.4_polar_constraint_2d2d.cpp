/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       7.4_polar_constrains_2d2d.cpp
 * @author     Speike  <shao-haoluo@foxmail.com>
 * @date       2023/08/16 16:43:32
 * @brief      使用opencv进行两张图像间的orb特征提取与匹配，
 *             同时使用opencv中自带的方法求取基础矩阵F和本质矩阵E，
 **/

#include <experimental/filesystem>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tic_toc.h"
using namespace std;

/**
 * @brief 上一节的内容，使用opencv进行orb特征点的提取和匹配
 */
void opencv_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                            vector<cv::KeyPoint> &keypoints_1,
                            vector<cv::KeyPoint> &keypoints_2,
                            vector<cv::DMatch> &good_matches)
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    vector<cv::DMatch> matches;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步：检测Oriented FAST角点
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步：根据角点计算Rotated BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步：对Rotated BRIEF描述子进行匹配
    matcher->match(descriptors_1, descriptors_2, matches);

    //-- 第四步：匹配点对的筛选
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const cv::DMatch &m1, const cv::DMatch &m2)
                                  { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }
    cout << "一共找到了" << good_matches.size() << "组匹配点" << endl;
}

/**
 * @brief 使用opencv自带的函数分别求取基础矩阵F、本质矩阵E和单应矩阵H
 */
void opencv_cal_FEH(vector<cv::KeyPoint> &keypoints_1,
                    vector<cv::KeyPoint> &keypoints_2,
                    vector<cv::DMatch> &good_matches, cv::Mat &R, cv::Mat &t,
                    cv::Mat &E)
{
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points_1, points_2;
    for (int i = 0; i < (int)good_matches.size(); i++)
    {
        points_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵F, 8点法
    cv::Mat Fundamental_matrix =
        cv::findFundamentalMat(points_1, points_2, CV_FM_8POINT);
    cout << "基础矩阵为: " << endl << Fundamental_matrix << endl << endl;
    ;

    //-- 计算本质矩阵, 使用相机内参或相机光心和焦距都行
    // 相机内参,TUM Freiburg2
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    cv::Mat Essential_matrix_1 = cv::findEssentialMat(points_1, points_2, K);

    cv::Point2d principal_point(325.1, 249.7);  // 相机光心, TUM dataset标定值
    double focal_length = 521;                  // 相机焦距, TUM dataset标定值
    cv::Mat Essential_matrix_2 =
        cv::findEssentialMat(points_1, points_2, focal_length, principal_point);

    E = Essential_matrix_1;
    cout << "本质矩阵为(使用内参矩阵): " << endl << Essential_matrix_1 << endl;
    cout << "本质矩阵为(使用相机光心与焦距): " << endl
         << Essential_matrix_2 << endl
         << endl;
    ;

    //-- 计算单应矩阵, 但是本例中场景不是平面，单应矩阵意义不大
    cv::Mat Homography_matrix =
        cv::findHomography(points_1, points_2, cv::RANSAC, 3);
    cout << "单应矩阵为: " << endl << Homography_matrix << endl << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose(Essential_matrix_1, points_1, points_2, K, R, t);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
    cout << "||t|| = "
         << sqrt(t.at<double>(2, 0) * t.at<double>(2, 0) +
                 t.at<double>(1, 0) * t.at<double>(1, 0) +
                 t.at<double>(0, 0) * t.at<double>(0, 0))
         << "，在分解时，通常把t进行归一化，即让其模长为1" << endl
         << endl;
}

int main()
{
    //-- 读取图像
    cv::Mat img_1 =
        cv::imread("./src/ch7_visual_odometry_1/data/1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 =
        cv::imread("./src/ch7_visual_odometry_1/data/2.png", CV_LOAD_IMAGE_COLOR);
    if (img_1.data == nullptr || img_2.data == nullptr)
    {
        cerr << "文件"
             << "./src/ch7_visual_odometry_1/data/(1或2).png"
             << "读取失败，当前路径为" << experimental::filesystem::current_path()
             << endl;
        return EXIT_FAILURE;
    }

    //-- 使用opencv进行特征点提前与匹配
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> good_matches;
    opencv_feature_matches(img_1, img_2, keypoints_1, keypoints_2, good_matches);

    //-- 计算基础矩阵F、本质矩阵E和单应矩阵H
    cv::Mat R, t, E;
    opencv_cal_FEH(keypoints_1, keypoints_2, good_matches, R, t, E);

    //-- 验证本质矩阵 E=t^R*scale
    cv::Mat t_hat = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0),
                     t.at<double>(1, 0), t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                     -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout << "t^R =" << endl << t_hat * R << endl;
    cout << "t^R =/= E ?" << endl;
    cv::Scalar meanA = cv::mean(t_hat * R);
    cv::Scalar meanB = cv::mean(E);
    double scale = meanB[0] / meanA[0];
    cout << "scale= " << scale << endl;
    cout << "t^R*scale =" << endl << t_hat * R * scale << endl;
    cout << "t^R*scale == E ?" << endl << endl;

    //-- 验证对极约束 p2^{T} * K^{-T} * t^R * K^{-1} * p1
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < int(good_matches.size()); i++)
    {
        cv::Point2d pt1 = keypoints_1[good_matches[i].queryIdx].pt;
        cv::Point2d pt2 = keypoints_2[good_matches[i].trainIdx].pt;
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cout << "第" << i
             << "个匹配点p1和p2的对极约束p2^{T} * K^{-T} * t^R * K^{-1} * p1 = "
             << p2.t() * K.t().inv() * t_hat * R * K.inv() * p1 << "约等于零"
             << endl;
    }
}
