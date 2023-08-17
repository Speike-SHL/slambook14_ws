/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       7.6.1_triangulation.cpp
 * @author     Speike  <shao-haoluo@foxmail.com>
 * @date       2023/08/17 10:58:24
 * @brief      在7.4节估计完运动的基础上，用三角化估计实际尺度
**/

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
using namespace std;

/**
 * @brief 上上节的内容，使用opencv进行orb特征点的提取和匹配
*/
void opencv_feature_matches(const cv::Mat &img_1, 
                            const cv::Mat &img_2, 
                            vector<cv::KeyPoint> &keypoints_1, 
                            vector<cv::KeyPoint> &keypoints_2, 
                            vector<cv::DMatch> &good_matches)
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    vector<cv::DMatch> matches;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

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
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if(matches[i].distance <= std::max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }
    cout << "一共找到了" << good_matches.size() << "组匹配点" << endl;
}


/**
 * @brief 上节的内容，使用opencv自带的函数分别求取基础矩阵F、本质矩阵E和单应矩阵H
*/
void opencv_cal_FEH(vector<cv::KeyPoint> &keypoints_1,
                    vector<cv::KeyPoint> &keypoints_2,
                    vector<cv::DMatch> &good_matches,
                    cv::Mat &R, cv::Mat &t, cv::Mat &E)
{
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points_1, points_2;
    for (int i = 0; i < (int)good_matches.size(); i++)
    {
        points_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵F, 8点法
    cv::Mat Fundamental_matrix = cv::findFundamentalMat(points_1, points_2, CV_FM_8POINT);
    cout << "基础矩阵为: " << endl
         << Fundamental_matrix << endl
         << endl;

    //-- 计算本质矩阵, 使用相机内参或相机光心和焦距都行
    // 相机内参,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    cv::Mat Essential_matrix_1 = cv::findEssentialMat(points_1, points_2, K);

    cv::Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    cv::Mat Essential_matrix_2 = cv::findEssentialMat(points_1, points_2, focal_length, principal_point);

    E = Essential_matrix_1;
    cout << "本质矩阵为(使用内参矩阵): " << endl
         << Essential_matrix_1 << endl;
    cout << "本质矩阵为(使用相机光心与焦距): " << endl
         << Essential_matrix_2 << endl
         << endl;

    //-- 计算单应矩阵, 但是本例中场景不是平面，单应矩阵意义不大
    cv::Mat Homography_matrix = cv::findHomography(points_1, points_2, cv::RANSAC, 3);
    cout << "单应矩阵为: " << endl
         << Homography_matrix << endl
         << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose(Essential_matrix_1, points_1, points_2, K, R, t);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
    cout << "||t|| = " << sqrt(t.at<double>(2, 0) * t.at<double>(2, 0) + t.at<double>(1, 0) * t.at<double>(1, 0) + t.at<double>(0, 0) * t.at<double>(0, 0)) << "，在分解时，通常把t进行归一化，即让其模长为1" << endl << endl;
}

/**
 * @brief 使用opencv提供的函数进行相邻两帧间的三角化，得到所有特征点的3d坐标
*/
void triangulation (vector<cv::KeyPoint> &keypoints_1,
                    vector<cv::KeyPoint> &keypoints_2,
                    vector<cv::DMatch> &good_matches,
                    cv::Mat &R, cv::Mat &t,
                    vector<cv::Point3d> &points)
{
    // 以第一个相机为参考系，所以T1为[I3,0]
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 
                1, 0, 0, 0,
                0, 1, 0, 0,    
                0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<  //将由对极几何求出的(R，t)组成T21
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    //相机内参矩阵K
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  

    // 取出像素坐标
    vector<cv::Point2f> pt1, pt2;
    for (cv::DMatch m:good_matches)
    {
        pt1.push_back(keypoints_1[m.queryIdx].pt);
        pt2.push_back(keypoints_2[m.trainIdx].pt);
    }
    // 转为归一化坐标，课本公式5.5
    vector<cv::Point2f> x1, x2;
    for (int i = 0; i < (int)pt1.size(); i++)
    {
        
        x1.push_back( cv::Point2f(
            (pt1[i].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (pt1[i].y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        ));
        x2.push_back( cv::Point2f(
            (pt2[i].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (pt2[i].y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        ));
    }

    // 使用opencv的函数进行三角化, 为什么三角化函数返回值为4d, 见博客
    // https://blog.csdn.net/weixin_43956164/article/details/124266267
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, x1, x2, pts_4d);

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) 
    {   
        //pts_4d为4*n的矩阵；pts_4d.cols为匹配点的个数
        cv::Mat pt_4d = pts_4d.col(i);
        pt_4d /= pt_4d.at<float>(3, 0); //除以第四维进行归一化
        cv::Point3d p(  //得到三维点坐标
            pt_4d.at<float>(0, 0),
            pt_4d.at<float>(1, 0),
            pt_4d.at<float>(2, 0)
        );
        points.push_back(p);
    }
}

///  作图用
inline cv::Scalar get_color(float depth) 
{
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    //像素点越远，红色越深
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range)); 
}

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

    //-- 使用opencv进行特征点提前与匹配
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> good_matches;
    opencv_feature_matches(img_1, img_2, keypoints_1, keypoints_2, good_matches);

    //-- 计算基础矩阵F、本质矩阵E和单应矩阵H
    cv::Mat R, t, E;
    opencv_cal_FEH(keypoints_1, keypoints_2, good_matches, R, t, E);

    // -- 三角化，得到特征点在相机坐标系1下的3D坐标
    vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, good_matches, R, t, points);

    //-- 验证三角化点与特征点的重投影关系
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    cv::Mat img1_plot = img_1.clone();
    cv::Mat img2_plot = img_2.clone();
    for (int i = 0; i < (int)good_matches.size(); i++) 
    {
        // 第一个图中画出关键点并打印深度
        float depth1 = points[i].z;

        // 第二个图，根据相机坐标系1中的三维点转化到相机坐标系2中的三维点坐标获得深度，然后用不同颜色标在图中
        cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);

        cout << "point " << setw(2) << i <<" 在图1中depth: " << setw(7) <<depth1 <<", 在图2中depth: " << depth2 <<endl;
        cv::circle(img1_plot, keypoints_1[good_matches[i].queryIdx].pt, 2, get_color(depth1), 2);
        cv::circle(img2_plot, keypoints_2[good_matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::namedWindow("img 1", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("img 2", cv::WINDOW_KEEPRATIO);
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey(0);
    return EXIT_SUCCESS;
}
