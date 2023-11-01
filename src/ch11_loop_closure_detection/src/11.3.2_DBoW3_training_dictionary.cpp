/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       11.3.2_DBoW3_training_dictionary.cpp
 * @author     Speike
 * @date       2023/10/30 17:57:51
 * @brief      使用DBow3从十张图像中创建小型字典，字典的创建使用BRIEF描述子
 *             字典树形结构使用分支数量k=10，深度L=5, 字典的容纳单词数为10^5=100000个。
 *             DBoW库不支持c++17
 **/

#include <DBoW3/DBoW3.h>

#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main()
{
    // step 1：读取图像
    cout << "1. 读取图像..." << endl;
    vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i)
    {
        string path =
            "./src/ch11_loop_closure_detection/data/" + to_string(i + 1) + ".png";
        cv::Mat image = cv::imread(path);
        if (image.data == nullptr)
        {
            cerr << "文件\"" << path << "\"读取失败，当前路径为"
                 << experimental::filesystem::current_path() << endl;
            return EXIT_FAILURE;
        }
        images.push_back(image);
    }
    // step 2: 检测ORB特征
    cout << "2. 提取ORB特征..." << endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    vector<cv::Mat> descriptors;
    for (cv::Mat& image : images)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }
    // step 3: 创建字典
    cout << "3. 使用描述子创建字典..." << endl << endl;
    // 创建字典
    DBoW3::Vocabulary vocab;
    // 生成字典
    vocab.create(descriptors);  // 默认构造，k=10, l=5 //QUERY 使用描述子？
    cout << "vocabulary info: " << vocab << endl;
    // 保存字典
    vocab.save("./src/ch11_loop_closure_detection/data/vocabulary.yml.gz");
    cout << "\ndone! 字典保存在./src/ch11_loop_closure_detection/data/vocabulary.yml.gz"
         << endl;

    return EXIT_SUCCESS;
}
