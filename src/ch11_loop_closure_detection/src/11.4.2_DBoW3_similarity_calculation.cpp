/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       11.4.2_DBoW3_similarity_calculation.cpp
 * @author     Speike
 * @date       2023/11/01 13:51:16
 * @brief      使用11.3.2生成的小字典进行相似度计算并检测回环是否发生
 *             相似度计算可以图像与图像间对比，也可以进行数据库查询
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
    // step 1: 读取图像
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
    // step 2: 读取字典
    cout << "2. 读取字典..." << endl;
    string vocab_path = "./src/ch11_loop_closure_detection/data/vocabulary.yml.gz";
    // 读取字典
    DBoW3::Vocabulary vocab(vocab_path);
    if (vocab.empty())
    {
        cerr << "字典文件" << vocab_path << "不存在" << endl;
        return EXIT_FAILURE;
    }
    // step 3: 检测ORB特征点和描述子
    cout << "3. 提取ORB特征点和描述子..." << endl << endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    vector<cv::Mat> descriptors;
    for (cv::Mat &image : images)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    cout << "描述向量举例：描述向量中含有图像在词典中的单词ID和权重：" << endl;
    DBoW3::BowVector v1;                  // 创建图像的BoW向量描述
    vocab.transform(descriptors[0], v1);  // 把某个图像的描述子转为字典的BoW向量
    cout << "图像1在词典中的描述向量：" << v1 << endl << endl;

    // step 4: 图像间两两计算评分
    cout << "4. 图像间两两对比..." << endl;
    for (size_t i = 0; i < images.size(); ++i)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (size_t j = i; j < images.size(); ++j)
        {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            // 计算两个向量对比评分
            double score = vocab.score(v1, v2);
            cout << "图片" << i << " vs 图片" << j << " : " << score << endl;
        }
        cout << endl;
    }

    // step 5: 建立基于字典的数据库并查询相似度评分
    cout << "5. 建立基于字典的数据库并查询相似度前4的图像及相似度评分..." << endl;
    DBoW3::Database db(vocab, false, 0);
    for (size_t i = 0; i < descriptors.size(); ++i)
    {
        db.add(descriptors[i]);
    }
    cout << "database info: " << db << endl << endl;
    for (size_t i = 0; i < descriptors.size(); ++i)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);  // 查询与当前图像相似度前四的图像和评分
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "done." << endl;

    // NOTE 改变cout的输出颜色
    cout << "\033[32m"
         << "为什么人眼认为最相似的图像1.png(0)和图像10.png(9),但是相似度评分只要0.05,"
            "即5%？"
         << "\033[0m" << endl;

    cout << "\033[34m"
         << "可能由于字典规模太小，增大字典规模可以突出相似图像，缩小不相似图像评分。"
         << "但是往往还需要对评分进行归一化处理，见11.5.2节"
         << "\033[0m" << endl;
    return EXIT_SUCCESS;
}
