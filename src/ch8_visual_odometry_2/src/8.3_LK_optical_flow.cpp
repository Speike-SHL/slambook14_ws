/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       8.3_LK_optical_flow.cpp
 * @author     Speike
 * @date       2023/09/06 10:42:31
 * @brief      OpenCV实现光流法、手写高斯牛顿实现正向光流, 光流金字塔实现反向光流,
 *             同时调用tbb中的parallel_for_并行的计算每个关键点的光流估计
 **/

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "tic_toc.h"

using namespace std;

/// Optical flow tracker and interface
class OpticalFlowTracker
{
public:
    OpticalFlowTracker(
        const cv::Mat &img1_, const cv::Mat &img2_,
        const vector<cv::KeyPoint> &kp1_,  // 图一关键点
        vector<cv::KeyPoint> &kp2_,  //?根据光流计算的图二关键点，也可有初始值
        vector<bool> &success_,  // 布尔类型，输出光流是否计算成功
        bool inverse_ = true, bool has_initial_ = false)
        : img1(img1_),
          img2(img2_),
          kp1(kp1_),
          kp2(kp2_),
          success(success_),
          inverse(inverse_),
          has_initial(has_initial_)
    {
    }

    inline float GetPixelValue(const cv::Mat &img, float x, float y)
    {
        // boundary check 边缘检测
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= img.cols) x = img.cols - 1;
        if (y >= img.rows) y = img.rows - 1;
        uchar *data = &img.data[int(y) * img.step + int(x)];  // 对应点的灰度值
        float xx = x - floor(x);  // 小于等于该值的最大整数
        float yy = y - floor(y);
        return float(  // 双线性插值公式
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
    }

    Eigen::VectorXd calResidual(cv::KeyPoint &kp, double &dx, double &dy)
    {
        Eigen::VectorXd residual(2 * half_patch_size * 2 * half_patch_size);
        int count = 0;
        for (int x = -half_patch_size; x < half_patch_size; x++)
            for (int y = -half_patch_size; y < half_patch_size; y++)
            {
                residual[count] =
                    GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                    GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                count++;
            }
        return residual;
    }

    Eigen::MatrixXd calJacobian(cv::KeyPoint &kp, double &dx, double &dy)
    {
        Eigen::MatrixXd Jf(2 * half_patch_size * 2 * half_patch_size, 2);
        int count = 0;
        for (int x = -half_patch_size; x < half_patch_size; x++)
            for (int y = -half_patch_size; y < half_patch_size; y++)
            {
                Jf.block<1, 2>(count, 0) =
                    -1.0 * Eigen::Matrix<double, 1, 2>(
                               0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1,
                                                    kp.pt.y + dy + y) -
                                      GetPixelValue(img2, kp.pt.x + dx + x - 1,
                                                    kp.pt.y + dy + y)),
                               0.5 * (GetPixelValue(img2, kp.pt.x + dx + x,
                                                    kp.pt.y + dy + y + 1) -
                                      GetPixelValue(img2, kp.pt.x + dx + x,
                                                    kp.pt.y + dy + y - 1)));
                count++;
            }
        return Jf;
    }

    // 反向光流法的雅可比矩阵
    Eigen::MatrixXd calJacobian_inverse(cv::KeyPoint &kp)
    {
        Eigen::MatrixXd Jf(2 * half_patch_size * 2 * half_patch_size, 2);
        int count = 0;
        for (int x = -half_patch_size; x < half_patch_size; x++)
            for (int y = -half_patch_size; y < half_patch_size; y++)
            {
                Jf.block<1, 2>(count, 0) =
                    -1.0 *
                    Eigen::Matrix<double, 1, 2>(
                        0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                               GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                        0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                               GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                count++;
            }
        return Jf;
    }

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const vector<cv::KeyPoint> &kp1;
    vector<cv::KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
    int half_patch_size = 4;
    int iterations = 10;
};

/**
 * 用高斯牛顿法求解最小化灰度误差
 */
void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range)
{
    for (int i = range.start; i < range.end; i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0;  // dx,dy need to be estimated
        if (has_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;  // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::MatrixXd Jf, H;
        Eigen::VectorXd f, g;
        for (int iter = 0; iter < iterations; iter++)
        {
            // 反向光流法只在迭代初始时计算一次H
            if (inverse == true)
            {
                if (iter == 0)
                {
                    Jf = calJacobian_inverse(kp);
                    H = Jf.transpose() * Jf;
                }
                f = calResidual(kp, dx, dy);
                cost = f.squaredNorm();
                g = -Jf.transpose() * f;
            }
            if (inverse == false)
            {
                Jf = calJacobian(kp, dx, dy);
                H = Jf.transpose() * Jf;
                f = calResidual(kp, dx, dy);
                g = -Jf.transpose() * f;
                cost = f.squaredNorm();
            }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(g);

            if (std::isnan(update[0]))
            {
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost)
            {
                // 迭代发散
                succ = false;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2)
            {
                // 增量足够小，迭代停止
                break;
            }
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

/**
 * 单层光流
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse 是否使用逆向光流法
 */
void OpticalFlowSingleLevel(const cv::Mat &img1, const cv::Mat &img2,
                            const vector<cv::KeyPoint> &kp1,
                            vector<cv::KeyPoint> &kp2, vector<bool> &success,
                            bool inverse = false, bool has_initial = false)
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    // 通过cv::parallel_for_并行计算，调用了calculateOpticalFlow函数，循环次数为kp1.size(),传入了tracker数据
    parallel_for_(cv::Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker,
                            placeholders::_1));
}

/**
 * 多层光流，建立图像金字塔，注释见ipad，p215
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(const cv::Mat &img1, const cv::Mat &img2,
                           const vector<cv::KeyPoint> &kp1,
                           vector<cv::KeyPoint> &kp2, vector<bool> &success,
                           bool inverse = false)
{
    int pyramids = 4;  // 金字塔层数
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 建立金字塔
    TicToc time;
    vector<cv::Mat> pyr1, pyr2;         // 图像1和2各自的图像金字塔
    for (int i = 0; i < pyramids; i++)  // 建立图像金字塔
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale,
                                pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale,
                                pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    cout << "建立图像金字塔花费时间：" << time.toc() << "ms." << endl;

    // 由粗至精的光流跟踪
    vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1)  // 获得最顶层的角点坐标
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);  // 最顶层图像1的角点坐标
        kp2_pyr.push_back(kp_top);  // 最顶层图像2的角点坐标：用图像1的初始化图像2的
    }

    for (int level = pyramids - 1; level >= 0; level--)  // 从最顶层开始进行光流追踪
    {
        success.clear();
        time.tic();
        // has_initial设置为true，表示图像2中的角点kp2_pyr进行了初始化
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success,
                               inverse, true);
        cout << "track pyr " << level << " cost time: " << time.toc() << "ms."
             << endl;  // 输出光流跟踪耗时

        if (level > 0)  // 由跟踪结果，对坐标放大倍率变到下一层
        {
            // 因为auto 后是引用的方式，所以直接能够改变kp1_pyr中的值
            for (auto &kp : kp1_pyr)
                kp.pt /= pyramid_scale;  // pyramidScale等于0.5，相当于乘了2
            for (auto &kp : kp2_pyr)
                kp.pt /= pyramid_scale;  // pyramidScale等于0.5，相当于乘了2
        }
    }

    for (auto &kp : kp2_pyr) kp2.push_back(kp);  // 存输出kp2
}

int main()
{
    // 读取图像
    cv::Mat img1 = cv::imread("./src/ch8_visual_odometry_2/data/LK1.png",
                              CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 = cv::imread("./src/ch8_visual_odometry_2/data/LK2.png",
                              CV_LOAD_IMAGE_GRAYSCALE);

    // 提取第一幅图中的关键点, using GFTT here.
    vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector =
        cv::GFTTDetector::create(500, 0.01, 20);  // maximum 500 keypoints
    detector->detect(img1, kp1);

    // 开始在第二幅图中跟踪图一中提取的关键点
    // 1. 使用Opencv的光流方法
    vector<cv::Point2f> pt1, pt2;
    for (auto &kp : kp1)
        pt1.push_back(kp.pt);  // kp1.pt 将图一中关键点的点坐标保存到pt1中
    vector<uchar> status;
    vector<float> error;
    TicToc time;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status,
                             error);  // opencv光流法的函数，成功时status置1
    cout << "1. OpenCV光流法: " << time.toc() << "ms." << endl;

    // 用高斯牛顿法手写光流
    // 2. 首先使用单层LK光流
    vector<cv::KeyPoint> kp2_single;
    vector<bool> success_single;
    time.tic();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    cout << "2. 高斯牛顿单层光流法：" << time.toc() << "ms." << endl;

    // 3. 然后使用多层LK光流, 使用反向光流法
    vector<cv::KeyPoint> kp2_multi;
    vector<bool> success_multi;
    time.tic();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    cout << "3. 高斯牛顿多层光流法: " << time.toc() << "ms." << endl;

    // 将三种方法的结果绘图显示
    cv::Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < (int)pt2.size(); i++)
    {
        if (status[i])
        {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0),
                       2);  // 在第二张图计算的关键点上画圆
            cv::line(img2_CV, pt1[i], pt2[i],
                     cv::Scalar(0, 250, 0));  // 从第一张图关键点到第二张图关键点划线
        }
    }

    cv::Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < (int)kp2_single.size(); i++)
    {
        if (success_single[i])
        {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt,
                     cv::Scalar(0, 250, 0));
        }
    }

    cv::Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < (int)kp2_multi.size(); i++)
    {
        if (success_multi[i])
        {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::namedWindow("单层光流", CV_WINDOW_KEEPRATIO);
    cv::namedWindow("多层光流", CV_WINDOW_KEEPRATIO);
    cv::namedWindow("OpenCV光流", CV_WINDOW_KEEPRATIO);
    cv::imshow("单层光流", img2_single);
    cv::imshow("多层光流", img2_multi);
    cv::imshow("OpenCV光流", img2_CV);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
