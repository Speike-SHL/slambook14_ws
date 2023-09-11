/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       8.5_direct_method.cpp
 * @author     Speike
 * @date       2023/09/08 17:52:02
 * @brief      OpenCV没有直接支持直接法, 使用单层和多层直接法。
 *             由于点过多, 误差过大，优化时没法累加矩阵GN，因此使用SUM(H)deltax = SUM(b)。
 *             但是这一节直接法的效果感觉并不好,从图上就能看出来。
 **/

#include <Eigen/Dense>
#include <boost/format.hpp>
#include <chrono>
#include <experimental/filesystem>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "tic_toc.h"
using namespace std;

using VecEigenV2d =
    vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

double baseline = 0.573;
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

// 双线性插值求像素对应灰度
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
                 (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

/**
 * @brief 定义一个类，并行加速累加雅可比和Error
 */
class JacobianAccumulator
{
public:
    JacobianAccumulator(const cv::Mat &img1_, const cv::Mat &img2_,
                        const VecEigenV2d &pixels_ref_,
                        const vector<double> &depth_ref_, Sophus::SE3d &T21_)
        : img1(img1_),
          img2(img2_),
          pixels_ref(pixels_ref_),
          depth_ref(depth_ref_),
          T21(T21_)
    {
        pixels_projection =
            VecEigenV2d(pixels_ref.size(), Eigen::Vector2d(0, 0));  // 初始化投影点
    }

    /// 在range范围内加速计算雅可比矩阵
    void calResidualandJacobian(const cv::Range &range);

    Matrix6d hessian() const { return H; }
    Vector6d bias() const { return b; }
    double cost_func() const { return cost; }
    VecEigenV2d projected_points() const { return pixels_projection; }
    void reset()
    {
        pixels_projection = VecEigenV2d(pixels_ref.size(), Eigen::Vector2d(0, 0));
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecEigenV2d &pixels_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecEigenV2d pixels_projection;  // projected points

    std::mutex hessian_mutex;  // 线程锁
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
    const int half_patch_size = 1;
};

/// 加速计算Jacobian，注释见ipad p223
void JacobianAccumulator::calResidualandJacobian(const cv::Range &range)
{
    int cnt_good = 0;  // 判断对应线程是否计算成功, 还是从continue结束了
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;
    for (int i = range.start; i < range.end; i++)
    {
        // 由随机选取的像素点p1和深度, 还原出相机1下的3D点P1
        Eigen::Vector3d P1 =
            depth_ref[i] * Eigen::Vector3d((pixels_ref[i][0] - cx) / fx,
                                           (pixels_ref[i][1] - cy) / fy, 1);
        // 求出P2, 若深度小于零, 跳过
        Eigen::Vector3d P2 = T21 * P1;
        if (P2[2] < 0) continue;
        // 由P2求出图像2中对应的像素p2, 并判断p2是否落在图像中
        double u = fx * P2[0] / P2[2] + cx;
        double v = fy * P2[1] / P2[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;
        pixels_projection[i] = Eigen::Vector2d(u, v);
        // 为求雅可比准备一些变量
        double X = P2[0], Y = P2[1], Z = P2[2], Z_inv = 1.0 / Z,
               Z2_inv = Z_inv * Z_inv;
        cnt_good++;
        // 求像素点附近patch内的信息
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                // 误差f
                double f =
                    GetPixelValue(img1, pixels_ref[i][0] + x, pixels_ref[i][1] + y) -
                    GetPixelValue(img2, u + x, v + y);
                // 雅可比J_u_xi
                Matrix26d J_u_xi;
                J_u_xi(0, 0) = fx * Z_inv;
                J_u_xi(0, 1) = 0;
                J_u_xi(0, 2) = -fx * X * Z2_inv;
                J_u_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_u_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_u_xi(0, 5) = -fx * Y * Z_inv;
                J_u_xi(1, 0) = 0;
                J_u_xi(1, 1) = fy * Z_inv;
                J_u_xi(1, 2) = -fy * Y * Z2_inv;
                J_u_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_u_xi(1, 4) = fy * X * Y * Z2_inv;
                J_u_xi(1, 5) = fy * X * Z_inv;
                // 雅可比J_I2_u
                Eigen::Matrix<double, 1, 2> J_I2_u;
                J_I2_u = Eigen::Matrix<double, 1, 2>(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) -
                           GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) -
                           GetPixelValue(img2, u + x, v - 1 + y)));
                // 总雅可比
                Eigen::Matrix<double, 1, 6> Jf = -1.0 * J_I2_u * J_u_xi;
                // 累加求单个点patch内的H和b
                hessian += Jf.transpose() * Jf;
                bias += -Jf.transpose() * f;
                cost_tmp += f * f;
            }
    }
    if (cnt_good)
    {
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        // 多线程处理，一个线程可能累加了多个点的cost，取个均值
        cost += cost_tmp / cnt_good;
    }
}

/**
 * 单层直接法的位姿估计，代码注释见ipad p225
 * @param [in] img1 图像1
 * @param [in] img2 图像2
 * @param [in] pixels_ref 图像1中选取的参考像素点
 * @param [in] depth_ref 参考像素点的深度
 * @param [out] T21 图像1到图像2的变换
 */
void DirectMethodSingleLayer(const cv::Mat &img1, const cv::Mat &img2,
                             const VecEigenV2d &pixels_ref,
                             const vector<double> &depth_ref, Sophus::SE3d &T21,
                             cv::Mat &img2_show)
{
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    TicToc time;
    JacobianAccumulator jaco_accu(img1, img2, pixels_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, pixels_ref.size()),
                          std::bind(&JacobianAccumulator::calResidualandJacobian,
                                    &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            cout << "更新 is nan, 失败" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost)
        {
            cout << "Iter: " << iter << ", cost: " << cost
                 << " >= last cost: " << lastCost << ", 迭代发散" << endl;
            break;
        }
        if (update.norm() < 1e-3)
        {
            cout << "更新足够小,结束迭代" << endl;
            break;
        }

        lastCost = cost;
        if (iter % (iterations / 10) == 0)
            cout << "Iter: " << iter << ", cost: " << cost << endl;
    }

    cout << fixed << setprecision(5) << "迭代结束, 单层直接法耗时: " << time.toc()
         << "ms. T21= \n"
         << T21.matrix() << endl;

    // 绘制图像
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecEigenV2d pixels_projection = jaco_accu.projected_points();
    for (size_t i = 0; i < pixels_ref.size(); ++i)
    {
        auto p_ref = pixels_ref[i];
        auto p_cur = pixels_projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2,
                       cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]),
                     cv::Point2f(p_cur[0], p_cur[1]), cv::Scalar(0, 250, 0));
        }
    }
    cv::namedWindow("current", CV_WINDOW_NORMAL);
    cv::resizeWindow("current", cv::Size(img2_show.cols * 2, img2_show.rows * 2));
    cv::imshow("current", img2_show);
    cv::waitKey(500);
}

/**
 * 多层直接法的位姿估计，代码注释见ipad p226
 */
void DirectMethodMultiLayer(const cv::Mat &img1, const cv::Mat &img2,
                            const VecEigenV2d &pixels_ref,
                            const vector<double> depth_ref, Sophus::SE3d &T21,
                            vector<cv::Mat> &result_imgs)
{
    // 金字塔参数
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 建立金字塔
    vector<cv::Mat> pyr1, pyr2;  // 图像1和2各自的图像金字塔
    for (int i = 0; i < pyramids; i++)
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

    // 备份旧的相机内参
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;
    for (int level = pyramids - 1; level >= 0; level--)
    {
        cout << "-------------层" << 4 - level << "------------" << endl;
        // 得到对应层中图像1中带有深度的点像素坐标
        VecEigenV2d pixels_ref_pyr;
        for (auto &px : pixels_ref)
        {
            pixels_ref_pyr.push_back(scales[level] * px);
        }

        // NOTE 注意直接法用到了相机内参，所以要缩放
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        cv::Mat tmp_img;
        // NOTE 光流金字塔由粗到精得到下一层中点的位置，
        // NOTE
        // 而直接法金字塔则得到下一层中T21的初始值，由于是引用传递，直接能不断迭代
        DirectMethodSingleLayer(pyr1[level], pyr2[level], pixels_ref_pyr, depth_ref,
                                T21, tmp_img);
        result_imgs.push_back(tmp_img);
    }
}

int main()
{
    // INFO-- 读取left.png作为参考图像, 同时读取视差图
    cv::Mat ref_img = cv::imread("./src/ch8_visual_odometry_2/data/left.png",
                                 CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat disparity_img = cv::imread(
        "./src/ch8_visual_odometry_2/data/disparity.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (ref_img.data == nullptr || disparity_img.data == nullptr)
    {
        cerr << "文件"
             << "./src/ch8_visual_odometry_2/data/left(or)disparity.png"
             << "读取失败，当前路径为" << experimental::filesystem::current_path()
             << endl;
        return EXIT_FAILURE;
    }

    // INFO-- 从参考图像中随机选取一些像素点并恢复深度获得三维坐标
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecEigenV2d pixels_ref;
    vector<double> depth_ref;
    for (int i = 0; i < nPoints; i++)
    {
        // 随机选取x和y, 可能会重复, 不过无所谓
        int x = rng.uniform(boarder, ref_img.cols - boarder);
        int y = rng.uniform(boarder, ref_img.rows - boarder);
        int disparity = disparity_img.ptr<uchar>(y)[x];
        double depth = fx * baseline / disparity;  // 课本p104 式5.15disparity是视差
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // INFO-- 开始分别使用单层直接法和多层直接法进行求解 T
    boost::format format_path("./src/ch8_visual_odometry_2/data/%06d.png");
    Sophus::SE3d T_cur_ref;  // T_cur_ref由参考图像到当前图像的变换
    for (int i = 1; i < 6; i++)
    {
        cv::Mat img = cv::imread((format_path % i).str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat result_img;
        cout << "\n>>>>>>>>>>>>>>>>>> 1. 单层直接法: <<<<<<<<<<<<<<<<<<<" << endl;
        DirectMethodSingleLayer(ref_img, img, pixels_ref, depth_ref, T_cur_ref,
                                result_img);
    }

    vector<cv::Mat> result_imgs;
    for (int i = 1; i < 6; i++)
    {
        cv::Mat img = cv::imread((format_path % i).str(), CV_LOAD_IMAGE_GRAYSCALE);
        cout << "\n>>>>>>>>>>>>>>>>>> 2. 多层直接法: <<<<<<<<<<<<<<<<<<<" << endl;
        DirectMethodMultiLayer(ref_img, img, pixels_ref, depth_ref, T_cur_ref,
                               result_imgs);
    }
    // 展示
    cv::destroyAllWindows();
    cv::Mat img000001, img000002, img000003, img000004;
    img000001.push_back(result_imgs[0]);
    img000001.push_back(result_imgs[4]);
    img000001.push_back(result_imgs[8]);
    img000001.push_back(result_imgs[12]);
    img000001.push_back(result_imgs[16]);
    img000002.push_back(result_imgs[1]);
    img000002.push_back(result_imgs[5]);
    img000002.push_back(result_imgs[9]);
    img000002.push_back(result_imgs[13]);
    img000002.push_back(result_imgs[17]);
    img000003.push_back(result_imgs[2]);
    img000003.push_back(result_imgs[6]);
    img000003.push_back(result_imgs[10]);
    img000003.push_back(result_imgs[14]);
    img000003.push_back(result_imgs[18]);
    img000004.push_back(result_imgs[3]);
    img000004.push_back(result_imgs[7]);
    img000004.push_back(result_imgs[11]);
    img000004.push_back(result_imgs[15]);
    img000004.push_back(result_imgs[19]);
    cv::namedWindow("1 floor", CV_WINDOW_NORMAL);
    cv::resizeWindow("1 floor", cv::Size(img000001.cols * 1.5, img000001.rows * 1.5));
    cv::imshow("1 floor", img000001);
    cv::namedWindow("2 floor", CV_WINDOW_NORMAL);
    cv::resizeWindow("2 floor", cv::Size(img000002.cols * 1.5, img000002.rows * 1.5));
    cv::imshow("2 floor", img000002);
    cv::namedWindow("3 floor", CV_WINDOW_NORMAL);
    cv::resizeWindow("3 floor", cv::Size(img000003.cols * 1.5, img000003.rows * 1.5));
    cv::imshow("3 floor", img000003);
    cv::namedWindow("4 floor", CV_WINDOW_NORMAL);
    cv::resizeWindow("4 floor", cv::Size(img000004.cols * 1.5, img000004.rows * 1.5));
    cv::imshow("4 floor", img000004);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
