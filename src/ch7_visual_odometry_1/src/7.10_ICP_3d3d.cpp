/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       7.10_ICP_3d3d.cpp
 * @author     Speike
 * @date       2023/08/31 21:36:28
 * @brief      分别使用SVD法、G2O上的非线性优化、CERES上的非线性优化等方法求解ICP问题
 **/

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Dense>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>

#include "tic_toc.h"

using namespace std;
float fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

struct COST_FUNCTION
{
    COST_FUNCTION(Eigen::Vector3d P1, Eigen::Vector3d P2) : _P1(P1), _P2(P2) {}
    template <typename T>
    bool operator()(const T *const estimate, T *residual) const
    {
        T P1[3];
        T P2_esti[3];
        P1[0] = T(_P1[0]);
        P1[1] = T(_P1[1]);
        P1[2] = T(_P1[2]);
        ceres::AngleAxisRotatePoint(estimate, P1, P2_esti);
        P2_esti[0] += estimate[3];
        P2_esti[1] += estimate[4];
        P2_esti[2] += estimate[5];
        residual[0] = T(_P2[0]) - T(P2_esti[0]);
        residual[1] = T(_P2[1]) - T(P2_esti[1]);
        residual[2] = T(_P2[2]) - T(P2_esti[2]);
        return true;
    }

private:
    const Eigen::Vector3d _P1, _P2;
};

void CERES_ICP(vector<cv::Point3f> pts1_3d, vector<cv::Point3f> pts2_3d)
{
    cout << "3. CERES求解非线性优化ICP.\n" << endl;
    // 构建最小二乘问题
    double params[6] = {0, 0, 0, 0, 0, 0};
    ceres::Problem problem;
    for (int i = 0; i < static_cast<int>(pts1_3d.size()); i++)
    {
        // 添加残差块, ceres会迭代到残差块取最小值
        problem.AddResidualBlock(
            // 自动求导，<误差类型，输出残差维度，输入带估计参数维度>
            new ceres::AutoDiffCostFunction<COST_FUNCTION, 3, 6>(new COST_FUNCTION(
                Eigen::Vector3d(pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z),
                Eigen::Vector3d(pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z))),
            nullptr,  // 核函数，这里不使用，为空
            params    // 待估计参数
        );
    }
    // 配置求解器开始求解
    ceres::Solver::Options options;  // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_SCHUR;  // 求解H*dx = g的方法
    options.minimizer_progress_to_stdout = true;      // 输出到cout
    ceres::Solver::Summary summary;                   // 求解器的摘要
    TicToc time;
    ceres::Solve(options, &problem, &summary);
    Sophus::Vector6d ksi;
    // sophus平移在前，旋转在后
    ksi << params[3], params[4], params[5], params[0], params[1], params[2];
    Sophus::SE3d T = Sophus::SE3d::exp(ksi);
    cout << "\n--> ceres-solver求解耗时: " << time.toc() << "ms. 估计结果为:\n"
         << T.matrix() << endl;
    cout << "ceres-solver求解报告为: " << endl;
    cout << summary.BriefReport() << endl;
}

// 顶点类, <顶点即待优化变量维度, 顶点类型>
class myVertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myVertex() = default;
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &out) const override { return true; }
    void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
    void oplusImpl(const double *update) override
    {
        Eigen::Vector<double, 6> dx;
        dx << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(dx) * _estimate;
    }
};

class myEdge : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, myVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myEdge(Eigen::Vector3d P1) : _P1(P1) {}
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &out) const override { return true; }
    void computeError() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Eigen::Vector3d P2_esti = T * _P1;
        _error = _measurement - P2_esti;
    }
    virtual void linearizeOplus() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Eigen::Vector3d P2_esti = T * _P1;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(P2_esti);
    }

private:
    Eigen::Vector3d _P1;
};

/**
 * @brief 使用非线性优化方法求解ICP问题
 */
void G2O_ICP(vector<cv::Point3f> pts1_3d, vector<cv::Point3f> pts2_3d)
{
    // 优化变量为位姿T的李代数为6维, 误差值三维点坐标为3维
    using BlockSolverType = g2o::BlockSolverPL<6, 3>;
    // 线性求解器类型
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    // 创建总求解器--GN法
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);  // 图模型
    optimizer.setVerbose(true);      // 打开调试输出

    // 往图中添加顶点
    myVertex *v = new myVertex();
    v->setId(0);
    v->setEstimate(Sophus::SE3d());
    optimizer.addVertex(v);

    // 往图中增加边
    for (size_t i = 0; i < pts1_3d.size(); i++)
    {
        // const Eigen::Vector3d P1{pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z};
        // const Eigen::Vector3d P2{pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z};
        Eigen::Vector3d P1, P2;
        P1 << pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z;
        P2 << pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z;
        myEdge *edge = new myEdge(P1);
        edge->setId(int(i));
        edge->setVertex(0, v);
        edge->setMeasurement(P2);
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    // 设置优化参数并开始优化
    TicToc time;
    cout << "2. G2O求解非线性优化ICP.\n" << endl;
    optimizer.initializeOptimization();  // 初始化
    optimizer.optimize(5);  // 设置迭代次数,设置几次迭代几次
    cout << "\n--> G2O求解耗时: " << time.toc() << "ms. 估计结果为:\n"
         << v->estimate().matrix() << endl;
}

/**
 * @brief 使用opencv进行orb特征点的提取和匹配
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
 * @brief 使用SVD方法求解ICP问题
 */
void SVD_ICP(vector<cv::Point3f> pts1_3d, vector<cv::Point3f> pts2_3d)
{
    TicToc time;
    cout << "1. SVD分解求ICP.\n" << endl;
    // 求两组点的质心坐标
    cv::Point3f p1, p2;
    int N = pts1_3d.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1_3d[i];
        p2 += pts2_3d[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    // 求两组点的去质心坐标
    vector<cv::Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1_3d[i] - p1;
        q2[i] = pts2_3d[i] - p2;
    }
    // 由两组点集计算 W = 求和(q1*q2^T)
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) *
             Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W 矩阵为: \n" << W << endl;
    // 对W进行奇异值SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U 矩阵为: \n" << U << endl;
    cout << "V 矩阵为: \n" << V << endl;
    // 求取R
    Eigen::Matrix3d R = U * V.transpose();
    if (R.determinant() < 0) R = -R;
    // 求取t
    Eigen::Vector3d t =
        Eigen::Vector3d(p1.x, p1.y, p1.z) - R * Eigen::Vector3d(p2.x, p2.y, p2.z);
    cout << "--> SVD求解ICP: " << time.toc() << "ms. 估计结果为:\n"
         << "R 矩阵为:\n"
         << R << endl
         << "t 矩阵为:\n"
         << t << endl
         << "R_inv 为:\n"
         << R.inverse() << endl
         << "t_inv 为:\n"
         << -R.transpose() * t << endl;
    cout << "估计结果的逆才为R21, 结果与3D2D "
            "PnP的结果还是有一定的差距,\n理论上我们在PnP->ICP的过程中, "
            "使用了越来越多的数据,\n精度也会得到相应的提高,"
            "不过由于Kinect的深度图存在噪声,\n可能会导致ICP精度变低."
         << endl;
}

int main()
{
    //-- 读取图像
    cv::Mat img_1 =
        cv::imread("./src/ch7_visual_odometry_1/data/1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 =
        cv::imread("./src/ch7_visual_odometry_1/data/2.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat imgdepth_1 = cv::imread("./src/ch7_visual_odometry_1/data/1_depth.png",
                                    CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imgdepth_2 = cv::imread("./src/ch7_visual_odometry_1/data/2_depth.png",
                                    CV_LOAD_IMAGE_UNCHANGED);
    if (img_1.data == nullptr || img_2.data == nullptr ||
        imgdepth_1.data == nullptr || imgdepth_2.data == nullptr)
    {
        cerr << "文件"
             << "./src/ch7_visual_odometry_1/data/*.png"
             << "读取失败，当前路径为" << experimental::filesystem::current_path()
             << endl;
        return EXIT_FAILURE;
    }
    //-- 使用opencv进行特征点提取与匹配
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> good_matches;
    opencv_feature_matches(img_1, img_2, keypoints_1, keypoints_2, good_matches);
    //-- 准备3d3d点
    vector<cv::Point3f> pts1_3d, pts2_3d;
    for (auto &m : good_matches)
    {
        cv::Point2f pt1 = keypoints_1[m.queryIdx].pt;  // 图1 2d匹配点
        cv::Point2f pt2 = keypoints_2[m.trainIdx].pt;  // 图2 2d匹配点
        ushort d1 = imgdepth_1.ptr<ushort>(int(pt1.y))[int(pt1.x)];
        ushort d2 = imgdepth_2.ptr<ushort>(int(pt2.y))[int(pt2.x)];
        if (d1 == 0 || d2 == 0) continue;
        float Z1 = d1 / 5000.0;             // 获取图1点深度
        float X1 = (pt1.x - cx) / fx * Z1;  // 像素坐标转3D坐标
        float Y1 = (pt1.y - cy) / fy * Z1;
        pts1_3d.emplace_back(cv::Point3f(X1, Y1, Z1));
        float Z2 = d2 / 5000.0;             // 获取图2点深度
        float X2 = (pt2.x - cx) / fx * Z2;  // 像素坐标转3D坐标
        float Y2 = (pt2.y - cy) / fy * Z2;
        pts2_3d.emplace_back(cv::Point3f(X2, Y2, Z2));
    }
    cout << "建立的3d-3d点对有: " << pts1_3d.size() << "对" << endl;
    cout << "=======================================================" << endl;
    // SVD法求解ICP问题
    SVD_ICP(pts1_3d, pts2_3d);
    cout << "=======================================================" << endl;
    // G2O非线性优化求ICP
    G2O_ICP(pts1_3d, pts2_3d);
    cout << "=======================================================" << endl;
    // CERES非线性优化求ICP
    CERES_ICP(pts1_3d, pts2_3d);
    return EXIT_SUCCESS;
}
