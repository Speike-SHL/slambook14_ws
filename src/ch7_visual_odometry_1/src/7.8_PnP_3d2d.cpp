/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * --------------------------------------------------------------------------
 */

/**
 * @file       7.8_PnP_3d2d.cpp
 * @author     Speike
 * @date       2023/08/22 21:31:15
 * @brief      分别使用OpenCV的EPnP, 手写高斯牛顿法解BAPnP,
 *             使用G2O求解BAPnP以及使用Ceres-solver解BAPnP
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
#include <functional>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>

#include "tic_toc.h"

using namespace std;
using namespace cv;

using VecEigenV3d =
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using VecEigenV2d =
    vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

float fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

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
//============================ g2o求解BAPnP =================================//
// 顶点类, <顶点即待优化变量维度, 顶点类型>
class myVertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myVertex() = default;
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &out) const override { return true; }
    // 顶点重置函数，设定被优化变量的原始值。
    void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
    // 顶点更新函数，很重要，用于更新delta_x
    void oplusImpl(const double *update) override
    {
        Eigen::Vector<double, 6> dx;
        dx << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(dx) * _estimate;
    }
};

class myEdge : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, myVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myEdge(const Eigen::Vector3d &P, const Eigen::Matrix3d &K) : _P(P), _K(K) {}
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &out) const override { return true; }
    // 定义误差模型
    void computeError() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Sophus::Vector3d P2 = T * _P;
        double s = P2[2];
        Eigen::Vector2d p2 = ((1 / s) * _K * P2).topRows(2);
        _error = _measurement - p2;
    }
    // 雅可比矩阵
    void linearizeOplus() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Sophus::Vector3d P2 = T * _P;
        double X = P2[0], Y = P2[1], Z = P2[2];
        double X2 = X * X, Y2 = Y * Y, Z2 = Z * Z;
        _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2,
            -fx - fx * X2 / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / Z2,
            fy + fy * Y2 / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

private:
    Eigen::Vector3d _P;
    Eigen::Matrix3d _K;
};

/**
 * @brief 使用g2o进行BAPnP求解
 * @details 该函数使用g2o进行BAPnP求解, 优化变量为位姿T, 3D点P不进行优化,
 *          即节点为位姿T的李代数, 边为观测方程, 误差使用齐次坐标为3维
 */
void BAPnP_G2O(vector<cv::Point3f> &pts_3d, vector<cv::Point2f> &pts_2d)
{
    // 转为Eigen类型的3d-2d点
    VecEigenV3d points_3d;
    VecEigenV2d points_2d;
    for (size_t i = 0; i < pts_3d.size(); i++)
    {
        points_3d.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        points_2d.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    // 相机内参
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    // 优化变量为位姿T的李代数为6维, 误差值像素坐标为2维
    using BlockSolverType = g2o::BlockSolverPL<6, 2>;
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

    // 往图中添加边
    for (size_t i = 0; i < points_3d.size(); i++)
    {
        const auto &p3d = points_3d[i];
        const auto &p2d = points_2d[i];
        myEdge *edge = new myEdge(p3d, K);
        edge->setId(int(i));
        edge->setVertex(0, v);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    // 设置优化参数并开始优化
    TicToc time;
    cout << "3. G2O求解BA-PnP.\n" << endl;
    optimizer.initializeOptimization();  // 初始化
    optimizer.optimize(10);  // 设置迭代次数,设置几次迭代几次
    cout << "\033[33m"
         << "--> G2O求解耗时: " << time.toc() << "ms. 估计结果为:\n"
         << v->estimate().matrix() << "\033[0m" << endl;
}

//======================== ceres-solver求解BAPnP ============================//
struct COST_FUNCTION
{
    COST_FUNCTION(Eigen::Vector2d p2, Eigen::Vector3d P1) : _p2(p2), _P1(P1) {}
    template <typename T>
    bool operator()(const T *const estimate, T *residual) const
    {
        // RP+t 求P2
        T P2[3];
        T point[3];
        point[0] = T(_P1[0]);
        point[1] = T(_P1[1]);
        point[2] = T(_P1[2]);
        ceres::AngleAxisRotatePoint(estimate, point, P2);
        P2[0] += estimate[3];
        P2[1] += estimate[4];
        P2[2] += estimate[5];
        residual[0] = T(_p2[0]) - (T(fx) * P2[0] / P2[2] + T(cx));
        residual[1] = T(_p2[1]) - (T(fy) * P2[1] / P2[2] + T(cy));
        return true;
    }

private:
    const Eigen::Vector2d _p2;
    const Eigen::Vector3d _P1;
};

void BAPnP_CERES(vector<cv::Point3f> &pts_3d, vector<cv::Point2f> &pts_2d)
{
    // 转为Eigen类型的3d-2d点
    VecEigenV3d points_3d;
    VecEigenV2d points_2d;
    for (size_t i = 0; i < pts_3d.size(); i++)
    {
        points_3d.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        points_2d.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    // 相机内参
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    // 构建最小二乘问题
    double params[6] = {0, 0, 0, 0, 0, 0};
    ceres::Problem problem;
    for (int i = 0; i < static_cast<int>(points_3d.size()); i++)
    {
        const auto &p2d = points_2d[i];
        const auto &p3d = points_3d[i];
        // 添加残差块, ceres会迭代到残差块取最小值
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<COST_FUNCTION, 2, 6>(new COST_FUNCTION(
                p2d,
                p3d)),  // 自动求导，<误差类型，输出残差维度，输入带估计参数维度>
            nullptr,  // 核函数，这里不使用，为空
            params    // 待估计参数
        );
    }
    // 配置求解器开始求解
    ceres::Solver::Options options;  // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_SCHUR;  // 求解H*dx = g的方法
    options.minimizer_progress_to_stdout = true;      // 输出到cout

    ceres::Solver::Summary summary;  // 求解器的摘要
    TicToc time;
    ceres::Solve(options, &problem, &summary);
    Sophus::Vector6d ksi;
    // sophus平移在前，旋转在后
    ksi << params[3], params[4], params[5], params[0], params[1], params[2];
    Sophus::SE3d T = Sophus::SE3d::exp(ksi);
    cout << "\033[33m"
         << "--> ceres-solver求解耗时: " << time.toc() << "ms. 估计结果为:\n"
         << T.matrix() << "\033[0m" << endl;
    cout << "ceres-solver求解报告为: " << endl;
    cout << summary.BriefReport() << endl;
}

//===================== 手写高斯牛顿法求解BAPnP相关函数
//=========================//
/**
 * @brief 计算残差e,又叫f,维度为2nx1, n为3D-2D点对的个数
 */
Eigen::VectorXd cal_Residual(VecEigenV3d &points_3d, VecEigenV2d &points_2d,
                             function<Eigen::Vector2d(Eigen::Vector3d)> func)
{
    Eigen::VectorXd residual(points_2d.size() * 2);
    for (size_t i = 0; i < points_2d.size(); i++)
        residual.block<2, 1>(i * 2, 0) = points_2d[i] - func(points_3d[i]);
    return residual;
}

/**
 * @brief 计算雅可比矩阵的转置Jf^T,维度为2nx6, n为3D-2D点对的个数
 */
Eigen::MatrixXd cal_JacobianT(
    VecEigenV3d &points_3d, VecEigenV2d &points_2d,
    function<Eigen::Matrix<double, 2, 6>(Eigen::Vector3d)> Jacobian_func)
{
    Eigen::MatrixXd Jf(points_2d.size() * 2, 6);
    for (size_t i = 0; i < points_2d.size(); i++)
        Jf.block<2, 6>(i * 2, 0) = Jacobian_func(points_3d[i]);
    return Jf;
}

/**
 * @param input points_3d 3D点
 * @param input points_2d 2D点
 * @param input func 估计值函数
 * @param input Jacobian_func 误差函数的Jacobian
 * @param input T 迭代初值
 * @param input iterations 迭代次数
 * @param input learning_rate 学习率
 */
void GaussNewton(VecEigenV3d &points_3d, VecEigenV2d &points_2d,
                 function<Eigen::Vector2d(Sophus::SE3d, Eigen::Vector3d)> func,
                 function<Eigen::Matrix<double, 2, 6>(Sophus::SE3d, Eigen::Vector3d)>
                     JacobianT_func,
                 Sophus::SE3d T, int iterations = 100, double learning_rate = 1)
{
    TicToc time;
    double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost
    for (int iter = 0; iter < iterations; iter++)
    {
        // 计算残差和Jacobian矩阵
        Eigen::VectorXd f = cal_Residual(
            points_3d, points_2d, bind(func, T, placeholders::_1));  // 残差向量
        Eigen::MatrixXd JfT = cal_JacobianT(
            points_3d, points_2d,
            bind(JacobianT_func, T, placeholders::_1));  // Jacobian矩阵
        cost = f.norm();

        // 计算Hessian矩阵和梯度向量
        Eigen::MatrixXd H = JfT.transpose() * JfT;
        Eigen::VectorXd g = -JfT.transpose() * f;

        // 求解线性方程 H*dx = g
        Eigen::VectorXd dx = H.ldlt().solve(g);
        if (isnan(dx[0]))
        {
            cout << "delta_x足够小,结束迭代" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost)
        {
            cout << fixed << setprecision(10) << left << "Iter: " << setw(5) << iter
                 << "cost: " << cost << " >= last cost: " << lastCost << ",迭代发散 "
                 << endl;
            break;
        }
        // 更新参数估计值和cost
        T = Sophus::SE3d::exp(learning_rate * dx) * T;
        lastCost = cost;

        if (iter % (iterations / 10) == 0)
            cout << fixed << setprecision(10) << left << "Iter: " << setw(4) << iter
                 << setw(12) << " total cost: " << setw(18) << cost
                 << "estimated se3: " << fixed << setprecision(5)
                 << T.log().transpose() << endl;
    }

    cout << fixed << "\033[33m"
         << "--> 高斯牛顿法耗时: " << time.toc() << "ms. 估计结果SE3为:\n "
         << T.matrix() << "\033[0m" << endl;
}

/**
 * @brief 手写高斯牛顿法，使用最小二乘法求PnP, 只对位姿优化, 不对3D点进行优化
 * H(6x6) * delta_x(6x1) = g(6x1),
 * H(6x6) = Jf(6x2n) * Jf^T(2nx6),
 * g(6x1) = -Jf(6x2n) * f(2nx1)
 * Jf(6x2n) = [Jf1(6x2), Jf2(6x2), ..., Jfn(6x2)], 课本分子布局，这里分母布局
 * n为3D-2D点对的个数
 */
void BAPnP_GN_byhand(vector<cv::Point3f> &pts_3d, vector<cv::Point2f> &pts_2d)
{
    cout << "2. 手写高斯牛顿法求解BA-PnP.\n" << endl;
    // 转为Eigen类型的3d-2d点
    VecEigenV3d points_3d;
    VecEigenV2d points_2d;
    for (size_t i = 0; i < pts_3d.size(); i++)
    {
        points_3d.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        points_2d.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    // 相机内参
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    // 估计值函数
    function<Eigen::Vector2d(Sophus::SE3d, Eigen::Vector3d)> func =
        [&K](Sophus::SE3d T, Eigen::Vector3d P)
    {
        Sophus::Vector3d P2 = T * P;
        double s = P2[2];
        Eigen::Vector2d p2 = ((1 / s) * K * P2).topRows(2);
        return p2;
    };
    // 对变换矩阵T的雅可比矩阵的转置函数
    function<Eigen::Matrix<double, 2, 6>(Sophus::SE3d, Eigen::Vector3d)>
        JacobianT_func = [](Sophus::SE3d T, Eigen::Vector3d P)
    {
        Sophus::Vector3d P2 = T * P;
        double X = P2[0], Y = P2[1], Z = P2[2];
        double X2 = X * X, Y2 = Y * Y, Z2 = Z * Z;
        Eigen::Matrix<double, 2, 6> Jacobian;
        Jacobian << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X2 / Z2,
            fx * Y / Z, 0, -fy / Z, fy * Y / Z2, fy + fy * Y2 / Z2, -fy * X * Y / Z2,
            -fy * X / Z;
        return Jacobian;
    };
    // 参考6.3.1中的手写高斯牛顿法进行迭代
    GaussNewton(points_3d, points_2d, func, JacobianT_func, Sophus::SE3d(), 10);
}

//====================== opencv内置EPnP求解BAPnP ==========================//
void EPnP_opencv(vector<cv::Point3f> &pts_3d, vector<cv::Point2f> &pts_2d)
{
    cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat R, r, t;
    TicToc time;
    // 还可以选择很多其他方法求解PnP, 注意该方法返回的r为旋转向量
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    // 使用罗德里格斯公式将旋转向量转为旋转矩阵
    cv::Rodrigues(r, R);
    cout << "1. opencv使用EPnP求解耗时: " << time.toc() << "ms.\n" << endl;
    cout << "R =\n" << R << endl;
    cout << "t =\n" << t << endl;

    cv::Matx33d R_2d2d(0.9980710357051413, -0.05456664562625779, 0.02960893229356823,
                       0.05353514702484302, 0.9979670863589692, 0.0345786433686732,
                       -0.03143558047014727, -0.03292682385666658,
                       0.9989632768781929);
    cv::Matx31d t_2d2d(-0.9412055577891585, -0.1569179203170939, 0.2991803206598944);
    cout << "7.4节中通过对极几何求出的结果为:\n"
         << "R =\n"
         << R_2d2d << endl
         << "t =\n"
         << t_2d2d << endl;
    cout << "可以看出R基本一致, 而t相差较大, "
            "因为这里使用深度图的3D点引入了深度信息,\n不过深度相机本身会有误差, "
            "因此在较大规模的BA中, 我们希望把位姿和所有3D点放在一起优化"
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
    if (img_1.data == nullptr || img_2.data == nullptr || imgdepth_1.data == nullptr)
    {
        cerr << "文件"
             << "./src/ch7_visual_odometry_1/data/(1或2).png"
             << "读取失败，当前路径为" << experimental::filesystem::current_path()
             << endl;
        return EXIT_FAILURE;
    }
    //-- 使用opencv进行特征点提取与匹配
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> good_matches;
    opencv_feature_matches(img_1, img_2, keypoints_1, keypoints_2, good_matches);
    //-- 准备3d2d点
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    for (auto &m : good_matches)
    {
        cv::Point2f pt1 = keypoints_1[m.queryIdx].pt;  // 图1 2d匹配点
        cv::Point2f pt2 = keypoints_2[m.trainIdx].pt;  // 图2 2d匹配点
        ushort d = imgdepth_1.ptr<ushort>(int(pt1.y))[int(pt1.x)];
        if (d == 0) continue;
        float Z = d / 5000.0;             // 获取图1点深度
        float X = (pt1.x - cx) / fx * Z;  // 像素坐标转3D坐标
        float Y = (pt1.y - cy) / fy * Z;
        pts_3d.push_back(cv::Point3f(X, Y, Z));
        pts_2d.push_back(pt2);
    }
    cout << "建立的3d-2d点对有: " << pts_3d.size() << "对" << endl;
    cout << "=======================================================" << endl;
    //-- opencv内置函数实现EPnP求解
    EPnP_opencv(pts_3d, pts_2d);
    cout << "=======================================================" << endl;
    //-- 手写高斯牛顿实现PnP求解
    BAPnP_GN_byhand(pts_3d, pts_2d);
    cout << "=======================================================" << endl;
    //-- 使用g2o进行BAPnP求解
    BAPnP_G2O(pts_3d, pts_2d);
    cout << "=======================================================" << endl;
    //-- 使用ceres-solver进行BAPnP求解
    BAPnP_CERES(pts_3d, pts_2d);
    return EXIT_SUCCESS;
}
