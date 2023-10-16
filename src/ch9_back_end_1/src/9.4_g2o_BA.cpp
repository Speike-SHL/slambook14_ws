/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       9.4_g2o_BA.cpp
 * @author     Speike
 * @date       2023/10/16 15:36:44
 * @brief      使用ceres求解大规模BA，同时优化三维点位置、相机位姿、与相机内参f、k1、k2
 *             g2o中自定义顶点的数据类型, 注意最后要把g2o的优化重新输出到对应内存,
 *             需要手动设置二元边中点对应的顶点边缘化, 这里还使用了g2o的自动数值求导
 *
 * BAL数据集网址：https://grail.cs.washington.edu/projects/bal/index.html
 **/

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include "BAL.h"
#include "sophus/se3.hpp"

string bal_data_file = "./src/ch9_back_end_1/data/problem-16-22106-pre.txt";

/**
 * @brief 自定义顶点数据类型
 */
struct PoseAndIntrinsics
{
    PoseAndIntrinsics() = default;

    explicit PoseAndIntrinsics(double *data_addr)
    {
        rotation =
            Sophus::SO3d::exp(Eigen::Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // NOTE 函数名是自己起的, 跟g2o无关, 作用是把g2o优化后的结果重新写到对应的内存中
    void set_to(double *data_addr) const
    {
        auto r = rotation.log();  // 又变成轴角存起来了
        data_addr[0] = r[0];
        data_addr[1] = r[1];
        data_addr[2] = r[2];
        data_addr[3] = translation[0];
        data_addr[4] = translation[1];
        data_addr[5] = translation[2];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0, k1 = 0, k2 = 0;
};

/**
 * @brief 相机顶点类 九维, 三维旋转, 三维平移, f, k1, k2
 * //NOTE 这里通过自定义数据类型实现，应该也可以定义为Eigen::V9d
 */
class CameraVertex : public g2o::BaseVertex<9, PoseAndIntrinsics>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraVertex() = default;
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &in) const override { return true; }

    // 顶点重置函数
    void setToOriginImpl() override { _estimate = PoseAndIntrinsics(); }
    // 顶点更新函数, 连相机内参也要进行更新
    void oplusImpl(const double *update) override
    {
        _estimate.rotation =
            Sophus::SO3d::exp(Eigen::Vector3d(update[0], update[1], update[2])) *
            _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }
    // 将3D路标点转为像素坐标
    Eigen::Vector2d project(const Eigen::Vector3d &point) const
    {
        Eigen::Vector3d P_c = _estimate.rotation * point + _estimate.translation;
        P_c = -P_c / P_c[2];
        double r2 = P_c[0] * P_c[0] + P_c[1] * P_c[1];
        double distortion = 1.0 + _estimate.k1 * r2 + _estimate.k2 * r2 * r2;
        return Eigen::Vector2d(_estimate.focal * distortion * P_c[0],
                               _estimate.focal * distortion * P_c[1]);
    }
};

/**
 * @brief 路标类
 */
class PointVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointVertex() = default;
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &in) const override { return true; }
    // 顶点重置函数
    void setToOriginImpl() override { _estimate = Eigen::Vector3d::Zero(); }
    // 顶点更新函数
    void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }
};

/**
 * @brief 边类, 这里是二元边
 */
class myEdge : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, CameraVertex, PointVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read([[maybe_unused]] istream &in) override { return true; }
    bool write([[maybe_unused]] ostream &in) const override { return true; }
    // 计算误差
    void computeError() override
    {
        const CameraVertex *v0 = static_cast<const CameraVertex *>(_vertices[0]);
        const PointVertex *v1 = static_cast<const PointVertex *>(_vertices[1]);
        _error = _measurement - v0->project(v1->estimate());
    }
    // 误差对优化变量的偏导
    // void linearizeOplus() override
    // {
    //     // NOTE 这里没有实现雅可比，g2o会自动数值求导
    //     // 若要实现解析导，首先应该实现类的默认构造传入相机系下的三维点,然后根据PnP一节
    //     // 已经推导出的雅可比结果, 传入_jacobianOplusXi(2,9)和_jacobianOplusXj(2,3)
    //     // 由于这里考虑了畸变，故实际上与PnP一节中给出的公式还有不同, 比较复杂,
    //     // 可以使用在线网站进行求导, 或在obsidian笔记中寻找
    // }
};

void SolveBA(BAL &bal_problem)
{
    double *points = bal_problem.parameters_ +
                     9 * bal_problem.num_cameras_;  // 存储路标点point的首地址
    double *cameras = bal_problem.parameters_;      // 存储相机位姿pose的首地址
    const double *observations = bal_problem.observations_;  // 观测值的首地址

    // step 1: 创建BlockSolver类型, <顶点0的维度,顶点1的维度>
    using BlockSolverType = g2o::BlockSolverPL<9, 3>;
    // step 2: 创建线性求解器类型,这一步是为了解H delta(x) = g,使用稀疏分解法
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;
    // step 3: 创建总求解器Solver,使用LM法
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // step 4: 配置稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    // step 5: 向图中添加顶点,并把顶点指针存起来
    vector<CameraVertex *> cameravertex_vector;
    for (int i = 0; i < bal_problem.num_cameras_; ++i)
    {
        auto v = new CameraVertex();
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(cameras + 9 * i));
        optimizer.addVertex(v);
        cameravertex_vector.push_back(v);
    }
    vector<PointVertex *> pointvertex_vector;
    for (int i = 0; i < bal_problem.num_points_; ++i)
    {
        auto v = new PointVertex();
        double *point = points + 3 * i;
        v->setId(i + bal_problem.num_cameras_);
        v->setEstimate(Eigen::Vector3d(point[0], point[1], point[2]));
        // NOTE 课本此处先Schur边缘化掉了点的增量，求解相机位姿增量后再求三维点的增量
        // NOTE 也可以在相机处设置边缘化，但是这样，求解点的矩阵仍然很大，因此先边缘化掉点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        pointvertex_vector.push_back(v);
    }

    // step 6: 向图中添加边
    for (int i = 0; i < bal_problem.num_observations_; ++i)
    {
        auto edge = new myEdge;
        edge->setId(i);
        edge->setVertex(0, cameravertex_vector[bal_problem.camera_index_[i]]);
        edge->setVertex(1, pointvertex_vector[bal_problem.point_index_[i]]);
        edge->setMeasurement(
            Eigen::Vector2d(observations[2 * i], observations[2 * i + 1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // step 7: 设置优化参数并开始优化
    cout << "------------------start optimization----------------" << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(40);  // 设置迭代次数

    // step 8: 输出优化值到内存, 即bal类中, 方便最后打印输出
    for (int i = 0; i < bal_problem.num_cameras_; ++i)
    {
        auto v = cameravertex_vector[i];
        v->estimate().set_to(cameras + 9 * i);
    }
    for (int i = 0; i < bal_problem.num_points_; ++i)
    {
        auto v = pointvertex_vector[i];
        auto point = points + 3 * i;
        point[0] = v->estimate()[0];
        point[1] = v->estimate()[1];
        point[2] = v->estimate()[2];
    }
}

int main()
{
    BAL bal_problem(bal_data_file);  // 读取数据, 初始化bal类
    bal_problem.Normalize();         // 数据归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);  // 添加噪声 （相机旋转、相机平移、路标点）、
    bal_problem.WriteToFile("./src/ch9_back_end_1/data/16-22106-g2o-initial.txt");
    bal_problem.WriteToPLYFile("./src/ch9_back_end_1/data/16-22106-g2o-initial.ply");
    SolveBA(bal_problem);  // 求解BA问题
    bal_problem.WriteToFile("./src/ch9_back_end_1/data/16-22106-g2o-final.txt");
    bal_problem.WriteToPLYFile("./src/ch9_back_end_1/data/16-22106-g2o-final.ply");
    return EXIT_SUCCESS;
}
