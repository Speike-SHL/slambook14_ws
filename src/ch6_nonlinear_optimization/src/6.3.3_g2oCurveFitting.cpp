/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       6.3.3_g2oCurveFitting.cpp
 * @author     Speike
 * @date       2023/07/26 14:59:41
 * @brief      G2O曲线拟合实验
**/
#include <functional>
#include <opencv2/core/core.hpp>
#include "tic_toc.h"
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>

using namespace std;

//TAG 生成拟合所需数据
/**
 * @param input func 生成数据的函数
 * @param input N 数据数量
 * @param input w_sigma 噪声Sigma值
 * @param output <x_data,y_data> 对组
*/
pair<vector<double>,vector<double>> generateData(function<double(double)> func, int N, double w_sigma)
{
    cv::RNG rng;                         // OpenCV随机数产生器
    vector<double> x_data, y_data;       // 数据
    for (int i = 0; i < N; i++)
    {
        double x = i / double(N);
        x_data.push_back(x);
        y_data.push_back(func(x) + rng.gaussian(w_sigma * w_sigma));
    }
    return make_pair(x_data, y_data);
}

//顶点类, <顶点即待优化变量维度, 顶点类型>
class myVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myVertex(){}
    bool read(istream &in) override{}
    bool write(ostream &out) const override{}
    //XXX 重置
    void setToOriginImpl() override
    {
        _estimate = Eigen::Vector3d::Zero();
    }
    // 优化变量更新
    void oplusImpl(const double*update) override
    {
        _estimate += Eigen::Vector3d(update);
    }
};

//边类 <测量值维度, 测量值数据类型, 顶点的类型>
class myEdge : public g2o::BaseUnaryEdge<1, double, myVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myEdge(double x): BaseUnaryEdge(), _x(x) {}
    bool read(istream &in) override{}
    bool write(ostream &out) const override{}
    // 定义误差模型
    void computeError() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Eigen::Vector3d cofficient = v->estimate();
        _error(0, 0) = _measurement - exp(cofficient(0, 0) * _x * _x + cofficient(1, 0) * _x + cofficient(2, 0));
    }
    // 雅可比矩阵
    void linearizeOplus() override
    {
        const myVertex *v = static_cast<const myVertex *>(_vertices[0]);
        const Eigen::Vector3d cofficient = v->estimate();
        _jacobianOplusXi[0] = -_x * _x * exp(cofficient[0] * _x * _x + cofficient[1] * _x + cofficient[2]);
        _jacobianOplusXi[1] = -_x * exp(cofficient[0] * _x * _x + cofficient[1] * _x + cofficient[2]);
        _jacobianOplusXi[2] = -exp(cofficient[0] * _x * _x + cofficient[1] * _x + cofficient[2]);
    }
private:
    double _x;
};

void useG2O(pair<vector<double>,vector<double>> data, Eigen::Vector3d init_param, double sigma)
{
    using BlockSolverType = g2o::BlockSolverPL<3, 1>; // 优化变量维度为3, 误差值维度为1
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>; // 线性求解器类型
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );  // 使用GN法创建求解器
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出
    // 往图中增加顶点,即优化变量
    myVertex *v = new myVertex();
    v->setEstimate(init_param);
    v->setId(0);
    optimizer.addVertex(v);
    // 往图中增加边,即观测
    for (int i = 0; i < static_cast<int>(data.first.size()); i++)
    {
        myEdge *edge = new myEdge(data.first[i]);
        edge->setId(i);
        edge->setVertex(0, v);  // 设置连接的顶点(点id,点指针)
        edge->setMeasurement(data.second[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (sigma * sigma)); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
    }
    // 执行优化
    TicToc time;
    optimizer.initializeOptimization();
    optimizer.optimize(20); // 设置迭代次数
    cout << "--> G2O求解耗时: " << time.toc() << "ms. 估计结果为 :" << v->estimate() << endl;
}

int main()
{
    // 定义拟合函数模型
    function<double(Eigen::VectorXd, double)> func = [](Eigen::VectorXd coefficient, double x)
    { return exp(coefficient[0] * x * x + coefficient[1] * x + coefficient[2]); };
    // 生成数据
    double sigma = 1.0;
    auto data = generateData(std::bind(func, Eigen::VectorXd{{1.0, 2.0, 1.0}}, std::placeholders::_1), 100, sigma);
    // G2O曲线拟合
    useG2O(data, Eigen::Vector3d(2.0, -1.0, 5.0), sigma);
    return EXIT_SUCCESS;
}
