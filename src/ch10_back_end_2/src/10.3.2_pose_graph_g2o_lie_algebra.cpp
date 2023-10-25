/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       10.3.2_pose_graph_g2o_lie_algebra.cpp
 * @author     Speike
 * @date       2023/10/25 13:31:18
 * @brief      使用Sophus自定义顶点和边类，实现g2o_viewer能读取的类型
 *             实现了g2o顶点类和边类的read和write函数
 * @result     1. 当SE3的右雅可比的逆用单位阵时，迭代9次，时间0.457846s，损失127578.573836
 *             2. 当S右雅可比的逆使用公式10.11，迭代7次，时间0.450532s，损失127578.157860
 *             3. 当不给出雅可比使用自动求导时， 迭代20次，时间6.51467s，损失127578.157935
 **/

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <Eigen/Dense>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>

using namespace std;

string sphere_file_path = "./src/ch10_back_end_2/data/sphere_10.3.2.g2o";

// 实现se3的右雅可比
Eigen::Matrix<double, 6, 6> JrofTinv(Eigen::Matrix<double, 6, 1> ksi)
{
    Eigen::Vector3d rho = ksi.topRows(3);
    Eigen::Vector3d phi = ksi.bottomRows(3);
    Eigen::Matrix<double, 6, 6> J;
    J.block(0, 0, 3, 3) = Sophus::SO3d::hat(phi);
    J.block(0, 3, 3, 3) = Sophus::SO3d::hat(rho);
    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
    J.block(3, 3, 3, 3) = Sophus::SO3d::hat(phi);
    // 十四讲公式10.11
    J = Eigen::Matrix<double, 6, 6>::Identity() + 0.5 * J;
    // 或者如果误差接近于0，可直接设置为单位阵
    // J = Eigen::Matrix<double, 6, 6>::Identity();
    return J;
}

class myVertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myVertexSE3LieAlgebra() = default;
    // NOTE g2o顶点read函数实现, 一般用于设置顶点估计值_estimate
    bool read(istream& in) override
    {
        double data[7];
        for (int i = 0; i < 7; ++i) in >> data[i];
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Vector3d t(data[0], data[1], data[2]);
        Sophus::SE3d se3d(q, t);
        // 设置顶点估计值，类型与顶点类中的Type一样，这里为Sophus::SE3d
        setEstimate(se3d);
        return true;
    }
    // NOTE g2o顶点write函数实现
    bool write(ostream& out) const override
    {
        out << id() << " ";  // 顶点id
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        Eigen::Vector3d t = _estimate.translation();
        // tx ty tz qx qy qz qw
        out << t.transpose() << " " << q.coeffs().transpose() << endl;
        return true;
    }
    void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
    void oplusImpl(const double* update) override
    {
        Eigen::Matrix<double, 6, 1> upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(upd) * _estimate;
    }
};

class myEdgeSE3LieAlgebra
    : public g2o::BaseBinaryEdge<6, Sophus::SE3d, myVertexSE3LieAlgebra,
                                 myVertexSE3LieAlgebra>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myEdgeSE3LieAlgebra() = default;
    // NOTE g2o边类read函数实现, 一般用于设置边观测值_measure和信息矩阵
    bool read(istream& in) override
    {
        double data[7];
        for (int i = 0; i < 7; ++i) in >> data[i];
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Vector3d t(data[0], data[1], data[2]);
        Sophus::SE3d se3d(q, t);
        // 设置边的观测值，类型与边类的Type一样
        setMeasurement(se3d);
        // 设置信息矩阵，对称的
        for (int i = 0; i < information().rows() && in.good(); ++i)
            for (int j = i; j < information().cols() && in.good(); ++j)
            {
                in >> information()(i, j);
                if (i != j) information()(j, i) = information()(i, j);
            }
        return true;
    }
    // NOTE g2o边类write函数实现，用于从顶点读取数据并输出
    bool write(ostream& out) const override
    {
        myVertexSE3LieAlgebra* v1 = static_cast<myVertexSE3LieAlgebra*>(_vertices[0]);
        myVertexSE3LieAlgebra* v2 = static_cast<myVertexSE3LieAlgebra*>(_vertices[1]);
        // 输出两个顶点id
        out << v1->id() << " " << v2->id() << " ";
        Eigen::Quaterniond q = _measurement.unit_quaternion();
        Eigen::Vector3d t = _measurement.translation();
        // 输出 tx ty tz qx qy qz qw
        out << t.transpose() << " " << q.coeffs().transpose() << " ";
        // 输出信息矩阵右上角
        for (int i = 0; i < information().rows(); ++i)
            for (int j = i; j < information().cols(); ++j)
                out << information()(i, j) << " ";
        out << endl;
        return true;
    }
    void computeError() override
    {
        const myVertexSE3LieAlgebra* v1 =
            static_cast<const myVertexSE3LieAlgebra*>(_vertices[0]);
        const myVertexSE3LieAlgebra* v2 =
            static_cast<const myVertexSE3LieAlgebra*>(_vertices[1]);
        Sophus::SE3d Ti = v1->estimate();
        Sophus::SE3d Tj = v2->estimate();
        // 十四讲课本公式10.4
        _error = (_measurement.inverse() * Ti.inverse() * Tj).log();
    }
    // !!这个函数不重写时g2o默认数值求导!!
    virtual void linearizeOplus() override
    {
        const myVertexSE3LieAlgebra* v1 =
            static_cast<const myVertexSE3LieAlgebra*>(_vertices[0]);
        const myVertexSE3LieAlgebra* v2 =
            static_cast<const myVertexSE3LieAlgebra*>(_vertices[1]);
        [[maybe_unused]] Sophus::SE3d Ti = v1->estimate();
        Sophus::SE3d Tj = v2->estimate();
        Eigen::Matrix<double, 6, 6> Jrinv = JrofTinv(_error);
        _jacobianOplusXi = -Jrinv * Tj.inverse().Adj();
        _jacobianOplusXj = Jrinv * Tj.inverse().Adj();
    }
};

int main()
{
    ifstream fin(sphere_file_path);
    if (!fin)
    {
        cerr << "文件\"" << sphere_file_path << "\"读取失败，当前路径为"
             << experimental::filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }

    // step 1: 创建BlockSolver类型
    using BlockSolverType = g2o::BlockSolverPL<6, 6>;
    // step 2: 创建线性求解器类型
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    // step 3: 创建总求解器Solver
    auto* solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // step 4: 配置稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    // step 5 6: 向图中添加顶点和边
    int vertexCnt = 0, edgeCnt = 0;
    vector<myVertexSE3LieAlgebra*> vertexs_ptr;
    vector<myEdgeSE3LieAlgebra*> edges_ptr;
    while (!fin.eof())
    {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            myVertexSE3LieAlgebra* v = new myVertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            // SLAM中为了防止优化时轨迹乱飘，一般固定第一帧
            if (index == 0) v->setFixed(true);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vertexs_ptr.push_back(v);
        } else if (name == "EDGE_SE3:QUAT")
        {
            myEdgeSE3LieAlgebra* edge = new myEdgeSE3LieAlgebra();
            int index_vertex1, index_vertex2;
            fin >> index_vertex1 >> index_vertex2;
            edge->setId(edgeCnt++);
            edge->setVertex(0, optimizer.vertices()[index_vertex1]);
            edge->setVertex(1, optimizer.vertices()[index_vertex2]);
            edge->read(fin);
            optimizer.addEdge(edge);
            edges_ptr.push_back(edge);
        }
        if (!fin.good()) break;  // 文件流不良好，就停止
    }

    cout << "一共读取了" << vertexCnt << "个顶点，读取了" << edgeCnt << "个边." << endl;

    // step 7: 设置优化参数并开始优化
    cout << "start optimization" << endl;
    optimizer.initializeOptimization();  // 初始化
    optimizer.optimize(30);              // 设置迭代次数,设置几次迭代几次

    // NOTE 因为用了自定义顶点，且没有向g2o注册，自己实现保存
    ofstream fout("./src/ch10_back_end_2/data/result_10.3.2.g2o");
    for (auto* v : vertexs_ptr)
    {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (auto* edge : edges_ptr)
    {
        fout << "EDGE_SE3:QUAT ";
        edge->write(fout);
    }
    fout.close();
    return EXIT_SUCCESS;
}
