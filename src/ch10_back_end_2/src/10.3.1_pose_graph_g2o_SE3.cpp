/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       10.3.1_pose_graph_g2o_SE3.cpp
 * @author     Speike
 * @date       2023/10/24 17:13:40
 * @brief      本程序演示如何用g2o进行位姿图优化
 *             sphere.g2o是人工用g2o/bin/create_sphere工具生成的文件
 *             文件前半部分是节点(ID tx ty tz qx qy qz qw)
 *             后半部分是边(节点ID 节点ID tx ty tz qx qy qz qw 信息矩阵的右上角)
 * @brief      通过随意修改sphere.g2o文件可以发现，随意修改顶点，最后都可以通过优化得到
 *             正确的结果，但是如果随意修改边的约束，则会对节点位姿的估计产生较大的影响，
 *             说明只要相邻两帧位姿间的观测正确，即使位姿顶点通过误差积累偏移较大，
 *             也可以正确修复。但是短时间的观测误差一定不能太大，不然谁都救不了你
 * @brief      在读取sphere.g2o时，尽管可以直接通过load函数读取整个图，
 *             但是还是自己写读取代码，来获得更深刻理解
 * @brief      这里使用g2o提供的SE3表示位姿，他实际上是四元数而非李代数
 *             // NOTE 使用了g2o预置的顶点和边类，因此不需要定义顶点和边
 **/

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <experimental/filesystem>
#include <fstream>
#include <iostream>

using namespace std;

string sphere_file_path = "./src/ch10_back_end_2/data/sphere_10.3.1.g2o";

int main()
{
    ifstream fin(sphere_file_path);
    if (!fin)
    {
        cerr << "文件\"" << sphere_file_path << "\"读取失败，当前路径为"
             << experimental::filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }

    // step 1：创建BlockerSolver, <6维位姿顶点1, 6维位姿顶点2>
    using BlockerSolverType = g2o::BlockSolverPL<6, 6>;
    // step 2: 创建线性求解器类型, 稀疏求解
    using LinearSolverType = g2o::LinearSolverEigen<BlockerSolverType::PoseMatrixType>;
    // step 3: 创建总求解器
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockerSolverType>(g2o::make_unique<LinearSolverType>()));
    // step 4: 配置稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    // step 5 6: 向图中添加顶点和边
    int vertexCnt = 0, edgeCnt = 0;
    while (!fin.eof())
    {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            g2o::VertexSE3* v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            // NOTE 固定节点0，这里不固定会优化的比较快，但是不固定的话，会导致轨迹
            // NOTE 乱飘，所以SLAM中优化时一般都固定第一帧不优化
            // NOTE 这里为了方便理解，在sphere.g2o中将第一帧节点大幅远离了初始位置
            // NOTE 可以看到固定第一帧时其他帧都在往这个方向靠近。
            if (index == 0) v->setFixed(true);
            // 好像不固定还会求解的快些
            v->read(fin);  // QUERY read 函数是什么, 调试也进不去
            optimizer.addVertex(v);
            vertexCnt++;
        } else if (name == "EDGE_SE3:QUAT")
        {
            g2o::EdgeSE3* edge = new g2o::EdgeSE3();
            int index_vertex1, index_vertex2;
            fin >> index_vertex1 >> index_vertex2;
            edge->setId(edgeCnt++);
            edge->setVertex(0, optimizer.vertices()[index_vertex1]);
            edge->setVertex(1, optimizer.vertices()[index_vertex2]);
            edge->read(fin);
            optimizer.addEdge(edge);
        }
        if (!fin.good()) break;  // 文件流不良好，就停止
    }

    cout << "一共读取了" << vertexCnt << "个顶点，读取了" << edgeCnt << "个边." << endl;

    // step 7: 设置优化参数并开始优化
    cout << "start optimization" << endl;
    optimizer.initializeOptimization();  // 初始化
    optimizer.optimize(1);               // 设置迭代次数,设置几次迭代几次
    // NOTE 好像只有g2o预先定义的顶点和边做优化时才能使用save，自定义的不能用
    optimizer.save("./src/ch10_back_end_2/data/result1_10.3.1.g2o");
    cout << "保存优化结果图到./src/ch10_back_end_2/data/result1_10.3.1.g2o" << endl;
    optimizer.optimize(200);  // 设置迭代次数,设置几次迭代几次
    optimizer.save("./src/ch10_back_end_2/data/result2_10.3.1.g2o");
    cout << "保存优化结果图到./src/ch10_back_end_2/data/result2_10.3.1.g2o" << endl;
    optimizer.optimize(200);  // 设置迭代次数,设置几次迭代几次
    optimizer.save("./src/ch10_back_end_2/data/result3_10.3.1.g2o");
    cout << "保存优化结果图到./src/ch10_back_end_2/data/result3_10.3.1.g2o" << endl;
    optimizer.optimize(300);  // 设置迭代次数,设置几次迭代几次
    optimizer.save("./src/ch10_back_end_2/data/result4_10.3.1.g2o");
    cout << "保存优化结果图到./src/ch10_back_end_2/data/result4_10.3.1.g2o" << endl;
    optimizer.optimize(400);  // 设置迭代次数,设置几次迭代几次
    optimizer.save("./src/ch10_back_end_2/data/result5_10.3.1.g2o");
    cout << "保存优化结果图到./src/ch10_back_end_2/data/result5_10.3.1.g2o" << endl;

    return EXIT_SUCCESS;
}
