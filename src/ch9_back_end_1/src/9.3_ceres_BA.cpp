/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       9.3_ceres_BA.cpp
 * @author     Speike
 * @date       2023/09/28 22:06:51
 * @brief      使用ceres求解大规模BA，同时优化三维点位置、相机位姿、与相机内参f、k1、k2
 * BAL数据集网址：https://grail.cs.washington.edu/projects/bal/index.html
 **/

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "BAL.h"

string bal_data_file = "./src/ch9_back_end_1/data/problem-16-22106-pre.txt";

struct COST_FUNCTION
{
    COST_FUNCTION(double observation_x, double observation_y)
        : observed_x(observation_x), observed_y(observation_y)
    {
    }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residual) const
    {
        // P = R * X + t 把世界系转换为相机系
        T P_c[3];
        ceres::AngleAxisRotatePoint(camera, point, P_c);
        P_c[0] = P_c[0] + camera[3];
        P_c[1] = P_c[1] + camera[4];
        P_c[2] = P_c[2] + camera[5];

        // p = -P / P.z 相机坐标系到归一化坐标
        T px = -P_c[0] / P_c[2];
        T py = -P_c[1] / P_c[2];

        // 去畸变系数
        T p2 = px * px + py * py;
        const T& k1 = camera[7];
        const T& k2 = camera[8];
        T r = T(1.0) + k1 * p2 + k2 * p2 * p2;

        // 转换到像素坐标
        T prediction[2];
        const T& f = camera[6];
        prediction[0] = f * r * px;
        prediction[1] = f * r * py;

        // 计算残差
        residual[0] = prediction[0] - T(observed_x);
        residual[1] = prediction[1] - T(observed_y);

        return true;
    }

private:
    double observed_x;
    double observed_y;
};

void SolveBA(BAL& bal_problem)
{
    double* points = bal_problem.parameters_ +
                     9 * bal_problem.num_cameras_;  // 存储路标点point的首地址
    double* cameras = bal_problem.parameters_;      // 存储相机位姿pose的首地址
    const double* observations = bal_problem.observations_;  // 观测值的首地址

    // 构建最小二乘问题
    ceres::Problem problem;
    // 循环添加观测残差，循环数为观测数量
    for (int i = 0; i < bal_problem.num_observations_; ++i)
    {
        // 添加残差块, ceres会迭代到残差块取最小值
        problem.AddResidualBlock(
            // 自动求导，<误差类型，输出残差维度，...每个参数块中待估计参数维度>
            // NOTE 这里连同相机内参一起作了优化
            // 因为bal数据跟camera相关的是9维度，第二个参数块point 3维度
            new ceres::AutoDiffCostFunction<COST_FUNCTION, 2, 9, 3>(
                new COST_FUNCTION(observations[2 * i], observations[2 * i + 1])),
            new ceres::HuberLoss(1.0),  // 鲁棒核函数
            cameras + 9 * bal_problem.camera_index_[i],
            points + 3 * bal_problem.point_index_[i]);
    }
    // 配置求解器开始求解
    ceres::Solver::Options options;
    options.linear_solver_type =
        ceres::LinearSolverType::SPARSE_SCHUR;  // 求解H*dx = g的方法,使用Schur消元法
    options.minimizer_progress_to_stdout = true;  // 输出到cout
    ceres::Solver::Summary summary;               // 求解器的摘要
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

int main()
{
    BAL bal_problem(bal_data_file);  // 读取数据, 初始化bal类
    bal_problem.Normalize();         // 数据归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);  // 添加噪声 （相机旋转、相机平移、路标点）
    bal_problem.WriteToFile("./src/ch9_back_end_1/data/16-22106-ceres-initial.txt");
    bal_problem.WriteToPLYFile("./src/ch9_back_end_1/data/16-22106-ceres-initial.ply");
    SolveBA(bal_problem);  // 求解BA问题
    bal_problem.WriteToFile("./src/ch9_back_end_1/data/16-22106-ceres-final.txt");
    bal_problem.WriteToPLYFile("./src/ch9_back_end_1/data/16-22106-ceres-final.ply");
    return EXIT_SUCCESS;
}
