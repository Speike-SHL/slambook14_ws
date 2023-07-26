/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       6.3.2_ceresCurveFitting.cpp
 * @author     Speike
 * @date       2023/07/25 21:36:38
 * @brief      ceres曲线拟合实验
**/

#include <iostream>
#include <functional>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "tic_toc.h"

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

//NOTE: 1 构造残差函数的计算模型
struct COST_FUNCTION
{
    COST_FUNCTION(double x, double y): _x(x), _y(y){}
    template <typename T>
    bool operator()(const T* const cofficient, T* residual) const // 不要少const
    {
        residual[0] = T(_y) - exp(cofficient[0] * T(_x) * T(_x) + cofficient[1] * T(_x) + cofficient[2]);    // residual 残差, cofficient 待估计参数
        return true;
    }
private:
    const double _x, _y;
};

void useceres(pair<vector<double>,vector<double>> data, double params[3])
{
    //NOTE: 2 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < static_cast<int>(data.first.size()); i++)
    {
        // 添加残差块, ceres会迭代到残差块取最小值
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<COST_FUNCTION, 1, 3>(
                new COST_FUNCTION(data.first[i], data.second[i])
            ),      // 自动求导，<误差类型，输出残差维度，输入带估计参数维度>
            nullptr,// 核函数，这里不使用，为空
            params  // 待估计参数
        );
    }

    //NOTE: 3 配置求解器开始求解
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 求解H*dx = g的方法
    options.trust_region_strategy_type = ceres::DOGLEG; // DogLeg法
    options.minimizer_progress_to_stdout = true; // 输出到cout

    ceres::Solver::Summary summary;     // 求解器的摘要
    TicToc time;
    ceres::Solve(options, &problem, &summary);
    cout << "--> ceres-solver求解耗时: " << time.toc() << "ms. 估计结果为: " << params[0] << " " << params[1] << " " << params[2] << "." << endl;
    cout << "ceres-solver求解报告为: " << endl;
    cout << summary.BriefReport() << endl; 
}

int main()
{
    // 定义拟合函数模型
    function<double(Eigen::VectorXd, double)> func = [](Eigen::VectorXd coefficient, double x)
    { return exp(coefficient[0] * x * x + coefficient[1] * x + coefficient[2]); };
    // 生成数据
    auto data = generateData(std::bind(func, Eigen::VectorXd{{1.0, 2.0, 1.0}}, std::placeholders::_1), 100, 1.0);
    // 使用ceres-solver进行曲线拟合
    useceres(data, new double[3]{2.0, -1.0, 5.0});
    useceres(data, new double[3]{0.1, -10.0, 7.0});
    return EXIT_SUCCESS;
}
