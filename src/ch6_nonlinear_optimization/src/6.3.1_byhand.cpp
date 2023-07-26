/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       6.3.1_byhand.cpp
 * @author     Speike
 * @date       2023/06/26 15:58:05
 * @brief      手写非线性优化解曲线拟合问题:
 *               GN法、梯度下降法、LM法, Dogleg法
 *               关于牛顿法，由于求导较为复杂，没有实现，如果雅可比和海森矩阵使用数值解时可以考虑实现一下
 */

#include "tic_toc.h"
#include <iostream>
#include <functional>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

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

//TAG 计算残差
/**
 * @param input data 拟合所需的数据
 * @param input func 拟合的函数模型
 * @return residual 残差矩阵
 */
Eigen::VectorXd cal_Residual(pair<vector<double>, vector<double>> &data, function<double(double)> func)
{
    Eigen::VectorXd residual(data.first.size());
    for (size_t i = 0; i < data.first.size(); i++)
    {
        double xi = data.first[i], yi = data.second[i];
        residual[i] = yi - func(xi);
    }
    return residual;
}

//TAG 计算Jacobian矩阵
/**
 * @param input data 拟合所需的数据
 * @param input func_Jacobian Jacobian的解析式
 * @param input params 迭代参数
 * @return Jacobian 矩阵
*/
Eigen::MatrixXd cal_Jacobian(pair<vector<double>, vector<double>> &data, 
                            vector<function<double(Eigen::VectorXd, double)>> func_Jacobian,
                            Eigen::VectorXd params
                            )
{
    Eigen::MatrixXd Jacobian(data.first.size(), func_Jacobian.size());
    for (size_t i = 0; i < data.first.size(); i++)
    {
        for (size_t j = 0; j < func_Jacobian.size(); j++)
        {
            Jacobian(i, j) = func_Jacobian[j](params, data.first[i]);
        }
    }
    return Jacobian;
}

//TAG 手写高斯牛顿法
/**
 * @param input data 拟合所需的数据
 * @param input func 拟合的函数模型
 * @param input Jacobian 误差函数的Jacobian矩阵
 * @param input params 迭代初值
 * @param input iterations 迭代次数
 * @param input learning_rate 学习率
 */
void GaussNewton(pair<vector<double>, vector<double>> &data,
                function<double(Eigen::VectorXd, double x)> func,
                vector<function<double(Eigen::VectorXd, double)>> Jacobian,
                Eigen::VectorXd params, int iterations=100, double learning_rate = 1)
{
    TicToc time;
    double cost = 0, lastCost = 0;          // 本次迭代的cost和上一次迭代的cost
    for (int iter = 0; iter < iterations; iter++)
    {
        // 计算残差和Jacobian矩阵
        Eigen::MatrixXd Jf = cal_Jacobian(data, Jacobian, params);   // Jacobian矩阵
        Eigen::VectorXd f = cal_Residual(data, bind(func, params, placeholders::_1)); // 残差向量
        cost = f.norm();

        // 计算Hessian矩阵和梯度向量
        Eigen::MatrixXd H = Jf.transpose() * Jf;
        Eigen::VectorXd g = -Jf.transpose() * f;

        // 求解线性方程 H*dx = g
        Eigen::VectorXd dx = H.ldlt().solve(g);
        if (isnan(dx[0]))
        {
            cout << "delta_x足够小,结束迭代" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost)
        {
            cout << fixed << setprecision(16) 
                 << left
                 << "Iter: " << setw(9) << iter
                 << "cost: " << cost
                 << " >= last cost: " << lastCost << ", 迭代发散" << endl;
            break;
        }
        // 更新参数估计值和cost
        params += learning_rate * dx;
        lastCost = cost;

        if (iter % (iterations / 10) == 0)
            cout << fixed << setprecision(16) 
                << left
                << "Iter: " << setw(8) << iter
                << setw(12) << " total cost: "
                << setw(28) << cost
                << "estimated params: "
                << fixed << setprecision(5)
                << params.transpose() << endl;
    }

    cout << fixed << setprecision(5) << "\033[33m"
         << "--> 高斯牛顿法耗时: " << time.toc() << "ms. 估计结果为: "
         << params.transpose() << "." << "\033[0m" << endl;
}

//TAG 手写梯度下降法
/**
 * @param input data 拟合所需的数据
 * @param input func 拟合的函数模型
 * @param input Jacobian 误差函数的Jacobian矩阵
 * @param input params 迭代初值
 * @param input iterations 迭代次数
 * @param input learning_rate 学习率
 */
void GradientDescent(pair<vector<double>, vector<double>> &data,
                    function<double(Eigen::VectorXd, double x)> func,
                    vector<function<double(Eigen::VectorXd, double)>> Jacobian,
                    Eigen::VectorXd params, int iterations=100, double learning_rate = 1)
{
    TicToc time;
    double cost = 0, lastCost = 0;          // 本次迭代的cost和上一次迭代的cost
    for (int iter = 0; iter < iterations; iter++)
    {
        // 计算残差和Jacobian矩阵
        Eigen::MatrixXd Jf = cal_Jacobian(data, Jacobian, params);   // Jacobian矩阵
        Eigen::VectorXd f = cal_Residual(data, bind(func, params, placeholders::_1)); // 残差向量
        cost = f.norm();

        // 计算梯度
        Eigen::VectorXd JF = Jf.transpose() * f;

        // dx等于梯度的反方向
        Eigen::VectorXd dx = - JF;
        if (isnan(dx[0]))
        {
            cout << "delta_x足够小,结束迭代" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost)
        {
            cout << fixed << setprecision(16) 
                 << left
                 << "Iter: " << setw(9) << iter
                 << "cost: " << cost
                 << " >= last cost: " << lastCost << ", 迭代发散" << endl;
            break;
        }

        // 更新参数估计值和cost
        params += learning_rate * dx;
        lastCost = cost;

        if (iter % (iterations / 1000) == 0)
            cout << fixed << setprecision(16) 
                << left
                << "Iter: " << setw(8) << iter
                << setw(12) << " total cost: "
                << setw(28) << cost
                << "estimated params: "
                << fixed << setprecision(5)
                << params.transpose() << endl;
    }

    cout << fixed << setprecision(5) << "\033[34m"
         << "--> 梯度下降法耗时: " << time.toc() << "ms. 估计结果为: "
         << params.transpose() << "." << "\033[0m" << endl;
}

//TAG 手写列文伯格-马夸特法
/**
 * @brief 实现流程看本节手写笔记的伪代码
 * @param input data 拟合所需的数据
 * @param input func 拟合的函数模型
 * @param input Jacobian 误差函数的Jacobian矩阵
 * @param input params 迭代初值
 * @param input iterations 迭代次数
 */
void LevenbergMarquardt(pair<vector<double>, vector<double>> &data,
                    function<double(Eigen::VectorXd, double x)> func,
                    vector<function<double(Eigen::VectorXd, double)>> Jacobian,
                    Eigen::VectorXd params, int iterations=100)
{
    TicToc time;
    int iter = 0;
    double v = 2;
    Eigen::MatrixXd Jf = cal_Jacobian(data, Jacobian, params);   // Jacobian矩阵
    Eigen::VectorXd f = cal_Residual(data, bind(func, params, placeholders::_1)); // 残差向量
    Eigen::MatrixXd H = Jf.transpose() * Jf;
    Eigen::VectorXd g = -Jf.transpose() * f;
    bool stop = (g.lpNorm<Eigen::Infinity>() <= 10e-15);
    double u = 10e-3 * H.diagonal().maxCoeff();
    double rho;
    while (!stop && iter < iterations)
    {
        if (iter % (iterations / 10) == 0)
            cout << fixed << setprecision(16) 
                << left
                << "Iter: " << setw(8) << iter
                << setw(12) << " total cost: "
                << setw(28) << f.norm()
                << "estimated params: "
                << fixed << setprecision(5)
                << params.transpose() << endl;
        iter += 1;
        while (true)
        {
            Eigen::VectorXd dx = (H + u * Eigen::MatrixXd::Identity(H.rows(), H.cols())).ldlt().solve(g);
            if (dx.norm() <= 10e-15 * (params.norm()))
            {
                stop = true;
                cout << left << "Iter: " << setw(9) << iter << "delta_x足够小,结束迭代" << endl;
            }
            else
            {
                Eigen::VectorXd new_params = params + dx;
                Eigen::VectorXd new_f = cal_Residual(data, bind(func, new_params, placeholders::_1));
                rho = (f.norm() * f.norm() - new_f.norm() * new_f.norm()) / (dx.transpose() * (u * dx + g));
                if (rho > 0)
                {
                    params = new_params;
                    f = new_f;
                    Jf = cal_Jacobian(data, Jacobian, params);
                    H = Jf.transpose() * Jf;
                    g = -Jf.transpose() * f;
                    stop = (g.lpNorm<Eigen::Infinity>() <= 10e-15) || (f.norm() * f.norm() <= 10e-15);
                    u = u * max(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1.0, 3));
                    v = 2;
                }
                else
                {
                    u = u * v;
                    v = 2 * v;
                }
            }
            if (rho > 0 || stop)
                break;
        }
    }
    cout << fixed << setprecision(5)  << "\033[32m"
         << "--> 列文伯格-马夸尔特法耗时: " << time.toc() << "ms. 估计结果为: "
         << params.transpose() << "." << "\033[0m" << endl;
}

//TAG 手写Dogleg法
/**
 * @brief 实现流程看本节手写笔记的伪代码
 * @param input data 拟合所需的数据
 * @param input func 拟合的函数模型
 * @param input Jacobian 误差函数的Jacobian矩阵
 * @param input params 迭代初值
 * @param input iterations 迭代次数
 */
void DogLeg(pair<vector<double>, vector<double>> &data,
            function<double(Eigen::VectorXd, double x)> func,
            vector<function<double(Eigen::VectorXd, double)>> Jacobian,
            Eigen::VectorXd params,int iterations=100)
{
    TicToc time;
    int iter = 0;
    double delta = 1.0;
    Eigen::MatrixXd Jf = cal_Jacobian(data, Jacobian, params);   // Jacobian矩阵
    Eigen::MatrixXd H = Jf.transpose() * Jf;
    Eigen::VectorXd f = cal_Residual(data, bind(func, params, placeholders::_1)); // 残差向量
    Eigen::VectorXd g = -Jf.transpose() * f;
    bool stop = (g.lpNorm<Eigen::Infinity>() <= 10e-15) || (f.lpNorm<Eigen::Infinity>() <= 10e-20);
    Eigen::VectorXd dx_sd, dx_gn, dx_dl;
    double rho;
    while (!stop && iter < iterations)
    {
        if (iter % (iterations / 10) == 0)
            cout << fixed << setprecision(16) 
                << left
                << "Iter: " << setw(8) << iter
                << setw(12) << " total cost: "
                << setw(28) << f.norm()
                << "estimated params: "
                << fixed << setprecision(5)
                << params.transpose() << endl;
        iter += 1;
        double alpha = g.squaredNorm() / (Jf*g).squaredNorm();
        dx_sd = alpha * g;
        bool GNcomputed = false;
        while (true)
        {
            double LL;
            if (dx_sd.norm() >= delta)
            {
                dx_dl = (delta/dx_sd.norm())*dx_sd;
                LL = delta*(2*(alpha*g).norm()-delta)/(2*alpha);
            }
            else
            {
                if (!GNcomputed)
                {
                    dx_gn = H.ldlt().solve(g);
                    GNcomputed = true;
                }
                if (dx_gn.norm()<=delta)
                {
                    dx_dl=dx_gn;
                    LL = 1.0 / 2.0 * f.squaredNorm();
                }
                else
                {
                    auto &a = dx_sd;
                    auto &b = dx_gn;
                    double c = a.transpose() * (b - a);
                    double beta;
                    if (c <= 0)
                        beta = (-c + sqrt(c * c + (b - a).squaredNorm() * (delta * delta - a.squaredNorm()))) / (b - a).squaredNorm();
                    else
                        beta = (delta * delta - a.squaredNorm()) / (c + sqrt(c * c + (b - a).squaredNorm() * (delta * delta - a.squaredNorm())));
                    dx_dl = dx_sd + beta * (dx_gn - dx_sd);
                    LL = 1.0 / 2.0 * (alpha * (1 - beta) * (1 - beta) * g.squaredNorm() + beta * (2 - beta) * f.squaredNorm());
                }
            }
            if (dx_dl.norm() <= 10e-15 * params.norm())
            {
                stop = true;
                cout << left << "Iter: " << setw(9) << iter << "delta_x足够小,结束迭代" << endl;
            }
            else
            {
                auto params_new = params + dx_dl;
                auto f_new = cal_Residual(data, bind(func, params_new, placeholders::_1));
                rho = (1.0 / 2.0 * f.squaredNorm() - 1.0 / 2.0 * f_new.squaredNorm()) / LL;
                if (rho > 0)
                {
                    params = params_new;
                    f = f_new;
                    Jf = cal_Jacobian(data, Jacobian, params);
                    H = Jf.transpose() * Jf;
                    g = -Jf.transpose() * f;
                    stop = (g.lpNorm<Eigen::Infinity>() <= 10e-15) || (f.lpNorm<Eigen::Infinity>() <= 10e-20);
                }
                if (rho > 0.75)
                    delta = max(delta, 3.0 * dx_dl.norm());
                else if (rho < 0.25)
                {
                    delta = delta / 2.0;
                    stop = (delta <= (10e-15 * params.norm() + 10e-15));
                }
            }
            if (rho > 0 || stop)
                break;
        }
    }
    cout << fixed << setprecision(5) << "\033[35m"
         << "--> Dog-Leg法耗时: " << time.toc() << "ms. 估计结果为: "
         << params.transpose() << "." << "\033[0m" << endl;
}

int main()
{
    // 定义拟合函数模型
    function<double(Eigen::VectorXd, double)> func = [](Eigen::VectorXd coefficient, double x) { return exp(coefficient[0] * x * x + coefficient[1] * x + coefficient[2]); };
    // 误差函数的雅可比矩阵
    vector<function<double(Eigen::VectorXd, double)>> Jacobian = {
        [&func](Eigen::VectorXd coefficient, double x){ return -x * x * func(coefficient, x); },
        [&func](Eigen::VectorXd coefficient, double x){ return -x * func(coefficient, x); },
        [&func](Eigen::VectorXd coefficient, double x){ return -func(coefficient, x); }
    };
    // 生成数据
    auto data = generateData(std::bind(func, Eigen::VectorXd{{1.0, 2.0, 1.0}}, std::placeholders::_1), 100, 1.0);
    // 不同方法不同初值求解
    GaussNewton(data, func, Jacobian, Eigen::VectorXd{{2.0, -1.0, 5.0}}, 20, 1.0);
    GradientDescent(data, func, Jacobian, Eigen::VectorXd{{2.0, -1.0, 5.0}}, 500000, 0.0000005);
    LevenbergMarquardt(data, func, Jacobian, Eigen::VectorXd{{2.0, -1.0, 5.0}}, 20);
    DogLeg(data, func, Jacobian, Eigen::VectorXd{{2.0, -1.0, 5.0}}, 20);
    cout << "---------------------------------------------------------------------------" << endl;
    GaussNewton(data, func, Jacobian, Eigen::VectorXd{{0.0, 0.0, 0.0}}, 200, 0.1);
    GradientDescent(data, func, Jacobian, Eigen::VectorXd{{0.0, 0.0, 0.0}}, 500000, 0.0000005);
    LevenbergMarquardt(data, func, Jacobian, Eigen::VectorXd{{0.0, 0.0, 0.0}}, 20);
    DogLeg(data, func, Jacobian, Eigen::VectorXd{{0.0, 0.0, 0.0}}, 20);
    cout << "---------------------------------------------------------------------------" << endl;
    GaussNewton(data, func, Jacobian, Eigen::VectorXd{{0.1, -10.0, 7.0}}, 200, 0.1);
    GradientDescent(data, func, Jacobian, Eigen::VectorXd{{0.1, -10.0, 7.0}}, 50000000, 0.0000001);
    LevenbergMarquardt(data, func, Jacobian, Eigen::VectorXd{{0.1, -10.0, 7.0}}, 50);
    DogLeg(data, func, Jacobian, Eigen::VectorXd{{0.1, -10.0, 7.0}}, 20);
    return 0;
}
