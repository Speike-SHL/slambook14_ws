/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       4.4.1_useSophus.cpp
 * @author     Speike
 * @date       2023/06/02 10:09:22
 * @brief      如何使用Sophus库
**/

#include <iostream>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std; 

int main(int, char**)
{
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix();   // 初始化绕Z轴旋转45度的旋转矩阵
    Eigen::Quaterniond q(R);    // 初始化绕Z轴旋转45度的四元数
    cout << ">>Eigen旋转矩阵为:\n"
         << R << endl;
    cout << ">>Eigen四元数为:\n" << q.coeffs().transpose() << endl << endl;\

    /******************************转化为Sophus*******************************/
    Sophus::SO3d SO3_R(R);
    cout << ">>通过Eigen::Matrix3d构造SO(3):\n"
         << SO3_R.matrix() << endl;
    Sophus::SO3d SO3_q(q);
    cout << ">>通过Eigen::Quaterniond构造SO(3):\n"
         << SO3_q.matrix() << endl << endl;
    
    /********************************对数映射********************************/
    Eigen::Vector3d so3 = SO3_R.log();
    cout << ">>Sophus: SO(3)->so(3)对数映射:\n"
         << so3.transpose() << endl;
    Eigen::Vector3d so3_Eigen = Eigen::AngleAxisd(R).angle() * Eigen::AngleAxisd(R).axis();

    cout << ">>Eigen: SO(3)->so(3)直接转化为角轴实现对数映射\n" << so3_Eigen.transpose() << endl << endl;
    
    /********************************指数映射********************************/
    cout << ">>Sophus: so(3)->SO(3)指数映射:\n"
         << Sophus::SO3d::exp(so3).matrix() << endl << endl;

    /********************************求反对称********************************/
    cout << ">>Sophus: hat求反对称矩阵\n"
         << Sophus::SO3d::hat(so3) << endl;
    cout << ">>Sophus: vee反对称到向量\n"
         << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl
         << endl;

    /******************************SO3扰动模型*******************************/
    Eigen::Vector3d update_so3(1e-4, 0, 0); // 定义小扰动
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << ">>扰动模型SO3 updated = \n" << SO3_updated.matrix() << endl << endl;

    /******************************SE(3)的操作******************************/
    Eigen::Vector3d t(1, 0, 0);         // 沿X轴平移1
    Sophus::SE3d SE3_Rt(R, t);          // 从R,t构造SE(3)
    Sophus::SE3d SE3_qt(q, t);          // 从q,t构造SE(3)
    cout << ">>SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
    cout << ">>SE3 from q,t= \n" << SE3_qt.matrix() << endl;
    cout << ">>Sophus SE3中取R部分:\n"
         << SE3_Rt.so3().matrix() << endl;
    cout << ">>Sophus SE3中取t部分:\n"
         << SE3_Rt.translation().transpose() << endl;
    // 李代数se(3) 是一个六维向量，方便起见先typedef一下
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << ">>se3 = \n" << se3.transpose() << endl;
    // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.
    // 同样的，有hat和vee两个算符
    cout << ">>se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
    cout << ">>se3 hat vee = \n" << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;
    // 最后，演示一下更新
    Vector6d update_se3; //更新量
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << ">>SE3 updated = " << endl << SE3_updated.matrix() << endl;
    return 0;
}
