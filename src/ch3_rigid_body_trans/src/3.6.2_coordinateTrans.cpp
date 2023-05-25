/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       3.6.2_coordinateTrans.cpp
 * @author     Speike
 * @date       2023/05/23 17:07:59
 * @brief      3.6.2 实际坐标变换例子 以及 全日制学生混三下作业
**/

#include <iostream>
#include <Eigen/Geometry>
using namespace std;

void example_1();
void example_2();

int main(int, char**)
{
    example_1();
    example_2();
    return 0;
}

/**
 * @brief slambook14 3.6.2 小萝卜实际坐标变换
*/
void example_1()
{
    // NOTE: 这里初始化变换阵只能Identity,因为后面是根据初始化结果进行的旋转
    Eigen::Isometry3d T_R1W = Eigen::Isometry3d::Identity(); // 分别表示世界系到R1和R2的变换
    Eigen::Isometry3d T_R2W = Eigen::Isometry3d::Identity();
    T_R1W.prerotate(Eigen::Quaterniond(0.35, 0.2, 0.3, 0.1).normalized()); // 注意四元数归一化
    T_R1W.pretranslate(Eigen::Vector3d(0.3, 0.1, 0.1));
    T_R2W.rotate(Eigen::Quaterniond(-0.5, 0.4, -0.1, 0.2).normalized());
    T_R2W.pretranslate(Eigen::Vector3d(-0.1, 0.5, 0.3));

    Eigen::Vector3d p_R1(0.5, 0, 0.2);  // R1系下的点坐标

    Eigen::Vector3d p_R2 = T_R2W * T_R1W.inverse() * p_R1;
    cout << "p_R2: " << p_R2.transpose() << endl;
}

/**
 * @details 有两个右手系1和2,其中2系的x轴与1系的y轴方向相同，2系的y轴与1系z轴方向相反，2系的z轴与1系的x轴相反,两个坐标系原点重合, 求R12, 求1系中(1,1,1)在2系中的坐标。
 * @author 全日制学生混(bilibili)
 * @link https://github.com/cckaixin/Practical_Homework_for_slambook14
*/
void example_2()
{
    Eigen::Vector3d p1(1, 1, 1);
    // 方法一: 直接写出2系相对于1系的旋转矩阵
    Eigen::Matrix3d R12{{0, 0, -1}, {1, 0, 0}, {0, -1, 0}};
    Eigen::Vector3d p2 = R12.inverse() * p1;
    cout << ">> 使用旋转矩阵计算p2: " << p2.transpose() << endl;
    cout << ">> R12:\n"
         << "2系相对于1系的姿态\n"
         << "1系下2系的姿态\n"
         << "将2系中向量转到1系中\n"
         << "从2系到1系的旋转矩阵\n"
         << R12 << endl;
    // 方法二: ZYX欧拉角
    double yaw = M_PI / 2;
    double pitch = 0;
    double roll = -M_PI / 2;
    // 注意这里是R12,因为ZYX欧拉角将I系转到II系,得到的其实是II系在I系下的姿态
    R12 = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) 
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) 
        * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());    
    p2 = R12.inverse() * p1;
    cout << ">> 使用欧拉角计算p2: " << p2.transpose() << endl;
}
