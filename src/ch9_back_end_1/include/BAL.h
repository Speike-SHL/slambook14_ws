/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       BAL.h
 * @author     Speike
 * @date       2023/09/29 17:31:34
 * @brief      包含了对BAL数据集的一系列操作
 **/

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

/**
 * @brief 从文件读入BAL dataset数据
 * @details 使用针孔相机模型, 每个相机的参数有 R, t, 焦距f, 径向畸变参数k1, k2
 * 将3D点投影到像素坐标的公式为:
 * P  =  R * X + t     (把世界坐标转换为相机坐标)
 * p  = -P / P.z       (相机坐标归一化)
 * p' =  f * r(p) * p  (转换得到像素坐标)
 * r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4
 * 由上面公式可以发现
 * 图像的原点是图像的中心，正x轴向右，正y轴向上. 此外，在相机系中，正z轴向后
 * @details 数据集格式为:
 * 第一行：<相机数量> <路标点数量> <观测数量>
 * ----
 * 第二行 ~ 第(1+观测数量)行：<相机编号> <路标点编号> <x像素坐标> <y像素坐标>
 * ----
 * 第(2+观测数量)行 ~ 第(1+观测数量+相机数量*9)行：
 * 每行分别代表 -R(3行) t(3行) f k1 k2, 9行为一组
 * ----
 * 第(2+观测数量+相机数量*9) ~ 第(1+观测数量+相机数量*9+路标点数量*3)行:
 * 世界系下路标点的三维坐标，3行为一组
 * @example 在problem-16-22106-pre.txt中
 * 第一行: <16个相机> <22106个路标点> <83718次观测>
 * ----
 * 第二行 ~ 第 83719 行: 以第二行为例
 * <相机0> <路标点0> <像素坐标x:-3.859900e+02> <像素坐标y:3.871200e+02>
 * ----
 * 第 83720 ~ 第 1+83719+16*9=83863 行: 共16个相机的参数, 每个相机9维
 * ----
 * 第 83864 ~ 第 1+83718+16*9+22106*3 = 150181 行: 共22106个路标点的坐标
 */
class BAL
{
public:
    BAL(const string &bal_data_file);

    void WriteToFile(const std::string &filename) const;

    void WriteToPLYFile(const std::string &filename) const;

    void CameraToAngelAxisAndCenter(const double *camera, double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis, const double *center,
                                    double *camera) const;

    void Normalize();

    void Perturb(const double rotation_sigma, const double translation_sigma,
                 const double point_sigma);
    ~BAL()
    {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    int num_cameras_;       // 相机数量
    int num_points_;        // 路标点数量
    int num_observations_;  // 观测数量
    // 所有相机参数和3D点坐标 num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    int num_parameters_;

    int *point_index_;      // 每个observation中的point index
    int *camera_index_;     // 每个observation中的camera index
    double *observations_;  // 每个observation中的2D像素坐标
    double *parameters_;    // 所有相机参数和3D点坐标
};
