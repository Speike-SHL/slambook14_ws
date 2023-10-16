/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       BAL.cpp
 * @author     Speike
 * @date       2023/09/29 17:30:54
 * @brief      包含了对BAL数据集的一系列操作
 **/

#include "BAL.h"

#include <Eigen/Dense>
#include <experimental/filesystem>

#include "random.h"
#include "rotation.h"

using VectorRef = Eigen::Map<Eigen::VectorXd>;
using ConstVectorRef = Eigen::Map<const Eigen::VectorXd>;

template <typename T>
// 从 fptr 中读取能够匹配 format 格式的数据存入 value 中
void FscanfOrDie(FILE *fptr, const char *format, T *value)
{
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) std::cerr << "Invalid UW data file. ";
}

// 添加方差为 sigma 的高斯噪声
void PerturbPoint3(const double sigma, double *point)
{
    for (int i = 0; i < 3; ++i) point[i] += RandNormal() * sigma;
}

// 得到data数组中的中位数
double Median(std::vector<double> *data)
{
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}

// 从bal_data_file中读取数据并存到BAL的private变量
// (point_index_,camera_index_,observations_,parameters_)中
BAL::BAL(const string &bal_data_file)
{
    FILE *fptr = fopen(bal_data_file.c_str(), "r");
    if (fptr == nullptr)
    {
        cerr << "Error: unable to open file " << bal_data_file << ", 当前路径为"
             << experimental::filesystem::current_path() << endl;
        return;
    }
    // 第一行：<相机数量> <路标点数量> <观测数量>
    FscanfOrDie(fptr, "%d", &num_cameras_);       // 总相机数量
    FscanfOrDie(fptr, "%d", &num_points_);        // 总路标点数量
    FscanfOrDie(fptr, "%d", &num_observations_);  // 总观测数量
    cout << "Header: <" << num_cameras_ << "个相机> <" << num_points_ << "个路标点> <"
         << num_observations_ << "次观测>" << endl;

    point_index_ = new int[num_observations_];          // 观测的路标点编号
    camera_index_ = new int[num_observations_];         // 观测的相机编号
    observations_ = new double[num_observations_ * 2];  // 观测的2D像素坐标

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // 第二行 ~ 第(1+观测数量)行：<相机编号> <路标点编号> <x像素坐标> <y像素坐标>
    for (int i = 0; i < num_observations_; ++i)
    {
        FscanfOrDie(fptr, "%d", camera_index_ + i);           // <相机编号>
        FscanfOrDie(fptr, "%d", point_index_ + i);            // <路标点编号>
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i);      // <x像素坐标>
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + 1);  // <y像素坐标>
    }

    // 第(2+观测数量)行 ~ 第(1+观测数量+相机数量*9+路标点数量*3)行
    for (int i = 0; i < num_parameters_; ++i)
    {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    fclose(fptr);
}

// 把BAL类中的数据输出到普通文件中
void BAL::WriteToFile(const std::string &filename) const
{
    FILE *fptr = fopen(filename.c_str(), "w");  // 打开可写文件

    if (fptr == nullptr)
    {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    // 第一行：<相机数量><路标点数量><观测数量>
    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

    // 第二行 ~第(1 + 观测数量) 行：<相机编号><路标点编号><x像素坐标> <y像素坐标>
    for (int i = 0; i < num_observations_; ++i)
    {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j)
        {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    // 第(2+观测数量)行 ~ 第(1+观测数量+相机数量*9)行：
    // 每行分别代表 - R(3行) t(3行) f k1 k2, 9行为一组
    for (int i = 0; i < num_cameras_; ++i)
    {
        double angleaxis[9];
        // 这里用memcpy()函数进行复制，从 parameters_ + 9 * i + 4 位置开始的 9 *
        // sizeof(double) 内存空间的数据放入起始地址为angleaxis 的内存空间里
        memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        for (int j = 0; j < 9; ++j)
        {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    // 第(2+观测数量+相机数量*9) ~ 第(1+观测数量+相机数量*9+路标点数量*3)行:
    // 世界系下路标点的三维坐标，3行为一组
    const double *points = parameters_ + 9 * num_cameras_;
    for (int i = 0; i < num_points_; ++i)
    {
        const double *point = points + i * 3;
        for (int j = 0; j < 3; ++j)
        {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);  // 打开文件就要关闭文件
}

// 把BAL类中的数据输出的 PLY 文件中
void BAL::WriteToPLYFile(const std::string &filename) const
{
    std::ofstream of(filename.c_str());

    // ply 头， 主要包含element 类型(这里为vertex 点),
    // 数量为相机数+路标数以及element的属性
    of << "ply" << '\n'
       << "format ascii 1.0" << '\n'
       << "element vertex " << num_cameras_ + num_points_ << '\n'
       << "property float x" << '\n'
       << "property float y" << '\n'
       << "property float z" << '\n'
       << "property uchar red" << '\n'
       << "property uchar green" << '\n'
       << "property uchar blue" << '\n'
       << "end_header" << std::endl;

    // 创建两个数组，用于承接CameraToAngelAxisAndCenter()解析出来的相机旋转姿态和相机位置中心
    double angle_axis[3];
    double center[3];
    // 输出所有相机在世界系下位置，用绿色点表示
    for (int i = 0; i < num_cameras_; ++i)
    {
        // 取相机参数的首地址
        const double *camera = parameters_ + 9 * i;
        // 解析相机参数为世界系下的旋转和平移
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // 输出相机在世界系下的位置，绿色点表示
        of << center[0] << ' ' << center[1] << ' ' << center[2] << " 0 255 0" << '\n';
    }

    // 输出路标在世界系下的位置，用白色点表示
    const double *points = parameters_ + 9 * num_cameras_;
    for (int i = 0; i < num_points_; ++i)
    {
        // 取路标点的首地址
        const double *point = points + i * 3;
        // 输出路标点坐标
        for (int j = 0; j < 3; ++j)
        {
            of << point[j] << ' ';
        }
        // 加上颜色
        of << "255 255 255\n";
    }
    of.close();
}

/**
 * @brief 把camera中的相机外参，分解转换为世界系下的相机位姿
 * @param [in]  camera 相机参数的首地址，主要用到前六维 旋转和平移 inv(R_cw) t_cw
 * @param [out] angle_axis 旋转向量首地址，输出世界系下的相机姿态 R_wc
 * @param [out] center 相机原点首地址，输出世界系下的相机位置 t_wc
 * @note 这里的旋转都是旋转向量形式的，但为了表示方便，仍然写成旋转矩阵的形式
 * @details P_c = R_cwP_w + t_cw; inv(R_cw)P_c - inv(R_cw)t_cw = P_w;
 *          R_wcP_c - R_wct_cw = P_w = R_wcP_c + t_wc
 *          因此t_wc = - R_wct_cw
 * // QUERY 但是代码却和上面的公式不一样，为什么？
 */
void BAL::CameraToAngelAxisAndCenter(const double *camera, double *angle_axis,
                                     double *center) const
{
    // 创建 Vector3d 的 angle_axis_ref , 将其通过引用与 angle_axis 绑定
    VectorRef angle_axis_ref(angle_axis, 3);
    // 将camera的前三维数据赋值给 angle_axis_ref, 同时通过引用赋值了angle_axis
    // 因为camera前三维为 负的旋转向量，相当于旋转矩阵取逆，即
    // angle_axis_ref = angle_axis =  -theta n = inv(R_cw) = R_wc
    angle_axis_ref = ConstVectorRef(camera, 3);
    // inverse_rotation = -angle_axis_ref = theta n = R_cw
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;  // QUERY 这里为什么要取反
    // center = R_cw * t_cw = -R_wct_cw
    AngleAxisRotatePoint(inverse_rotation.data(), camera + 3, center);
    VectorRef(center, 3) *= -1.0;
}

void BAL::AngleAxisAndCenterToCamera(const double *angle_axis, const double *center,
                                     double *camera) const
{
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    VectorRef(camera, 3) = angle_axis_ref;

    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + 3);
    VectorRef(camera + 3, 3) *= -1.0;
}

// 对BAL类中的数据进行归一化处理
void BAL::Normalize()
{
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    // 获取路标3D点的位置  即parameters_ 中首个3d坐标的地址
    double *points = parameters_ + 9 * num_cameras_;
    // 分别处理x,y,z,一个大循环只处理一个，将point的3d点
    // 逐个放在tmp（vector<double>）中，然后求中位数
    // 最后获得所有3D点的中位数，即为中心点
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < num_points_; ++j)
        {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);
    }

    // 利用中心点，求每个点到中心点的偏差的绝对值之和
    for (int i = 0; i < num_points_; ++i)
    {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();  // 1范数，即绝对值之和
    }

    // 每个点到中心点的偏差的绝对值之和的中位数
    const double median_absolute_deviation = Median(&tmp);

    // 尺度因子
    const double scale = 100.0 / median_absolute_deviation;

    // 缩放3D点 X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i)
    {
        VectorRef point(points + 3 * i, 3);
        // 对每个3D点进行处理，VectorRef是引用，会改变原数据
        point = scale * (point - median);
    }

    // camera参数的起始地址
    double *cameras = parameters_;
    double angle_axis[3] = {0, 0, 0};
    double center[3] = {0, 0, 0};
    for (int i = 0; i < num_cameras_; ++i)
    {
        double *camera = cameras + 9 * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // 缩放相机中点 center = scale * (center - median)
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

// 给BAL类中的数据添加噪声，包括相机的旋转、平移、以及路标点的位置
void BAL::Perturb(const double rotation_sigma, const double translation_sigma,
                  const double point_sigma)
{
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *points = parameters_ + 9 * num_cameras_;
    // 给观测路标增加噪声
    if (point_sigma > 0)
    {
        for (int i = 0; i < num_points_; ++i)
        {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for (int i = 0; i < num_cameras_; ++i)
    {
        double *camera = parameters_ + 9 * i;
        double angle_axis[3];
        double center[3];
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // 添加旋转噪声
        if (rotation_sigma > 0.0)
        {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        // 添加平移噪声
        if (translation_sigma > 0.0) PerturbPoint3(translation_sigma, camera + 3);
    }
}
