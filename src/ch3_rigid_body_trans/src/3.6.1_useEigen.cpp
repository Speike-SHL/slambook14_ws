/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       useEigen.cpp
 * @author     Speike
 * @date       2023/05/22 22:22:03
 * @brief      演示了Eigen的使用以及如何在Eigen中使用旋转矩阵、四元数、角轴和欧拉角
**/

#include <iostream>
#include <cmath>
#include <Eigen/Geometry>
using namespace std;

int main(int, char **) {
	/*****************************************************************
	 * 旋转矩阵 --> Eigen::Matrix3d
	 * 旋转向量 --> Eigen::AngleAxisd
	 * 欧拉角 --> Eigen::Vector3d
	 * 四元数 --> Eigen::Quaterniond
	 * 变换矩阵(4X4) --> Eigen::Isometry3d
	 * 仿射变换(4X4) --> Eigen::Affine3d
	 * 射影变换(4X4) --> Eigen::Projective3d 
	 *****************************************************************/
	// --------------------------------- 旋转的多种表示 ---------------------------------
	cout << "========绕Z旋转45度========" << endl;
	Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1)); // 旋转向量绕Z转45度
	cout.precision(3);
	cout << ">> 旋转向量: 角|轴\n"
		 << rotation_vector.angle() << "|" << rotation_vector.axis().transpose() << endl;

	Eigen::Matrix3d R = rotation_vector.toRotationMatrix(); // 旋转向量 --> 旋转矩阵
	cout << ">> 旋转矩阵:\n"
		 << R << endl;

	Eigen::Vector3d euler_angles = R.eulerAngles(2, 1, 0); // 旋转矩阵 --> ZYX欧拉角
	cout << ">> 欧拉角: yaw pitch roll:\n"
		 << euler_angles.transpose() << endl;

	Eigen::Quaterniond q = Eigen::Quaterniond(R); // 旋转矩阵 --> 四元数
	q = Eigen::Quaterniond(rotation_vector);	  // 旋转向量 --> 四元数
	cout << ">> 四元数: x y z w\n"
		 << q.coeffs().transpose() << endl;

	// ------------------------------- 多种方式进行坐标变换 -------------------------------
	cout << "====多种方式进行坐标变换====" << endl;
	Eigen::Vector3d p(1, 0, 0); // 一个点p(1, 0 , 0), 通过不同方式绕z轴旋转45度
	Eigen::Vector3d p_rotated = rotation_vector * p ;
	cout << ">> p(1,0,0)通过旋转向量绕z轴旋转45度后为:\n"
		 << p_rotated.transpose() << endl;

	p_rotated = R * p;
	cout << ">> p(1,0,0)通过旋转矩阵绕z轴旋转45度后为:\n"
		<< p_rotated.transpose() << endl;

	cout << ">> 欧拉角无法直接进行旋转，需要先转化为旋转矩阵" << endl;

	p_rotated = q * p; // 已经重载了计算,实际为(q *Eigen::Quaterniond(0, p.x(), p.y(), p.z()) * q.inverse())
	cout << ">> p(1,0,0)通过四元数运算绕z轴旋转45度后为:\n"
		<< p_rotated.transpose() << endl;

	// -------------------------------- 变换矩阵和坐标变换 --------------------------------
	cout << "====变换矩阵和坐标变换====" << endl;
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	T.rotate(rotation_vector);	// 在初始化为单位阵的基础上右乘旋转
	T.pretranslate(Eigen::Vector3d(1, 2, 3));
	cout << ">> 变换矩阵T:绕Z转45度,再平移(1, 2, 3)\n"
		 << T.matrix() << endl;

	p_rotated = T * p;
	cout << ">> p(1,0,0)通过变换矩阵绕Z转45度,再平移(1, 2, 3)后为:\n"
		 << p_rotated.transpose() << endl;

	// -------------------------------- 仿射变换和射影变换 --------------------------------
	// 略
	return 0;
}
