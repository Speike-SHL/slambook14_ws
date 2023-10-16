#ifndef ROTATION_H
#define ROTATION_H

#include <limits>
#include <memory>

// 点乘 a·b 十四讲P43
template <typename T>
inline T DotProduct(const T *a, const T *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// 叉乘 a×b 十四讲P43
template <typename T>
inline void CrossProduct(const T *a, const T *b, T res[3])
{
    res[0] = a[1] * b[2] - a[2] * b[1];  // x
    res[1] = a[2] * b[0] - a[0] * b[2];  // y
    res[2] = a[0] * b[1] - a[1] * b[0];  // z
}

// 角轴-->四元数 见obsidian笔记
template <typename T>
inline void AngleAxis2Quaternion(const T *angle_axis, T *quaternion)
{
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;
    if (theta_squared > T(std::numeric_limits<double>::epsilon()))
    {
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta;
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else
    {
        const T k(0.5);
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

// 四元数-->角轴 见obsidian笔记
template <typename T>
inline void Quaternion2AngleAxis(const T *quaternion, T *angle_axis)
{
    const T &q0 = quaternion[0];
    const T &q1 = quaternion[1];
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_half_theta = q1 * q1 + q2 * q2 + q3 * q3;
    if (sin_squared_half_theta > T(std::numeric_limits<double>::epsilon()))
    {
        const T sin_half_theta = sqrt(sin_squared_half_theta);
        const T &cos_half_theta = q0;
        const T theta = T(2.0) * ((cos_half_theta < 0.0)
                                      ? atan2(-sin_half_theta, -cos_half_theta)
                                      : atan2(sin_half_theta, cos_half_theta));
        const T k = theta / sin_half_theta;

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else
    {
        const T k(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}

// 使用角轴旋转点 见obsidian笔记
template <typename T>
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3])
{
    const T theta_squared = DotProduct(angle_axis, angle_axis);
    if (theta_squared > T(std::numeric_limits<double>::epsilon()))
    {
        const T theta = sqrt(theta_squared);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = T(1.0) / theta;

        // 角轴的单位向量
        const T n[3] = {angle_axis[0] * theta_inverse, angle_axis[1] * theta_inverse,
                        angle_axis[2] * theta_inverse};
        // n 叉乘 pt
        T n_cross_pt[3];
        CrossProduct(n, pt, n_cross_pt);
        // n 点乘 pt (1-cos(theta))
        const T tmp = DotProduct(n, pt) * (T(1.0) - costheta);
        // 罗德里格斯公式
        result[0] = pt[0] * costheta + n_cross_pt[0] * sintheta + n[0] * tmp;
        result[1] = pt[1] * costheta + n_cross_pt[1] * sintheta + n[1] * tmp;
        result[2] = pt[2] * costheta + n_cross_pt[2] * sintheta + n[2] * tmp;
    } else
    {
        // a 叉乘 pt
        T a_cross_pt[3];
        CrossProduct(angle_axis, pt, a_cross_pt);
        // 罗德里格斯公式
        result[0] = pt[0] + a_cross_pt[0];
        result[1] = pt[1] + a_cross_pt[1];
        result[2] = pt[2] + a_cross_pt[2];
    }
}
#endif
