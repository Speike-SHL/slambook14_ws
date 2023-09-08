/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       tic_toc.h
 * @author     Speike
 * @date       2023/04/14 19:21:20
 * @brief      用于程序计数的类,参考VINS-MONO对应文件
**/

#pragma once

#include <ctime>
#include <chrono>

class TicToc
{
public:
    TicToc()
    {
        tic();
    }
    inline void tic()
    {
        start = std::chrono::system_clock::now();
    }
    // 返回值的单位是毫秒
    inline double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
