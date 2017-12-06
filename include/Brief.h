/**
 * This file is part of PiSlam.
 *
 * PiSlam is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PiSlam is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PiSlam.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2017 Carl Chatfield
 */

#ifndef PISLAM_BRIEF_H_
#define PISLAM_BRIEF_H_

#include <cmath>
#include <algorithm>

namespace pislam {

template <int vstep, int rot, int dx0, int dy0, int dx1, int dy1>
bool briefBit(uint8_t base[][vstep]) {
  constexpr float theta = (rot * M_PI / 15);
  constexpr float c = cosf(theta);
  constexpr float s = sinf(theta);

  // rotate pattern
  constexpr int rdx0 = roundf(c*dx0-s*dy0);
  constexpr int rdy0 = roundf(s*dx0+c*dy0);
  constexpr int rdx1 = roundf(c*dx1-s*dy1);
  constexpr int rdy1 = roundf(s*dx1+c*dy1);

  // don't rotate the bit pattern outside the 31x31 pixel block.
  // std::min/max not constexpr until c++14
  constexpr int mindx0 = rdx0 < -15 ? -15 : rdx0;
  constexpr int mindy0 = rdy0 < -15 ? -15 : rdy0;
  constexpr int mindx1 = rdx1 < -15 ? -15 : rdx1;
  constexpr int mindy1 = rdy1 < -15 ? -15 : rdy1;

  constexpr int cdx0 = mindx0 > 15 ? 15 : mindx0;
  constexpr int cdy0 = mindy0 > 15 ? 15 : mindy0;
  constexpr int cdx1 = mindx1 > 15 ? 15 : mindx1;
  constexpr int cdy1 = mindy1 > 15 ? 15 : mindy1;

  return base[cdy0][cdx0] < base[cdy1][cdx1];
}

/// Compute BRIEF descriptor at a particular rotation, descretized `[0..30)`.
///
template <int vstep, int rot, int words>
void briefDescribeRot(uint8_t img[][vstep], int x, int y, uint32_t descriptor[words]) {
  // gcc generates stupid code without telling it explicitly
  // to keep a pointer to the middle of the pattern for relative loads.
  uint8_t (* const base)[vstep] = (uint8_t (*)[vstep])(&img[y][x]);

  // generated bit pattern from opencv ORB.cpp
  // Experimentally directly embedding the values in instructions is
  // 3ms / 1000 features faster than using a memory lookup table.
  // See commit 'BriefPatterns for memory based implementation'
  // for details. (4dd2678a)

  uint32_t bits = 0;
  /*mean (0), correlation (0)*/
  if (briefBit<vstep, rot, 8,-3, 9,5>(base)) bits |= 1 << 0;
  /*mean (1.12461e-05), correlation (0.0437584)*/
  if (briefBit<vstep, rot, 4,2, 7,-12>(base)) bits |= 1 << 1;
  /*mean (3.37382e-05), correlation (0.0617409)*/
  if (briefBit<vstep, rot, -11,9, -8,2>(base)) bits |= 1 << 2;
  /*mean (5.62303e-05), correlation (0.0636977)*/
  if (briefBit<vstep, rot, 7,-12, 12,-13>(base)) bits |= 1 << 3;
  /*mean (0.000134953), correlation (0.085099)*/
  if (briefBit<vstep, rot, 2,-13, 2,12>(base)) bits |= 1 << 4;
  /*mean (0.000528565), correlation (0.0857175)*/
  if (briefBit<vstep, rot, 1,-7, 1,6>(base)) bits |= 1 << 5;
  /*mean (0.0188821), correlation (0.0985774)*/
  if (briefBit<vstep, rot, -2,-10, -2,-4>(base)) bits |= 1 << 6;
  /*mean (0.0363135), correlation (0.0899616)*/
  if (briefBit<vstep, rot, -13,-13, -11,-8>(base)) bits |= 1 << 7;
  /*mean (0.121806), correlation (0.099849)*/
  if (briefBit<vstep, rot, -13,-3, -12,-9>(base)) bits |= 1 << 8;
  /*mean (0.122065), correlation (0.093285)*/
  if (briefBit<vstep, rot, 10,4, 11,9>(base)) bits |= 1 << 9;
  /*mean (0.162787), correlation (0.0942748)*/
  if (briefBit<vstep, rot, -13,-8, -8,-9>(base)) bits |= 1 << 10;
  /*mean (0.21561), correlation (0.0974438)*/
  if (briefBit<vstep, rot, -11,7, -9,12>(base)) bits |= 1 << 11;
  /*mean (0.160583), correlation (0.130064)*/
  if (briefBit<vstep, rot, 7,7, 12,6>(base)) bits |= 1 << 12;
  /*mean (0.228171), correlation (0.132998)*/
  if (briefBit<vstep, rot, -4,-5, -3,0>(base)) bits |= 1 << 13;
  /*mean (0.00997526), correlation (0.145926)*/
  if (briefBit<vstep, rot, -13,2, -12,-3>(base)) bits |= 1 << 14;
  /*mean (0.198234), correlation (0.143636)*/
  if (briefBit<vstep, rot, -9,0, -7,5>(base)) bits |= 1 << 15;
  /*mean (0.0676226), correlation (0.16689)*/
  if (briefBit<vstep, rot, 12,-6, 12,-1>(base)) bits |= 1 << 16;
  /*mean (0.166847), correlation (0.171682)*/
  if (briefBit<vstep, rot, -3,6, -2,12>(base)) bits |= 1 << 17;
  /*mean (0.101215), correlation (0.179716)*/
  if (briefBit<vstep, rot, -6,-13, -4,-8>(base)) bits |= 1 << 18;
  /*mean (0.200641), correlation (0.192279)*/
  if (briefBit<vstep, rot, 11,-13, 12,-8>(base)) bits |= 1 << 19;
  /*mean (0.205106), correlation (0.186848)*/
  if (briefBit<vstep, rot, 4,7, 5,1>(base)) bits |= 1 << 20;
  /*mean (0.234908), correlation (0.192319)*/
  if (briefBit<vstep, rot, 5,-3, 10,-3>(base)) bits |= 1 << 21;
  /*mean (0.0709964), correlation (0.210872)*/
  if (briefBit<vstep, rot, 3,-7, 6,12>(base)) bits |= 1 << 22;
  /*mean (0.0939834), correlation (0.212589)*/
  if (briefBit<vstep, rot, -8,-7, -6,-2>(base)) bits |= 1 << 23;
  /*mean (0.127778), correlation (0.20866)*/
  if (briefBit<vstep, rot, -2,11, -1,-10>(base)) bits |= 1 << 24;
  /*mean (0.14783), correlation (0.206356)*/
  if (briefBit<vstep, rot, -13,12, -8,10>(base)) bits |= 1 << 25;
  /*mean (0.182141), correlation (0.198942)*/
  if (briefBit<vstep, rot, -7,3, -5,-3>(base)) bits |= 1 << 26;
  /*mean (0.188237), correlation (0.21384)*/
  if (briefBit<vstep, rot, -4,2, -3,7>(base)) bits |= 1 << 27;
  /*mean (0.14865), correlation (0.23571)*/
  if (briefBit<vstep, rot, -10,-12, -6,11>(base)) bits |= 1 << 28;
  /*mean (0.222312), correlation (0.23324)*/
  if (briefBit<vstep, rot, 5,-12, 6,-7>(base)) bits |= 1 << 29;
  /*mean (0.229082), correlation (0.23389)*/
  if (briefBit<vstep, rot, 5,-6, 7,-1>(base)) bits |= 1 << 30;
  /*mean (0.241577), correlation (0.215286)*/
  if (briefBit<vstep, rot, 1,0, 4,-5>(base)) bits |= 1 << 31;

  descriptor[0] = bits;
  if (words == 1) {
    return;
  }
  bits = 0;

  /*mean (0.00338507), correlation (0.251373)*/
  if (briefBit<vstep, rot, 9,11, 11,-13>(base)) bits |= 1 << 0;
  /*mean (0.131005), correlation (0.257622)*/
  if (briefBit<vstep, rot, 4,7, 4,12>(base)) bits |= 1 << 1;
  /*mean (0.152755), correlation (0.255205)*/
  if (briefBit<vstep, rot, 2,-1, 4,4>(base)) bits |= 1 << 2;
  /*mean (0.182771), correlation (0.244867)*/
  if (briefBit<vstep, rot, -4,-12, -2,7>(base)) bits |= 1 << 3;
  /*mean (0.186898), correlation (0.23901)*/
  if (briefBit<vstep, rot, -8,-5, -7,-10>(base)) bits |= 1 << 4;
  /*mean (0.226226), correlation (0.258255)*/
  if (briefBit<vstep, rot, 4,11, 9,12>(base)) bits |= 1 << 5;
  /*mean (0.0897886), correlation (0.274827)*/
  if (briefBit<vstep, rot, 0,-8, 1,-13>(base)) bits |= 1 << 6;
  /*mean (0.148774), correlation (0.28065)*/
  if (briefBit<vstep, rot, -13,-2, -8,2>(base)) bits |= 1 << 7;
  /*mean (0.153048), correlation (0.283063)*/
  if (briefBit<vstep, rot, -3,-2, -2,3>(base)) bits |= 1 << 8;
  /*mean (0.169523), correlation (0.278248)*/
  if (briefBit<vstep, rot, -6,9, -4,-9>(base)) bits |= 1 << 9;
  /*mean (0.225337), correlation (0.282851)*/
  if (briefBit<vstep, rot, 8,12, 10,7>(base)) bits |= 1 << 10;
  /*mean (0.226687), correlation (0.278734)*/
  if (briefBit<vstep, rot, 0,9, 1,3>(base)) bits |= 1 << 11;
  /*mean (0.00693882), correlation (0.305161)*/
  if (briefBit<vstep, rot, 7,-5, 11,-10>(base)) bits |= 1 << 12;
  /*mean (0.0227283), correlation (0.300181)*/
  if (briefBit<vstep, rot, -13,-6, -11,0>(base)) bits |= 1 << 13;
  /*mean (0.125517), correlation (0.31089)*/
  if (briefBit<vstep, rot, 10,7, 12,1>(base)) bits |= 1 << 14;
  /*mean (0.131748), correlation (0.312779)*/
  if (briefBit<vstep, rot, -6,-3, -6,12>(base)) bits |= 1 << 15;
  /*mean (0.144827), correlation (0.292797)*/
  if (briefBit<vstep, rot, 10,-9, 12,-4>(base)) bits |= 1 << 16;
  /*mean (0.149202), correlation (0.308918)*/
  if (briefBit<vstep, rot, -13,8, -8,-12>(base)) bits |= 1 << 17;
  /*mean (0.160909), correlation (0.310013)*/
  if (briefBit<vstep, rot, -13,0, -8,-4>(base)) bits |= 1 << 18;
  /*mean (0.177755), correlation (0.309394)*/
  if (briefBit<vstep, rot, 3,3, 7,8>(base)) bits |= 1 << 19;
  /*mean (0.212337), correlation (0.310315)*/
  if (briefBit<vstep, rot, 5,7, 10,-7>(base)) bits |= 1 << 20;
  /*mean (0.214429), correlation (0.311933)*/
  if (briefBit<vstep, rot, -1,7, 1,-12>(base)) bits |= 1 << 21;
  /*mean (0.235807), correlation (0.313104)*/
  if (briefBit<vstep, rot, 3,-10, 5,6>(base)) bits |= 1 << 22;
  /*mean (0.00494827), correlation (0.344948)*/
  if (briefBit<vstep, rot, 2,-4, 3,-10>(base)) bits |= 1 << 23;
  /*mean (0.0549145), correlation (0.344675)*/
  if (briefBit<vstep, rot, -13,0, -13,5>(base)) bits |= 1 << 24;
  /*mean (0.103385), correlation (0.342715)*/
  if (briefBit<vstep, rot, -13,-7, -12,12>(base)) bits |= 1 << 25;
  /*mean (0.134222), correlation (0.322922)*/
  if (briefBit<vstep, rot, -13,3, -11,8>(base)) bits |= 1 << 26;
  /*mean (0.153284), correlation (0.337061)*/
  if (briefBit<vstep, rot, -7,12, -4,7>(base)) bits |= 1 << 27;
  /*mean (0.154881), correlation (0.329257)*/
  if (briefBit<vstep, rot, 6,-10, 12,8>(base)) bits |= 1 << 28;
  /*mean (0.200967), correlation (0.33312)*/
  if (briefBit<vstep, rot, -9,-1, -7,-6>(base)) bits |= 1 << 29;
  /*mean (0.201518), correlation (0.340635)*/
  if (briefBit<vstep, rot, -2,-5, 0,12>(base)) bits |= 1 << 30;
  /*mean (0.207805), correlation (0.335631)*/
  if (briefBit<vstep, rot, -12,5, -7,5>(base)) bits |= 1 << 31;

  descriptor[1] = bits;
  if (words == 2) {
    return;
  }
  bits = 0;

  /*mean (0.224438), correlation (0.34504)*/
  if (briefBit<vstep, rot, 3,-10, 8,-13>(base)) bits |= 1 << 0;
  /*mean (0.239361), correlation (0.338053)*/
  if (briefBit<vstep, rot, -7,-7, -4,5>(base)) bits |= 1 << 1;
  /*mean (0.240744), correlation (0.344322)*/
  if (briefBit<vstep, rot, -3,-2, -1,-7>(base)) bits |= 1 << 2;
  /*mean (0.242949), correlation (0.34145)*/
  if (briefBit<vstep, rot, 2,9, 5,-11>(base)) bits |= 1 << 3;
  /*mean (0.244028), correlation (0.336861)*/
  if (briefBit<vstep, rot, -11,-13, -5,-13>(base)) bits |= 1 << 4;
  /*mean (0.247571), correlation (0.343684)*/
  if (briefBit<vstep, rot, -1,6, 0,-1>(base)) bits |= 1 << 5;
  /*mean (0.000697256), correlation (0.357265)*/
  if (briefBit<vstep, rot, 5,-3, 5,2>(base)) bits |= 1 << 6;
  /*mean (0.00213675), correlation (0.373827)*/
  if (briefBit<vstep, rot, -4,-13, -4,12>(base)) bits |= 1 << 7;
  /*mean (0.0126856), correlation (0.373938)*/
  if (briefBit<vstep, rot, -9,-6, -9,6>(base)) bits |= 1 << 8;
  /*mean (0.0152497), correlation (0.364237)*/
  if (briefBit<vstep, rot, -12,-10, -8,-4>(base)) bits |= 1 << 9;
  /*mean (0.0299933), correlation (0.345292)*/
  if (briefBit<vstep, rot, 10,2, 12,-3>(base)) bits |= 1 << 10;
  /*mean (0.0307242), correlation (0.366299)*/
  if (briefBit<vstep, rot, 7,12, 12,12>(base)) bits |= 1 << 11;
  /*mean (0.0534975), correlation (0.368357)*/
  if (briefBit<vstep, rot, -7,-13, -6,5>(base)) bits |= 1 << 12;
  /*mean (0.099865), correlation (0.372276)*/
  if (briefBit<vstep, rot, -4,9, -3,4>(base)) bits |= 1 << 13;
  /*mean (0.117083), correlation (0.364529)*/
  if (briefBit<vstep, rot, 7,-1, 12,2>(base)) bits |= 1 << 14;
  /*mean (0.126125), correlation (0.369606)*/
  if (briefBit<vstep, rot, -7,6, -5,1>(base)) bits |= 1 << 15;
  /*mean (0.130364), correlation (0.358502)*/
  if (briefBit<vstep, rot, -13,11, -12,5>(base)) bits |= 1 << 16;
  /*mean (0.131691), correlation (0.375531)*/
  if (briefBit<vstep, rot, -3,7, -2,-6>(base)) bits |= 1 << 17;
  /*mean (0.160166), correlation (0.379508)*/
  if (briefBit<vstep, rot, 7,-8, 12,-7>(base)) bits |= 1 << 18;
  /*mean (0.167848), correlation (0.353343)*/
  if (briefBit<vstep, rot, -13,-7, -11,-12>(base)) bits |= 1 << 19;
  /*mean (0.183378), correlation (0.371916)*/
  if (briefBit<vstep, rot, 1,-3, 12,12>(base)) bits |= 1 << 20;
  /*mean (0.228711), correlation (0.371761)*/
  if (briefBit<vstep, rot, 2,-6, 3,0>(base)) bits |= 1 << 21;
  /*mean (0.247211), correlation (0.364063)*/
  if (briefBit<vstep, rot, -4,3, -2,-13>(base)) bits |= 1 << 22;
  /*mean (0.249325), correlation (0.378139)*/
  if (briefBit<vstep, rot, -1,-13, 1,9>(base)) bits |= 1 << 23;
  /*mean (0.000652272), correlation (0.411682)*/
  if (briefBit<vstep, rot, 7,1, 8,-6>(base)) bits |= 1 << 24;
  /*mean (0.00248538), correlation (0.392988)*/
  if (briefBit<vstep, rot, 1,-1, 3,12>(base)) bits |= 1 << 25;
  /*mean (0.0206815), correlation (0.386106)*/
  if (briefBit<vstep, rot, 9,1, 12,6>(base)) bits |= 1 << 26;
  /*mean (0.0364485), correlation (0.410752)*/
  if (briefBit<vstep, rot, -1,-9, -1,3>(base)) bits |= 1 << 27;
  /*mean (0.0376068), correlation (0.398374)*/
  if (briefBit<vstep, rot, -13,-13, -10,5>(base)) bits |= 1 << 28;
  /*mean (0.0424202), correlation (0.405663)*/
  if (briefBit<vstep, rot, 7,7, 10,12>(base)) bits |= 1 << 29;
  /*mean (0.0942645), correlation (0.410422)*/
  if (briefBit<vstep, rot, 12,-5, 12,9>(base)) bits |= 1 << 30;
  /*mean (0.1074), correlation (0.413224)*/
  if (briefBit<vstep, rot, 6,3, 7,11>(base)) bits |= 1 << 31;

  descriptor[2] = bits;
  if (words == 3) {
    return;
  }
  bits = 0;

  /*mean (0.109256), correlation (0.408646)*/
  if (briefBit<vstep, rot, 5,-13, 6,10>(base)) bits |= 1 << 0;
  /*mean (0.131691), correlation (0.416076)*/
  if (briefBit<vstep, rot, 2,-12, 2,3>(base)) bits |= 1 << 1;
  /*mean (0.165081), correlation (0.417569)*/
  if (briefBit<vstep, rot, 3,8, 4,-6>(base)) bits |= 1 << 2;
  /*mean (0.171874), correlation (0.408471)*/
  if (briefBit<vstep, rot, 2,6, 12,-13>(base)) bits |= 1 << 3;
  /*mean (0.175146), correlation (0.41296)*/
  if (briefBit<vstep, rot, 9,-12, 10,3>(base)) bits |= 1 << 4;
  /*mean (0.183682), correlation (0.402956)*/
  if (briefBit<vstep, rot, -8,4, -7,9>(base)) bits |= 1 << 5;
  /*mean (0.184672), correlation (0.416125)*/
  if (briefBit<vstep, rot, -11,12, -4,-6>(base)) bits |= 1 << 6;
  /*mean (0.191487), correlation (0.386696)*/
  if (briefBit<vstep, rot, 1,12, 2,-8>(base)) bits |= 1 << 7;
  /*mean (0.192668), correlation (0.394771)*/
  if (briefBit<vstep, rot, 6,-9, 7,-4>(base)) bits |= 1 << 8;
  /*mean (0.200157), correlation (0.408303)*/
  if (briefBit<vstep, rot, 2,3, 3,-2>(base)) bits |= 1 << 9;
  /*mean (0.204588), correlation (0.411762)*/
  if (briefBit<vstep, rot, 6,3, 11,0>(base)) bits |= 1 << 10;
  /*mean (0.205904), correlation (0.416294)*/
  if (briefBit<vstep, rot, 3,-3, 8,-8>(base)) bits |= 1 << 11;
  /*mean (0.213237), correlation (0.409306)*/
  if (briefBit<vstep, rot, 7,8, 9,3>(base)) bits |= 1 << 12;
  /*mean (0.243444), correlation (0.395069)*/
  if (briefBit<vstep, rot, -11,-5, -6,-4>(base)) bits |= 1 << 13;
  /*mean (0.247672), correlation (0.413392)*/
  if (briefBit<vstep, rot, -10,11, -5,10>(base)) bits |= 1 << 14;
  /*mean (0.24774), correlation (0.411416)*/
  if (briefBit<vstep, rot, -5,-8, -3,12>(base)) bits |= 1 << 15;
  /*mean (0.00213675), correlation (0.454003)*/
  if (briefBit<vstep, rot, -10,5, -9,0>(base)) bits |= 1 << 16;
  /*mean (0.0293635), correlation (0.455368)*/
  if (briefBit<vstep, rot, 8,-1, 12,-6>(base)) bits |= 1 << 17;
  /*mean (0.0404971), correlation (0.457393)*/
  if (briefBit<vstep, rot, 4,-6, 6,-11>(base)) bits |= 1 << 18;
  /*mean (0.0481107), correlation (0.448364)*/
  if (briefBit<vstep, rot, -10,12, -8,7>(base)) bits |= 1 << 19;
  /*mean (0.050641), correlation (0.455019)*/
  if (briefBit<vstep, rot, 4,-2, 6,7>(base)) bits |= 1 << 20;
  /*mean (0.0525978), correlation (0.44338)*/
  if (briefBit<vstep, rot, -2,0, -2,12>(base)) bits |= 1 << 21;
  /*mean (0.0629667), correlation (0.457096)*/
  if (briefBit<vstep, rot, -5,-8, -5,2>(base)) bits |= 1 << 22;
  /*mean (0.0653846), correlation (0.445623)*/
  if (briefBit<vstep, rot, 7,-6, 10,12>(base)) bits |= 1 << 23;
  /*mean (0.0858749), correlation (0.449789)*/
  if (briefBit<vstep, rot, -9,-13, -8,-8>(base)) bits |= 1 << 24;
  /*mean (0.122402), correlation (0.450201)*/
  if (briefBit<vstep, rot, -5,-13, -5,-2>(base)) bits |= 1 << 25;
  /*mean (0.125416), correlation (0.453224)*/
  if (briefBit<vstep, rot, 8,-8, 9,-13>(base)) bits |= 1 << 26;
  /*mean (0.130128), correlation (0.458724)*/
  if (briefBit<vstep, rot, -9,-11, -9,0>(base)) bits |= 1 << 27;
  /*mean (0.132467), correlation (0.440133)*/
  if (briefBit<vstep, rot, 1,-8, 1,-2>(base)) bits |= 1 << 28;
  /*mean (0.132692), correlation (0.454)*/
  if (briefBit<vstep, rot, 7,-4, 9,1>(base)) bits |= 1 << 29;
  /*mean (0.135695), correlation (0.455739)*/
  if (briefBit<vstep, rot, -2,1, -1,-4>(base)) bits |= 1 << 30;
  /*mean (0.142904), correlation (0.446114)*/
  if (briefBit<vstep, rot, 11,-6, 12,-11>(base)) bits |= 1 << 31;

  descriptor[3] = bits;
  if (words == 4) {
    return;
  }
  bits = 0;

  /*mean (0.146165), correlation (0.451473)*/
  if (briefBit<vstep, rot, -12,-9, -6,4>(base)) bits |= 1 << 0;
  /*mean (0.147627), correlation (0.456643)*/
  if (briefBit<vstep, rot, 3,7, 7,12>(base)) bits |= 1 << 1;
  /*mean (0.152901), correlation (0.455036)*/
  if (briefBit<vstep, rot, 5,5, 10,8>(base)) bits |= 1 << 2;
  /*mean (0.167083), correlation (0.459315)*/
  if (briefBit<vstep, rot, 0,-4, 2,8>(base)) bits |= 1 << 3;
  /*mean (0.173234), correlation (0.454706)*/
  if (briefBit<vstep, rot, -9,12, -5,-13>(base)) bits |= 1 << 4;
  /*mean (0.18312), correlation (0.433855)*/
  if (briefBit<vstep, rot, 0,7, 2,12>(base)) bits |= 1 << 5;
  /*mean (0.185504), correlation (0.443838)*/
  if (briefBit<vstep, rot, -1,2, 1,7>(base)) bits |= 1 << 6;
  /*mean (0.185706), correlation (0.451123)*/
  if (briefBit<vstep, rot, 5,11, 7,-9>(base)) bits |= 1 << 7;
  /*mean (0.188968), correlation (0.455808)*/
  if (briefBit<vstep, rot, 3,5, 6,-8>(base)) bits |= 1 << 8;
  /*mean (0.191667), correlation (0.459128)*/
  if (briefBit<vstep, rot, -13,-4, -8,9>(base)) bits |= 1 << 9;
  /*mean (0.193196), correlation (0.458364)*/
  if (briefBit<vstep, rot, -5,9, -3,-3>(base)) bits |= 1 << 10;
  /*mean (0.196536), correlation (0.455782)*/
  if (briefBit<vstep, rot, -4,-7, -3,-12>(base)) bits |= 1 << 11;
  /*mean (0.1972), correlation (0.450481)*/
  if (briefBit<vstep, rot, 6,5, 8,0>(base)) bits |= 1 << 12;
  /*mean (0.199438), correlation (0.458156)*/
  if (briefBit<vstep, rot, -7,6, -6,12>(base)) bits |= 1 << 13;
  /*mean (0.211224), correlation (0.449548)*/
  if (briefBit<vstep, rot, -13,6, -5,-2>(base)) bits |= 1 << 14;
  /*mean (0.211718), correlation (0.440606)*/
  if (briefBit<vstep, rot, 1,-10, 3,10>(base)) bits |= 1 << 15;
  /*mean (0.213034), correlation (0.443177)*/
  if (briefBit<vstep, rot, 4,1, 8,-4>(base)) bits |= 1 << 16;
  /*mean (0.234334), correlation (0.455304)*/
  if (briefBit<vstep, rot, -2,-2, 2,-13>(base)) bits |= 1 << 17;
  /*mean (0.235684), correlation (0.443436)*/
  if (briefBit<vstep, rot, 2,-12, 12,12>(base)) bits |= 1 << 18;
  /*mean (0.237674), correlation (0.452525)*/
  if (briefBit<vstep, rot, -2,-13, 0,-6>(base)) bits |= 1 << 19;
  /*mean (0.23962), correlation (0.444824)*/
  if (briefBit<vstep, rot, 4,1, 9,3>(base)) bits |= 1 << 20;
  /*mean (0.248459), correlation (0.439621)*/
  if (briefBit<vstep, rot, -6,-10, -3,-5>(base)) bits |= 1 << 21;
  /*mean (0.249505), correlation (0.456666)*/
  if (briefBit<vstep, rot, -3,-13, -1,1>(base)) bits |= 1 << 22;
  /*mean (0.00119208), correlation (0.495466)*/
  if (briefBit<vstep, rot, 7,5, 12,-11>(base)) bits |= 1 << 23;
  /*mean (0.00372245), correlation (0.484214)*/
  if (briefBit<vstep, rot, 4,-2, 5,-7>(base)) bits |= 1 << 24;
  /*mean (0.00741116), correlation (0.499854)*/
  if (briefBit<vstep, rot, -13,9, -9,-5>(base)) bits |= 1 << 25;
  /*mean (0.0208952), correlation (0.499773)*/
  if (briefBit<vstep, rot, 7,1, 8,6>(base)) bits |= 1 << 26;
  /*mean (0.0220085), correlation (0.501609)*/
  if (briefBit<vstep, rot, 7,-8, 7,6>(base)) bits |= 1 << 27;
  /*mean (0.0233806), correlation (0.496568)*/
  if (briefBit<vstep, rot, -7,-4, -7,1>(base)) bits |= 1 << 28;
  /*mean (0.0236505), correlation (0.489719)*/
  if (briefBit<vstep, rot, -8,11, -7,-8>(base)) bits |= 1 << 29;
  /*mean (0.0268781), correlation (0.503487)*/
  if (briefBit<vstep, rot, -13,6, -12,-8>(base)) bits |= 1 << 30;
  /*mean (0.0323324), correlation (0.501938)*/
  if (briefBit<vstep, rot, 2,4, 3,9>(base)) bits |= 1 << 31;

  descriptor[4] = bits;
  if (words == 5) {
    return;
  }
  bits = 0;

  /*mean (0.0399235), correlation (0.494029)*/
  if (briefBit<vstep, rot, 10,-5, 12,3>(base)) bits |= 1 << 0;
  /*mean (0.0420153), correlation (0.486579)*/
  if (briefBit<vstep, rot, -6,-5, -6,7>(base)) bits |= 1 << 1;
  /*mean (0.0548021), correlation (0.484237)*/
  if (briefBit<vstep, rot, 8,-3, 9,-8>(base)) bits |= 1 << 2;
  /*mean (0.0616622), correlation (0.496642)*/
  if (briefBit<vstep, rot, 2,-12, 2,8>(base)) bits |= 1 << 3;
  /*mean (0.0627755), correlation (0.498563)*/
  if (briefBit<vstep, rot, -11,-2, -10,3>(base)) bits |= 1 << 4;
  /*mean (0.0829622), correlation (0.495491)*/
  if (briefBit<vstep, rot, -12,-13, -7,-9>(base)) bits |= 1 << 5;
  /*mean (0.0843342), correlation (0.487146)*/
  if (briefBit<vstep, rot, -11,0, -10,-5>(base)) bits |= 1 << 6;
  /*mean (0.0929937), correlation (0.502315)*/
  if (briefBit<vstep, rot, 5,-3, 11,8>(base)) bits |= 1 << 7;
  /*mean (0.113327), correlation (0.48941)*/
  if (briefBit<vstep, rot, -2,-13, -1,12>(base)) bits |= 1 << 8;
  /*mean (0.132119), correlation (0.467268)*/
  if (briefBit<vstep, rot, -1,-8, 0,9>(base)) bits |= 1 << 9;
  /*mean (0.136269), correlation (0.498771)*/
  if (briefBit<vstep, rot, -13,-11, -12,-5>(base)) bits |= 1 << 10;
  /*mean (0.142173), correlation (0.498714)*/
  if (briefBit<vstep, rot, -10,-2, -10,11>(base)) bits |= 1 << 11;
  /*mean (0.144141), correlation (0.491973)*/
  if (briefBit<vstep, rot, -3,9, -2,-13>(base)) bits |= 1 << 12;
  /*mean (0.14892), correlation (0.500782)*/
  if (briefBit<vstep, rot, 2,-3, 3,2>(base)) bits |= 1 << 13;
  /*mean (0.150371), correlation (0.498211)*/
  if (briefBit<vstep, rot, -9,-13, -4,0>(base)) bits |= 1 << 14;
  /*mean (0.152159), correlation (0.495547)*/
  if (briefBit<vstep, rot, -4,6, -3,-10>(base)) bits |= 1 << 15;
  /*mean (0.156152), correlation (0.496925)*/
  if (briefBit<vstep, rot, -4,12, -2,-7>(base)) bits |= 1 << 16;
  /*mean (0.15749), correlation (0.499222)*/
  if (briefBit<vstep, rot, -6,-11, -4,9>(base)) bits |= 1 << 17;
  /*mean (0.159211), correlation (0.503821)*/
  if (briefBit<vstep, rot, 6,-3, 6,11>(base)) bits |= 1 << 18;
  /*mean (0.162427), correlation (0.501907)*/
  if (briefBit<vstep, rot, -13,11, -5,5>(base)) bits |= 1 << 19;
  /*mean (0.16652), correlation (0.497632)*/
  if (briefBit<vstep, rot, 11,11, 12,6>(base)) bits |= 1 << 20;
  /*mean (0.169141), correlation (0.484474)*/
  if (briefBit<vstep, rot, 7,-5, 12,-2>(base)) bits |= 1 << 21;
  /*mean (0.169456), correlation (0.495339)*/
  if (briefBit<vstep, rot, -1,12, 0,7>(base)) bits |= 1 << 22;
  /*mean (0.171457), correlation (0.487251)*/
  if (briefBit<vstep, rot, -4,-8, -3,-2>(base)) bits |= 1 << 23;
  /*mean (0.175), correlation (0.500024)*/
  if (briefBit<vstep, rot, -7,1, -6,7>(base)) bits |= 1 << 24;
  /*mean (0.175866), correlation (0.497523)*/
  if (briefBit<vstep, rot, -13,-12, -8,-13>(base)) bits |= 1 << 25;
  /*mean (0.178273), correlation (0.501854)*/
  if (briefBit<vstep, rot, -7,-2, -6,-8>(base)) bits |= 1 << 26;
  /*mean (0.181107), correlation (0.494888)*/
  if (briefBit<vstep, rot, -8,5, -6,-9>(base)) bits |= 1 << 27;
  /*mean (0.190227), correlation (0.482557)*/
  if (briefBit<vstep, rot, -5,-1, -4,5>(base)) bits |= 1 << 28;
  /*mean (0.196739), correlation (0.496503)*/
  if (briefBit<vstep, rot, -13,7, -8,10>(base)) bits |= 1 << 29;
  /*mean (0.19973), correlation (0.499759)*/
  if (briefBit<vstep, rot, 1,5, 5,-13>(base)) bits |= 1 << 30;
  /*mean (0.204465), correlation (0.49873)*/
  if (briefBit<vstep, rot, 1,0, 10,-13>(base)) bits |= 1 << 31;

  descriptor[5] = bits;
  if (words == 6) {
    return;
  }
  bits = 0;

  /*mean (0.209334), correlation (0.49063)*/
  if (briefBit<vstep, rot, 9,12, 10,-1>(base)) bits |= 1 << 0;
  /*mean (0.211134), correlation (0.503011)*/
  if (briefBit<vstep, rot, 5,-8, 10,-9>(base)) bits |= 1 << 1;
  /*mean (0.212), correlation (0.499414)*/
  if (briefBit<vstep, rot, -1,11, 1,-13>(base)) bits |= 1 << 2;
  /*mean (0.212168), correlation (0.480739)*/
  if (briefBit<vstep, rot, -9,-3, -6,2>(base)) bits |= 1 << 3;
  /*mean (0.212731), correlation (0.502523)*/
  if (briefBit<vstep, rot, -1,-10, 1,12>(base)) bits |= 1 << 4;
  /*mean (0.21327), correlation (0.489786)*/
  if (briefBit<vstep, rot, -13,1, -8,-10>(base)) bits |= 1 << 5;
  /*mean (0.214159), correlation (0.488246)*/
  if (briefBit<vstep, rot, 8,-11, 10,-6>(base)) bits |= 1 << 6;
  /*mean (0.216993), correlation (0.50287)*/
  if (briefBit<vstep, rot, 2,-13, 3,-6>(base)) bits |= 1 << 7;
  /*mean (0.223639), correlation (0.470502)*/
  if (briefBit<vstep, rot, 7,-13, 12,-9>(base)) bits |= 1 << 8;
  /*mean (0.224089), correlation (0.500852)*/
  if (briefBit<vstep, rot, -10,-10, -5,-7>(base)) bits |= 1 << 9;
  /*mean (0.228666), correlation (0.502629)*/
  if (briefBit<vstep, rot, -10,-8, -8,-13>(base)) bits |= 1 << 10;
  /*mean (0.22906), correlation (0.498305)*/
  if (briefBit<vstep, rot, 4,-6, 8,5>(base)) bits |= 1 << 11;
  /*mean (0.233378), correlation (0.503825)*/
  if (briefBit<vstep, rot, 3,12, 8,-13>(base)) bits |= 1 << 12;
  /*mean (0.234323), correlation (0.476692)*/
  if (briefBit<vstep, rot, -4,2, -3,-3>(base)) bits |= 1 << 13;
  /*mean (0.236392), correlation (0.475462)*/
  if (briefBit<vstep, rot, 5,-13, 10,-12>(base)) bits |= 1 << 14;
  /*mean (0.236842), correlation (0.504132)*/
  if (briefBit<vstep, rot, 4,-13, 5,-1>(base)) bits |= 1 << 15;
  /*mean (0.236977), correlation (0.497739)*/
  if (briefBit<vstep, rot, -9,9, -4,3>(base)) bits |= 1 << 16;
  /*mean (0.24314), correlation (0.499398)*/
  if (briefBit<vstep, rot, 0,3, 3,-9>(base)) bits |= 1 << 17;
  /*mean (0.243297), correlation (0.489447)*/
  if (briefBit<vstep, rot, -12,1, -6,1>(base)) bits |= 1 << 18;
  /*mean (0.00155196), correlation (0.553496)*/
  if (briefBit<vstep, rot, 3,2, 4,-8>(base)) bits |= 1 << 19;
  /*mean (0.00239541), correlation (0.54297)*/
  if (briefBit<vstep, rot, -10,-10, -10,9>(base)) bits |= 1 << 20;
  /*mean (0.0034413), correlation (0.544361)*/
  if (briefBit<vstep, rot, 8,-13, 12,12>(base)) bits |= 1 << 21;
  /*mean (0.003565), correlation (0.551225)*/
  if (briefBit<vstep, rot, -8,-12, -6,-5>(base)) bits |= 1 << 22;
  /*mean (0.00835583), correlation (0.55285)*/
  if (briefBit<vstep, rot, 2,2, 3,7>(base)) bits |= 1 << 23;
  /*mean (0.00885065), correlation (0.540913)*/
  if (briefBit<vstep, rot, 10,6, 11,-8>(base)) bits |= 1 << 24;
  /*mean (0.0101552), correlation (0.551085)*/
  if (briefBit<vstep, rot, 6,8, 8,-12>(base)) bits |= 1 << 25;
  /*mean (0.0102227), correlation (0.533635)*/
  if (briefBit<vstep, rot, -7,10, -6,5>(base)) bits |= 1 << 26;
  /*mean (0.0110211), correlation (0.543121)*/
  if (briefBit<vstep, rot, -3,-9, -3,9>(base)) bits |= 1 << 27;
  /*mean (0.0113473), correlation (0.550173)*/
  if (briefBit<vstep, rot, -1,-13, -1,5>(base)) bits |= 1 << 28;
  /*mean (0.0140913), correlation (0.554774)*/
  if (briefBit<vstep, rot, -3,-7, -3,4>(base)) bits |= 1 << 29;
  /*mean (0.017049), correlation (0.55461)*/
  if (briefBit<vstep, rot, -8,-2, -8,3>(base)) bits |= 1 << 30;
  /*mean (0.01778), correlation (0.546921)*/
  if (briefBit<vstep, rot, 4,2, 12,12>(base)) bits |= 1 << 31;

  descriptor[6] = bits;
  if (words == 7) {
    return;
  }
  bits = 0;

  /*mean (0.0224022), correlation (0.549667)*/
  if (briefBit<vstep, rot, 2,-5, 3,11>(base)) bits |= 1 << 0;
  /*mean (0.029161), correlation (0.546295)*/
  if (briefBit<vstep, rot, 6,-9, 11,-13>(base)) bits |= 1 << 1;
  /*mean (0.0303081), correlation (0.548599)*/
  if (briefBit<vstep, rot, 3,-1, 7,12>(base)) bits |= 1 << 2;
  /*mean (0.0355151), correlation (0.523943)*/
  if (briefBit<vstep, rot, 11,-1, 12,4>(base)) bits |= 1 << 3;
  /*mean (0.0417904), correlation (0.543395)*/
  if (briefBit<vstep, rot, -3,0, -3,6>(base)) bits |= 1 << 4;
  /*mean (0.0487292), correlation (0.542818)*/
  if (briefBit<vstep, rot, 4,-11, 4,12>(base)) bits |= 1 << 5;
  /*mean (0.0575124), correlation (0.554888)*/
  if (briefBit<vstep, rot, 2,-4, 2,1>(base)) bits |= 1 << 6;
  /*mean (0.0594242), correlation (0.544026)*/
  if (briefBit<vstep, rot, -10,-6, -8,1>(base)) bits |= 1 << 7;
  /*mean (0.0597391), correlation (0.550524)*/
  if (briefBit<vstep, rot, -13,7, -11,1>(base)) bits |= 1 << 8;
  /*mean (0.0608974), correlation (0.55383)*/
  if (briefBit<vstep, rot, -13,12, -11,-13>(base)) bits |= 1 << 9;
  /*mean (0.065126), correlation (0.552006)*/
  if (briefBit<vstep, rot, 6,0, 11,-13>(base)) bits |= 1 << 10;
  /*mean (0.074224), correlation (0.546372)*/
  if (briefBit<vstep, rot, 0,-1, 1,4>(base)) bits |= 1 << 11;
  /*mean (0.0808592), correlation (0.554875)*/
  if (briefBit<vstep, rot, -13,3, -9,-2>(base)) bits |= 1 << 12;
  /*mean (0.0883378), correlation (0.551178)*/
  if (briefBit<vstep, rot, -9,8, -6,-3>(base)) bits |= 1 << 13;
  /*mean (0.0901035), correlation (0.548446)*/
  if (briefBit<vstep, rot, -13,-6, -8,-2>(base)) bits |= 1 << 14;
  /*mean (0.0949843), correlation (0.554694)*/
  if (briefBit<vstep, rot, 5,-9, 8,10>(base)) bits |= 1 << 15;
  /*mean (0.0994152), correlation (0.550979)*/
  if (briefBit<vstep, rot, 2,7, 3,-9>(base)) bits |= 1 << 16;
  /*mean (0.10045), correlation (0.552714)*/
  if (briefBit<vstep, rot, -1,-6, -1,-1>(base)) bits |= 1 << 17;
  /*mean (0.100686), correlation (0.552594)*/
  if (briefBit<vstep, rot, 9,5, 11,-2>(base)) bits |= 1 << 18;
  /*mean (0.101091), correlation (0.532394)*/
  if (briefBit<vstep, rot, 11,-3, 12,-8>(base)) bits |= 1 << 19;
  /*mean (0.101147), correlation (0.525576)*/
  if (briefBit<vstep, rot, 3,0, 3,5>(base)) bits |= 1 << 20;
  /*mean (0.105263), correlation (0.531498)*/
  if (briefBit<vstep, rot, -1,4, 0,10>(base)) bits |= 1 << 21;
  /*mean (0.110785), correlation (0.540491)*/
  if (briefBit<vstep, rot, 3,-6, 4,5>(base)) bits |= 1 << 22;
  /*mean (0.112798), correlation (0.536582)*/
  if (briefBit<vstep, rot, -13,0, -10,5>(base)) bits |= 1 << 23;
  /*mean (0.114181), correlation (0.555793)*/
  if (briefBit<vstep, rot, 5,8, 12,11>(base)) bits |= 1 << 24;
  /*mean (0.117431), correlation (0.553763)*/
  if (briefBit<vstep, rot, 8,9, 9,-6>(base)) bits |= 1 << 25;
  /*mean (0.118522), correlation (0.553452)*/
  if (briefBit<vstep, rot, 7,-4, 8,-12>(base)) bits |= 1 << 26;
  /*mean (0.12094), correlation (0.554785)*/
  if (briefBit<vstep, rot, -10,4, -10,9>(base)) bits |= 1 << 27;
  /*mean (0.122582), correlation (0.555825)*/
  if (briefBit<vstep, rot, 7,3, 12,4>(base)) bits |= 1 << 28;
  /*mean (0.124978), correlation (0.549846)*/
  if (briefBit<vstep, rot, 9,-7, 10,-2>(base)) bits |= 1 << 29;
  /*mean (0.127002), correlation (0.537452)*/
  if (briefBit<vstep, rot, 7,0, 12,-2>(base)) bits |= 1 << 30;
  /*mean (0.127148), correlation (0.547401)*/
  if (briefBit<vstep, rot, -1,-6, 0,-11>(base)) bits |= 1 << 31;

  descriptor[7] = bits;
}

/// Non-templated version of briefDescribeRot.
///
template <int vstep, int words>
void briefDescribe(uint8_t img[][vstep], int x, int y,
    int rot, uint32_t descriptor[words]) {

  switch(rot) {
  case  0:
    briefDescribeRot<vstep,  0, words>(img, x, y, descriptor);
    return;
  case  1:
    briefDescribeRot<vstep,  1, words>(img, x, y, descriptor);
    return;
  case  2:
    briefDescribeRot<vstep,  2, words>(img, x, y, descriptor);
    return;
  case  3:
    briefDescribeRot<vstep,  3, words>(img, x, y, descriptor);
    return;
  case  4:
    briefDescribeRot<vstep,  4, words>(img, x, y, descriptor);
    return;
  case  5:
    briefDescribeRot<vstep,  5, words>(img, x, y, descriptor);
    return;
  case  6:
    briefDescribeRot<vstep,  6, words>(img, x, y, descriptor);
    return;
  case  7:
    briefDescribeRot<vstep,  7, words>(img, x, y, descriptor);
    return;
  case  8:
    briefDescribeRot<vstep,  8, words>(img, x, y, descriptor);
    return;
  case  9:
    briefDescribeRot<vstep,  9, words>(img, x, y, descriptor);
    return;
  case 10:
    briefDescribeRot<vstep, 10, words>(img, x, y, descriptor);
    return;
  case 11:
    briefDescribeRot<vstep, 11, words>(img, x, y, descriptor);
    return;
  case 12:
    briefDescribeRot<vstep, 12, words>(img, x, y, descriptor);
    return;
  case 13:
    briefDescribeRot<vstep, 13, words>(img, x, y, descriptor);
    return;
  case 14:
    briefDescribeRot<vstep, 14, words>(img, x, y, descriptor);
    return;
  case 15:
    briefDescribeRot<vstep, 15, words>(img, x, y, descriptor);
    return;
  case 16:
    briefDescribeRot<vstep, 16, words>(img, x, y, descriptor);
    return;
  case 17:
    briefDescribeRot<vstep, 17, words>(img, x, y, descriptor);
    return;
  case 18:
    briefDescribeRot<vstep, 18, words>(img, x, y, descriptor);
    return;
  case 19:
    briefDescribeRot<vstep, 19, words>(img, x, y, descriptor);
    return;
  case 20:
    briefDescribeRot<vstep, 20, words>(img, x, y, descriptor);
    return;
  case 21:
    briefDescribeRot<vstep, 21, words>(img, x, y, descriptor);
    return;
  case 22:
    briefDescribeRot<vstep, 22, words>(img, x, y, descriptor);
    return;
  case 23:
    briefDescribeRot<vstep, 23, words>(img, x, y, descriptor);
    return;
  case 24:
    briefDescribeRot<vstep, 24, words>(img, x, y, descriptor);
    return;
  case 25:
    briefDescribeRot<vstep, 25, words>(img, x, y, descriptor);
    return;
  case 26:
    briefDescribeRot<vstep, 26, words>(img, x, y, descriptor);
    return;
  case 27:
    briefDescribeRot<vstep, 27, words>(img, x, y, descriptor);
    return;
  case 28:
    briefDescribeRot<vstep, 28, words>(img, x, y, descriptor);
    return;
  case 29:
    briefDescribeRot<vstep, 29, words>(img, x, y, descriptor);
    return;
  }
}
} /* namespace pislam */

#endif /* PISLAM_BRIEF_H_ */
