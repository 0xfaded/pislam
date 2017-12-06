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

#ifndef PISLAM_HARRIS_H_
#define PISLAM_HARRIS_H_

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <vector>

#include "arm_neon.h"

#include "Util.h"

namespace pislam {

// requires that (Ixx+Iyy)**2 < 2**32
static inline uint8_t harrisEval(uint32x2_t Ixx, uint32x2_t Iyy, int32x2_t Ixy,
    int32_t threshold) {

  // use k = 1/16 = 0.0625
  uint32x2_t trace32 = vadd_u32(Ixx, Iyy);
  trace32 = vmul_u32(trace32, trace32);
  trace32 = vshr_n_u32(trace32, 4);

  // determinant of autocorrelation is positive
  uint32x2_t det32 = vmul_u32(Ixx, Iyy);

  // Ixy32 may be negative, need to do signed multiplication
  det32 = vreinterpret_u32_s32(
      vmls_s32(vreinterpret_s32_u32(det32), Ixy, Ixy));

  // det < 2**30, trace32 < 2**28, so we are safe to convert to signed
  int32x2_t score32 = vsub_s32(
      vreinterpret_s32_u32(det32),
      vreinterpret_s32_u32(trace32));

  int32_t score = score32[0];
  if (threshold < score) {
    // IEEE single precision is encoded as
    // [sign (1 bit)][exponent (8 bits)][fraction (23 bits)]
    // We want to produce a "quarter precision" float by taking 5 exponent
    // bits and 3 fraction bits.
    float32x2_t scoref = vcvt_f32_s32(score32);
    uint32_t logscore = vreinterpret_u32_f32(scoref)[0];
    return (logscore >> 20) & 0xff;
  }

  return 0;
}

/// Compute Harris score using 3x3 Sobel operator. Due to NEON register
/// width, score is computed over a 6x6 region instead of a 7x7 region
/// as in the opencv implementation.
///
/// Result is an 8bit "quarter precision float", with 5 exponent bits
/// and 3 fraction bits. Higher score means a stronger corner response.
///
/// Running time is 3000 pixels / ms / ghz, or 100ms for a 640x480 VGA image.
///
template<int vstep>
uint8_t harrisScoreSobel(uint8_t img[][vstep], int x, int y,
    int32_t threshold) {

  // rows and columns
  uint8x8_t row0, row1, row2, row3, row4, row5, row6, row7;

  // derivatives
  int8x8_t dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
  int8x8_t dy0, dy1, dy2, dy3, dy4, dy5;

  // tmps needed for dx and dy sobel operator.
  int8x8_t tmpDy1, tmpDy2;
  uint8x8_t tmpRow;

  // accumulators
  int16x8_t xx, yy, xy;

  // as explained below, xx32 and yy32 are unsigned to prevent overflows.
  uint32x4_t xx32, yy32; int32x4_t xy32;
  uint32x2_t Ixx, Iyy, xx32l, xx32h, yy32l, yy32h; int32x2_t Ixy, xy32l, xy32h;

  // load pixels
  uint8_t (* const base)[vstep] = (uint8_t (*)[vstep])(&img[y][x-3]);
  row0 = vld1_u8(&base[-3][0]);
  row1 = vld1_u8(&base[-2][0]);
  row2 = vld1_u8(&base[-1][0]);
  row3 = vld1_u8(&base[ 0][0]);
  row4 = vld1_u8(&base[ 1][0]);
  row5 = vld1_u8(&base[ 2][0]);
  row6 = vld1_u8(&base[ 3][0]);
  row7 = vld1_u8(&base[ 4][0]);

  // Compute dy and dx. For dy, deltas can be directly computed as
  // dy_n = row_n-1 - row_n+1
  // The sobel operator is applied by shifting dy_n and adding
  // 0.25 * (dy << 8) + 0.5 * dy + 0.25 * (dy >> 8)
  // 
  // The above would result in two incomplete computations,
  // the left most and right most bytes. We would prefer the
  // incomplete computations were the two rightmost bits,
  // so instead we use
  // 0.25 * (dy << 16) + 0.5 * (dy << 8) + 0.25 * dy
#define PISLAM_HARRIS_DY_SOBEL(n0, n1, n2) \
  tmpDy1 = vreinterpret_s8_u8(vhsub_u8(row##n2, row##n0)); \
  tmpDy2 = vreinterpret_s8_u64(vshr_n_u64(vreinterpret_u64_s8(tmpDy1), 16)); \
  dy##n0 = vreinterpret_s8_u64(vshr_n_u64(vreinterpret_u64_s8(tmpDy1), 8)); \
  tmpDy1 = vhadd_s8(tmpDy1, tmpDy2); \
  dy##n0 = vhadd_s8(dy##n0, tmpDy1)

  PISLAM_HARRIS_DY_SOBEL(0, 1, 2);
  PISLAM_HARRIS_DY_SOBEL(1, 2, 3);
  PISLAM_HARRIS_DY_SOBEL(2, 3, 4);
  PISLAM_HARRIS_DY_SOBEL(3, 4, 5);
  PISLAM_HARRIS_DY_SOBEL(4, 5, 6);
  PISLAM_HARRIS_DY_SOBEL(5, 6, 7);

  // Compute dx in the opposite manner. Shift to compute deltas, sobel operator
  // can be directly applied by summing together rows.
#define PISLAM_HARRIS_DX_SOBEL_1(n) \
  tmpRow = vreinterpret_u8_u64(vshr_n_u64(vreinterpret_u64_u8(row##n), 16)); \
  dx##n = vreinterpret_s8_u8(vhsub_u8(tmpRow, row##n));

  // Apply sobel operator by summing 0.25 * row0 + 0.5 * row1 + 0.25 * row2
#define PISLAM_HARRIS_DX_SOBEL_2(n0, n1, n2) \
  dx##n0 = vhadd_s8(dx##n0, dx##n2); \
  dx##n0 = vhadd_s8(dx##n0, dx##n1); \

  PISLAM_HARRIS_DX_SOBEL_1(0);
  PISLAM_HARRIS_DX_SOBEL_1(1);
  PISLAM_HARRIS_DX_SOBEL_1(2);
  PISLAM_HARRIS_DX_SOBEL_1(3);
  PISLAM_HARRIS_DX_SOBEL_1(4);
  PISLAM_HARRIS_DX_SOBEL_1(5);
  PISLAM_HARRIS_DX_SOBEL_1(6);
  PISLAM_HARRIS_DX_SOBEL_1(7);

  PISLAM_HARRIS_DX_SOBEL_2(0, 1, 2);
  PISLAM_HARRIS_DX_SOBEL_2(1, 2, 3);
  PISLAM_HARRIS_DX_SOBEL_2(2, 3, 4);
  PISLAM_HARRIS_DX_SOBEL_2(3, 4, 5);
  PISLAM_HARRIS_DX_SOBEL_2(4, 5, 6);
  PISLAM_HARRIS_DX_SOBEL_2(5, 6, 7);

  // Now compute Ixx, Ixy and Iyy.
  // Two int8_t*int8_t muls fit into an int16_t without overflow.
  xx = vmull_s8(dx0, dx0);
  yy = vmull_s8(dy0, dy0);
  xy = vmull_s8(dx0, dy0);

  xx = vmlal_s8(xx, dx1, dx1);
  yy = vmlal_s8(yy, dy1, dy1);
  xy = vmlal_s8(xy, dx1, dy1);

  // xx and yy are guaranteed to be positive. They also suffer from an
  // overflowing edge case when both lanes are -128, as
  // -128*-128+-128*-128 = 0x8000.
  // The work around is to treat xx and yy as unsigned.
  //
  // xy does not have this edge case due to a simple proof. The premise is
  // that the largest value of xy would be created by the below image patch.
  //
  // 00 ff    dx = 0x80   dy = 0x80
  // ff 00    dx = 0x7f   dy = 0x7f
  // 00 ff    dx = 0x80   dy = 0x80
  //
  // Since a two adjacent -128 (0x80) values cannot occur, we are safe.
  xx32 = vpaddlq_u16(vreinterpretq_u16_s16(xx));
  yy32 = vpaddlq_u16(vreinterpretq_u16_s16(yy));
  xy32 = vpaddlq_s16(xy);

  xx = vmull_s8(dx2, dx2);
  yy = vmull_s8(dy2, dy2);
  xy = vmull_s8(dx2, dy2);

  xx = vmlal_s8(xx, dx3, dx3);
  yy = vmlal_s8(yy, dy3, dy3);
  xy = vmlal_s8(xy, dx3, dy3);
  
  xx32 = vpadalq_u16(xx32, vreinterpretq_u16_s16(xx));
  yy32 = vpadalq_u16(yy32, vreinterpretq_u16_s16(yy));
  xy32 = vpadalq_s16(xy32, xy);
  
  xx = vmull_s8(dx4, dx4);
  yy = vmull_s8(dy4, dy4);
  xy = vmull_s8(dx4, dy4);

  xx = vmlal_s8(xx, dx5, dx5);
  yy = vmlal_s8(yy, dy5, dy5);
  xy = vmlal_s8(xy, dx5, dy5);
  
  xx32 = vpadalq_u16(xx32, vreinterpretq_u16_s16(xx));
  yy32 = vpadalq_u16(yy32, vreinterpretq_u16_s16(yy));
  xy32 = vpadalq_s16(xy32, xy);

  // All vectors contain an extra two image bytes since we started
  // with 8x8 but could only compute the inner 6x6 differences.
  // The above computations are arranged such that the rubbish bytes
  // are placed in the last two bytes, and are now accumulated
  // in the highest word of xx32, yy32, xy32.
  //
  // Instead of masking these words out, which would require an
  // extra instruction each, we simply do not padd the high register.
  // Because harrisEval uses only the lower 32bits of Ixx, Iyy, Ixy,
  // the rubbish bytes remain uncounted.
  
  // padd the low register
  xx32l = vreinterpret_u32_u64(vpaddl_u32(vget_low_u32(xx32)));
  yy32l = vreinterpret_u32_u64(vpaddl_u32(vget_low_u32(yy32)));
  xy32l = vreinterpret_s32_s64(vpaddl_s32(vget_low_s32(xy32)));

  // add in the upper register
  xx32h = vget_high_u32(xx32);
  yy32h = vget_high_u32(yy32);
  xy32h = vget_high_s32(xy32);

  // add in the high register
  Ixx = vadd_u32(xx32l, xx32h);
  Iyy = vadd_u32(yy32l, yy32h);
  Ixy = vadd_s32(xy32l, xy32h);

  // Scoring requires (Ixx + Iyy)**2 < 2**32. Shifting off 4 bits
  // assures this.
  Ixx = vshr_n_u32(Ixx, 4);
  Iyy = vshr_n_u32(Iyy, 4);
  Ixy = vshr_n_s32(Ixy, 4);

  return harrisEval(Ixx, Iyy, Ixy, threshold);
}

} /* namespace pislam */
#endif /* PISLAM_HARRIS_H_ */
