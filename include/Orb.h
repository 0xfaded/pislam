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

#ifndef PISLAM_ORB_H_
#define PISLAM_ORB_H_

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <vector>
#include <cassert>

#include "arm_neon.h"

#include "Brief.h"
#include "Util.h"

namespace pislam {

#define PISLAM_CENTROID_SUM_ROW_NO_MASK(n, dx, weights) \
    yy += 1; \
    top = vld1_u8(&base[-n][dx]); \
    bot = vld1_u8(&base[n][dx]); \
    \
    ymoment16 = vmlal_u8(ymoment16, bot, yy); \
    ymoment16 = vmlsl_u8(ymoment16, top, yy); \
    xmoment16 = vmlal_u8(xmoment16, top, weights); \
    xmoment16 = vmlal_u8(xmoment16, bot, weights)

#define PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST(n, dx, weights) \
    yy += 1; \
    top = vld1_u8(&base[-n][dx]); \
    bot = vld1_u8(&base[n][dx]); \
    top = vset_lane_u8(0, top, 7); \
    bot = vset_lane_u8(0, bot, 7); \
    \
    ymoment16 = vmlal_u8(ymoment16, bot, yy); \
    ymoment16 = vmlsl_u8(ymoment16, top, yy); \
    xmoment16 = vmlal_u8(xmoment16, top, weights); \
    xmoment16 = vmlal_u8(xmoment16, bot, weights)


#define PISLAM_CENTROID_SUM_ROW_USING_MASK(n, dx, weights) \
    yy += 1; \
    top = vld1_u8(&base[-n][dx]); \
    top = vand_u8(top, mask); \
    bot = vld1_u8(&base[n][dx]); \
    bot = vand_u8(bot, mask); \
    \
    ymoment16 = vmlal_u8(ymoment16, bot, yy); \
    ymoment16 = vmlsl_u8(ymoment16, top, yy); \
    xmoment16 = vmlal_u8(xmoment16, top, weights); \
    xmoment16 = vmlal_u8(xmoment16, bot, weights)

// gcc is able to optimize out the superfluous yy -= 1; y += 1
#define PISLAM_CENTROID_SUM_ROW_MASKED(n, dx, maskn, weights) \
    yy += 1; \
    mask = vcle_u8(yy, maskn); \
    yy -= 1; \
    PISLAM_CENTROID_SUM_ROW_USING_MASK(n, dx, weights);

template<int vstep>
std::vector<int32_t> orbCentroids(uint8_t img[][vstep],
    const std::vector<uint32_t> &points) {

  // Circle looks like this, reflected about y = 0
  //
  //       -15 -13 -11 -9  -7  -5  -3  -1   1   3   5   7   9  11  13  15
  //         -14 -12 -10 -8  -6  -4  -2   0   2   4   6   8  10  12  14
  //
  //       |               |               |               |                |
  // -15   |               |    # # # # # #|# # # # #      |                |
  // -14   |               |# # # # # # # #|# # # # # # #  |                |
  // -13   |            # #|# # # # # # # #|# # # # # # # #|#               |
  // -12   |          # # #|# # # # # # # #|# # # # # # # #|# #             |
  // -11   |        # # # #|# # # # # # # #|# # # # # # # #|# # #           |
  // -10   |      # # # # #|# # # # # # # #|# # # # # # # #|# # # #         |
  //  -9   |    # # # # # #|# # # # # # # #|# # # # # # # #|# # # # #       |
  //  -8   |    # # # # # #|# # # # # # # #|# # # # # # # #|# # # # #       |
  //  -7   |  # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # #     |
  //  -6   |  # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # #     |
  //  -5   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //  -4   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //  -3   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //  -2   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //  -1   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //   0   |# # # # # # # #|# # # # # # # #|# # # # # # # #|# # # # # # #   |
  //       |               |               |               |                |
  //
  // It is divided into 4 columns, each of which is summed individually.
  // Incidentally read bytes are masked out. In the case of one extra
  // rightmost byte, setlane is used.
  //
  // round up to nearest 8
  std::vector<int32_t> centroids;
  centroids.resize((2*points.size() + 7) & (~0x7));

  // the trick is to create masks which are valid at row x by comparing
  // to the row number.
  uint8x8_t leftMask  = { 5, 7, 9, 10, 11, 12, 13, 13 };
  uint8x8_t rightMask = { 13, 12, 11, 10, 9, 7, 5, 0 };
  uint8x8_t topBotLeftMask  = { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
  uint8x8_t topBotRightMask = { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0 };

  uint8x8_t leftWeights        = { 15, 14, 13, 12, 11, 10,  9,  8 };
  uint8x8_t centerLeftWeights  = {  7,  6,  5,  4,  3,  2,  1,  0 };
  uint8x8_t centerRightWeights = {  1,  2,  3,  4,  5,  6,  7,  8 };
  uint8x8_t rightWeights       = {  9, 10, 11, 12, 13, 14, 15,  0 };

  // values
  uint8x8_t mid, top, bot;

  // referenced by macros.
  // yy must be kept because vmlal instruction do not exist for u8 immediate
  uint8x8_t yy;
  uint8x8_t mask;

  // accumulators
  // To avoid overflow in outer xmoments, use unsigned values.
  uint16x8_t xmoment16, ymoment16;
  uint32x4_t xmoment32, xmomentLeft32; int32x4_t ymoment32;
  int32_t xmomenti, ymomenti;

  int out = 0;
  for (uint32_t point : points) {
    int x = decodeFastX(point);
    int y = decodeFastY(point);

    uint8_t (* const base)[vstep] = (uint8_t (*)[vstep])(&img[y][x]);

    // center-right column, center columns are unmasked
    // middle
    mid = vld1_u8(&base[0][1]);
    xmoment16 = vmull_u8(mid, centerRightWeights);

    // row +-1
    yy = vdup_n_u8(1);
    top = vld1_u8(&base[-1][1]);
    bot = vld1_u8(&base[ 1][1]);

    ymoment16 = vsubl_u8(bot, top);
    xmoment16 = vmlal_u8(xmoment16, top, centerRightWeights);
    xmoment16 = vmlal_u8(xmoment16, bot, centerRightWeights);

    PISLAM_CENTROID_SUM_ROW_NO_MASK( 2, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 3, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 4, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 5, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 6, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 7, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 8, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 9, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(10, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(11, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(12, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(13, 1, centerRightWeights);
    PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST(14, 1, centerRightWeights);

    mask = topBotRightMask;
    PISLAM_CENTROID_SUM_ROW_USING_MASK(15, 1, centerRightWeights);

    // Each column sum fits into a 16-bit register ... just.
    // Store result into 32bit registers and reset.
    xmoment32 = vpaddlq_u16(xmoment16);

    // ymoment is signed. xmoment is computed in left and right half
    // and therefore is unsigned.
    ymoment32 = vpaddlq_s16(vreinterpretq_s16_u16(ymoment16));

    // right column
    mid = vld1_u8(&base[0][9]);
    mid = vset_lane_u8(0, mid, 7);
    xmoment16 = vmull_u8(mid, rightWeights);

    // row +-1
    yy = vdup_n_u8(1);
    top = vld1_u8(&base[-1][9]);
    bot = vld1_u8(&base[ 1][9]);
    top = vset_lane_u8(0, top, 7);
    bot = vset_lane_u8(0, bot, 7);

    // don't know how, but the multiply by yy=1 using mull and mlsl are faster
    // than using sub to achieve the same thing. Maybe set lane is utilising the
    // adding hardware?
    ymoment16 = vmull_u8(bot, yy);
    ymoment16 = vmlsl_u8(ymoment16, top, yy);
    xmoment16 = vmlal_u8(xmoment16, top, rightWeights);
    xmoment16 = vmlal_u8(xmoment16, bot, rightWeights);

    PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST( 2, 9, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST( 3, 9, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST( 4, 9, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASK_RIGHTMOST( 5, 9, rightWeights);

    PISLAM_CENTROID_SUM_ROW_MASKED( 6, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 7, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 8, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 9, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(10, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(11, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(12, 9, rightMask, rightWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(13, 9, rightMask, rightWeights);
    
    xmoment32 = vpadalq_u16(xmoment32, xmoment16);
    ymoment32 = vpadalq_s16(ymoment32, vreinterpretq_s16_u16(ymoment16));

    // left column
    mid = vld1_u8(&base[0][-15]);
    xmoment16 = vmull_u8(mid, leftWeights);

    // row +-1
    yy = vdup_n_u8(1);
    top = vld1_u8(&base[-1][-15]);
    bot = vld1_u8(&base[ 1][-15]);

    ymoment16 = vsubl_u8(bot, top);
    xmoment16 = vmlal_u8(xmoment16, top, leftWeights);
    xmoment16 = vmlal_u8(xmoment16, bot, leftWeights);

    PISLAM_CENTROID_SUM_ROW_NO_MASK(2, -15, leftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(3, -15, leftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(4, -15, leftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(5, -15, leftWeights);

    PISLAM_CENTROID_SUM_ROW_MASKED( 6, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 7, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 8, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED( 9, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(10, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(11, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(12, -15, leftMask, leftWeights);
    PISLAM_CENTROID_SUM_ROW_MASKED(13, -15, leftMask, leftWeights);

    // Left columns subtract from xmoment. Don't use vneg, which
    // may overflow a 16-bit register (which is why we are working
    // with unsigned values to begin with).
    xmomentLeft32 = vpaddlq_u16(xmoment16);
    ymoment32 = vpadalq_s16(ymoment32, vreinterpretq_s16_u16(ymoment16));

    // center-left column, center columns are unmasked
    mid = vld1_u8(&base[0][-7]);
    xmoment16 = vmull_u8(mid, centerLeftWeights);

    // row +-1
    yy = vdup_n_u8(1);
    top = vld1_u8(&base[-1][-7]);
    bot = vld1_u8(&base[ 1][-7]);

    ymoment16 = vsubl_u8(bot, top);
    xmoment16 = vmlal_u8(xmoment16, top, centerLeftWeights);
    xmoment16 = vmlal_u8(xmoment16, bot, centerLeftWeights);

    PISLAM_CENTROID_SUM_ROW_NO_MASK( 2, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 3, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 4, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 5, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 6, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 7, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 8, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK( 9, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(10, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(11, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(12, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(13, -7, centerLeftWeights);
    PISLAM_CENTROID_SUM_ROW_NO_MASK(14, -7, centerLeftWeights);

    mask = topBotLeftMask;
    PISLAM_CENTROID_SUM_ROW_USING_MASK(15, -7, centerLeftWeights);

    xmomentLeft32 = vpadalq_u16(xmomentLeft32, xmoment16);
    ymoment32 = vpadalq_s16(ymoment32, vreinterpretq_s16_u16(ymoment16));
    xmoment32 = vsubq_u32(xmoment32, xmomentLeft32);

    // faster to do the addition arm side (0.03 ms / 1000 points)
    xmomenti = int32_t(xmoment32[0]) + int32_t(xmoment32[1]) +
      int32_t(xmoment32[2]) + int32_t(xmoment32[3]);

    ymomenti= ymoment32[0] + ymoment32[1] + ymoment32[2] + ymoment32[3];

    centroids[out  ] = xmomenti;
    centroids[out+4] = ymomenti;

    out += 1;
    if (out % 4 == 0) {
      out += 4;
    }
  }
  centroids.resize((centroids.size() + 7) & ~(0x7));
  return centroids;
}

std::vector<uint8_t> atan2(const std::vector<int32_t> &xys) {
  // returning angles as uint8_t instead of uint32_t saved
  // 0.2 ms / frame with 1229 points.
  std::vector<uint8_t> angles;
  for (auto it = xys.begin(); it < xys.end(); it += 8) {
    int32x4_t xmoment32 = vld1q_s32(&it[0]);
    int32x4_t ymoment32 = vld1q_s32(&it[4]);

    // atan approximation from 
    // https://math.stackexchange.com/questions/1098487/atan2-faster-approximation
    float32x4_t xmomentf = vcvtq_f32_s32(xmoment32);
    float32x4_t ymomentf = vcvtq_f32_s32(ymoment32);

    xmomentf = vabsq_f32(xmomentf);
    ymomentf = vabsq_f32(ymomentf);

    float32x4_t zmaxf = vmaxq_f32(xmomentf, ymomentf);
    float32x4_t zminf = vminq_f32(xmomentf, ymomentf);

    float32x4_t zf = vrecpeq_f32(zmaxf);

    zf = vmulq_f32(zminf, zf);

    // atan z = z * (pi/4 + 0.273 * (z-1)) for z = [0..1).
    // But if we scale the constants by 60/pi, we get
    // atan z = [0..15) which is useful below.
    // Further, converting back to integers always rounds down.
    // Multiply constants by 256 to shift the decimal down.
    float32x4_t c0 = vdupq_n_f32(256*14.999998);
#if 0
    // Average error is 0.1313 degrees. Misclassifies 1/133
    float32x4_t c1 = vdupq_n_f32(256*5.35);
    float32x4_t anglef = zf * (c0 - c1 * (zf - 1));
#else
    // Average error is 0.054 degrees. Misclassifies 1/273
    float32x4_t c1 = vdupq_n_f32(256*4.723436);
    float32x4_t c2 = vdupq_n_f32(256*1.266240);
    float32x4_t anglef = zf * (c0 - (zf - 1) * (c1 + c2 * zf));
#endif

    int32x4_t angle32 = vcvtq_s32_f32(anglef);

    for (int i = 0; i < 4; i += 1) {
      int32_t x = it[i];
      int32_t y = it[i+4];
      int32_t angle = angle32[i];

      if (abs(x) > abs(y)) {
        if ((x^y) < 0) { // signs differ
          angle = -angle;
        }
        if (x < 0) {
          angle += 256*60;
        } else if (angle < 0) {
          angle += 256*120;
        }
      } else {
        if ((x^y) >= 0) { // signs same
          angle = -angle;
        }
        if (y >= 0) {
          angle = angle + 256*30;
        } else {
          angle = angle + 256*90;
        }
      }
      // scale back into [0..30]
      angle >>= 10;
      if (!(0 <= angle && angle < 30)) {
        // guard against possible NaN nonsense
        angle = 0;
      }
      angles.push_back(angle);
    }
  }

  return angles;
}

/// Compute ORB descriptions from keypoints. Set words to the number
/// of 32bit words the brief description should output, up to
/// 8 for a 256 bit descriptor. Descriptors are appended to the back of
/// `descriptors`.
///
/// Running time is 250 features / ms / GHz
///
template <int vstep, int words>
void orbCompute(uint8_t img[][vstep], const std::vector<uint32_t> &points,
    std::vector<uint32_t> &descriptors) {

  std::vector<int32_t> centroids = orbCentroids<vstep>(img, points);
  std::vector<uint8_t> angles = atan2(centroids);

  // The briefDescribe function is 1026 instructions long = 4104 bytes.
  // Unfortunately we get killed on cache performance, and it's actually
  // faster to iterate over the points 30 times calling only the
  // brief descriptor for a particular orientation. This also beats
  // sorting first.
  //
  // Experimentally it was found that reducing iterations by evaluating
  // pairs reduced execution time by .5 ms for 1000 features.
  // 3s, and 4s each slightly decreased execution time, but pairs were
  // chosen since the speed up is probably not worth the cache loss.
#define PISLAM_ORB_COMPUTE_DESCRIBE(rot) \
  for (size_t i = 0; i < points.size(); i += 1) { \
    if (rot*2 <= angles[i] && angles[i] < (rot + 1)*2) { \
      uint32_t point = points[i]; \
      int x = decodeFastX(point); \
      int y = decodeFastY(point); \
      briefDescribe<vstep, words>(img, x, y, angles[i], &out[i*words]); \
    } \
  }

  descriptors.resize(descriptors.size() + points.size()*words);
  auto out = descriptors.end() - (points.size()*words);

  PISLAM_ORB_COMPUTE_DESCRIBE(0);
  PISLAM_ORB_COMPUTE_DESCRIBE(1);
  PISLAM_ORB_COMPUTE_DESCRIBE(2);
  PISLAM_ORB_COMPUTE_DESCRIBE(3);
  PISLAM_ORB_COMPUTE_DESCRIBE(4);
  PISLAM_ORB_COMPUTE_DESCRIBE(5);
  PISLAM_ORB_COMPUTE_DESCRIBE(6);
  PISLAM_ORB_COMPUTE_DESCRIBE(7);
  PISLAM_ORB_COMPUTE_DESCRIBE(8);
  PISLAM_ORB_COMPUTE_DESCRIBE(9);
  PISLAM_ORB_COMPUTE_DESCRIBE(10);
  PISLAM_ORB_COMPUTE_DESCRIBE(11);
  PISLAM_ORB_COMPUTE_DESCRIBE(12);
  PISLAM_ORB_COMPUTE_DESCRIBE(13);
  PISLAM_ORB_COMPUTE_DESCRIBE(14);
}
} /* namespace pislam */

#endif /* PISLAM_ORB_H_ */
