#ifndef PISLAM_FAST_H_
#define PISLAM_FAST_H_

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "arm_neon.h"

#include "Util.h"
#include "Harris.h"

namespace pislam {

/// Detect FAST features and output to `out` as `0xff` for detected points and
/// `0x00` otherwise. Unless width is a multiple of 16, this method will
/// classify up to 15 extra pixels and write them past the end of the row.
/// In this case, `out[y][width]` and `out[y][width+1]` will be set to zero,
/// but values of extra classified features are undefined. 
///
/// `out` should be initialized to zeros if non-max suppression is to be used,
/// but can safely be reused without reinitializing if specified region remains
/// unchanged.
/// 
/// Additionally, FAST queries 3 pixels around the region classified. For
/// these reasons, the template parameter border should be set to at least 3.
/// If the points are to be scored using a Harris score, set to at least 4,
/// and if the points are to be described using ORB, set to at least 15.
///
/// Running time is independent of image contents and is approximately
/// 68k pixels / ms / GHz.
///
template <int vstep, int border>
void fastDetect(const int width, const int height,
    uint8_t img[][vstep], uint8_t out[][vstep], int threshold) {

  uint8x16_t vthreshold = vdupq_n_u8(threshold);

  for (int y = border; y < height - border; y += 1) {
    for (int x = border; x < width - border; x += 16) {
      uint8x16_t c = vld1q_u8(&img[y+0][x+0]);
      uint8x16_t light = vqaddq_u8(c, vthreshold);
      uint8x16_t dark = vqsubq_u8(c, vthreshold);

      uint8x16_t test = vld1q_u8(&img[y-3][x-1]);
      uint8x16_t d0 = vcgeq_u8(test, dark);
      uint8x16_t l0 = vcleq_u8(test, light);

      test = vld1q_u8(&img[y-3][x+0]);
      d0 = vbslq_u8(vdupq_n_u8(0x40u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x40u), vcleq_u8(test, light), l0);

      test = vld1q_u8(&img[y-3][x+1]);
      d0 = vbslq_u8(vdupq_n_u8(0x20u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x20u), vcleq_u8(test, light), l0);

      test = vld1q_u8(&img[y-2][x+2]);
      d0 = vbslq_u8(vdupq_n_u8(0x10u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x10u), vcleq_u8(test, light), l0);

      test = vld1q_u8(&img[y-1][x+3]);
      d0 = vbslq_u8(vdupq_n_u8(0x08u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x08u), vcleq_u8(test, light), l0);
      
      test = vld1q_u8(&img[y+0][x+3]);
      d0 = vbslq_u8(vdupq_n_u8(0x04u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x04u), vcleq_u8(test, light), l0);
      
      test = vld1q_u8(&img[y+1][x+3]);
      d0 = vbslq_u8(vdupq_n_u8(0x02u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x02u), vcleq_u8(test, light), l0);
      
      test = vld1q_u8(&img[y+2][x+2]);
      d0 = vbslq_u8(vdupq_n_u8(0x01u), vcgeq_u8(test, dark), d0);
      l0 = vbslq_u8(vdupq_n_u8(0x01u), vcleq_u8(test, light), l0);
      
      test = vld1q_u8(&img[y+3][x+1]);
      uint8x16_t d1 = vcgeq_u8(test, dark);
      uint8x16_t l1 = vcleq_u8(test, light);

      test = vld1q_u8(&img[y+3][x+0]);
      d1 = vbslq_u8(vdupq_n_u8(0x40u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x40u), vcleq_u8(test, light), l1);

      test = vld1q_u8(&img[y+3][x-1]);
      d1 = vbslq_u8(vdupq_n_u8(0x20u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x20u), vcleq_u8(test, light), l1);

      test = vld1q_u8(&img[y+2][x-2]);
      d1 = vbslq_u8(vdupq_n_u8(0x10u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x10u), vcleq_u8(test, light), l1);

      test = vld1q_u8(&img[y+1][x-3]);
      d1 = vbslq_u8(vdupq_n_u8(0x08u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x08u), vcleq_u8(test, light), l1);
      
      test = vld1q_u8(&img[y+0][x-3]);
      d1 = vbslq_u8(vdupq_n_u8(0x04u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x04u), vcleq_u8(test, light), l1);
      
      test = vld1q_u8(&img[y-1][x-3]);
      d1 = vbslq_u8(vdupq_n_u8(0x02u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x02u), vcleq_u8(test, light), l1);
      
      test = vld1q_u8(&img[y-2][x-2]);
      d1 = vbslq_u8(vdupq_n_u8(0x01u), vcgeq_u8(test, dark), d1);
      l1 = vbslq_u8(vdupq_n_u8(0x01u), vcleq_u8(test, light), l1);

      // Determine whether to test dark or light pattern.
      // 8 consecutive bits implies d0 & d1 == 0.
      uint8x16_t t0 = vtstq_u8(d0, d1);
      uint8x16_t t1 = vtstq_u8(d0, d1);

      t0 = vbslq_u8(t0, l0, d0);
      t1 = vbslq_u8(t1, l1, d1);

      uint8x16_t cntLo = vclzq_u8(t0);
      uint8x16_t testLo = t1 << (cntLo - 1);
      asm("vceq.u8  %q0, %q0, #0" : [val] "+w" (testLo));

      uint8x16_t cntHi = vclzq_u8(t1);
      uint8x16_t testHi = t0 << (cntHi - 1);
      asm("vceq.u8  %q0, %q0, #0" : [val] "+w" (testHi));

      uint8x16_t result = (cntLo & testLo) | (cntHi & testHi);
      result = vtstq_u8(result, result);

      vst1q_u8(&out[y][x], result);
    }
    // we've already written past the end of width, but we make the guarantee
    // to the caller that two zeros are present at the right edge.
    if (width % 16 != 0) {
      out[y][width  ] = 0;
      out[y][width+1] = 0;
    }
  }
}

/// Replace non-zero pixels in `out`, presumably detected points of interest,
/// with 8 bit harris score. Zero value remain zero.
///
/// Running time depends on number of non-zero pixels to be classified.
/// See Harris.h for exact details, but expect 2 ms for a 640x480 VGA image.
///
template <int vstep, int border>
void fastScoreHarris(int width, int height,
    uint8_t img[][vstep], int32_t threshold, uint8_t out[][vstep]) {

  int x, y;
  for (y = border; y < height - border; y += 1) {
    for (x = border; x < width - border; x += 1) {
      uint8_t encoded = out[y][x];
      if (!encoded) {
        continue;
      }
      out[y][x] = harrisScoreSobel<vstep>(img, x, y, threshold);
    }
  }
}

/// Extract FAST (or other) points with non-max suppression. Points are tested
/// against 8 surrounding pixels for maximality.
///
/// Optionally, `logBucketSize` and `bucketLimit` can be specified to suppress
/// non-max features within small regions of the image. For example,
/// `logBucketSize = 4` and  `bucketLimit = 5` limits the number of features
/// in a 16x16 region of the image to 5.
///
/// If `logBucketSize = 0`, region suppression is disabled and gcc is able to
/// completely optimize out the added overhead.
///
/// Running time is < ms for a 640x480 VGA image and bucket overhead is about
/// .1 ms.
/// 
template <int vstep, int border, int logBucketSize = 0, int bucketLimit = 5>
std::vector<uint32_t> fastExtract(const int width, const int height,
    uint8_t out[][vstep], std::vector<uint32_t> &results) {

  constexpr int bucketSize = 1 << logBucketSize;
  const int numBuckets = (width - 2*border - 1) / bucketSize + 1;
  uint32_t buckets[numBuckets][bucketLimit];
  int counts[numBuckets];

  typedef union {
    uint8_t *bytes;
    uint32_t *word;
  } aliased_uint32_ptr_t;

  for (int y = border; y < height - border; y += 2) {
    if ((logBucketSize != 0) && ((y-border) % bucketSize) == 0) {
      if (y == border) {
        // skip retain step on initialisation
        for (int b = 0; b < numBuckets; b += 1) {
          counts[b] = 0;
        }
      } else {
        // retain best points and reset buckets
        for (int b = 0; b < numBuckets; b += 1) {
          int count = counts[b];
          for (int i = 0; i < count; i += 1) {
            results.push_back(buckets[b][i]);
          }
          counts[b] = 0;
        }
      }
    }
    for (int x = border; x < width - border; x += 2) {
      aliased_uint32_ptr_t ptr0, ptr1, ptr2, ptr3;

      ptr1.bytes = &out[y+0][x-1];
      ptr2.bytes = &out[y+1][x-1];

      uint32_t row1 = *ptr1.word;
      uint32_t row2 = *ptr2.word;

      if (!((row1 & 0xffff00) || (row2 & 0xffff00))) {
        continue;
      }

      ptr0.bytes = &out[y-1][x-1];
      ptr3.bytes = &out[y+2][x-1];

      uint32_t row0 = *ptr0.word;
      uint32_t row3 = *ptr3.word;

      // Test a 4x4 block of pixels since that fits into 4 registers.
      // Of the four middle pixels, only one will not be suppressed.
      //
      //       b0 b1 b2 b3
      // row0  
      // row1     v0 v1
      // row2     v2 v3
      // row3
      //
      // Determine which pixel is strongest, and then check the remaining
      // values to see if pixel survives.
      uint8_t v0 = (row1 >> 8) & 0xff;
      uint8_t v1 = (row1 >> 16) & 0xff;
      uint8_t v2 = (row2 >> 8) & 0xff;
      uint8_t v3 = (row2 >> 16) & 0xff;

      uint32_t result;
      if (v0 > v1 && v0 > v2 && v0 > v3) {
        if ((v0 >= (row0 & 0xff)) && (v0 >= (row1 & 0xff)) && (v0 > (row2 & 0xff))) {
          row0 >>= 8;
          if (v0 >= (row0 & 0xff)) {
            row0 >>= 8;
            if (v0 >= (row0 & 0xff)) {
              result = encodeFast(v0, x, y);
              goto store;
            }
          }
        }
      } else if (v1 > v2 && v1 > v3) {
        row0 >>= 8;
        if (v1 >= (row0 & 0xff)) {
          row0 >>= 8;
          if (v1 >= (row0 & 0xff)) {
            row0 >>= 8; row1 >>= 24; row2 >>= 24;
            if ((v1 >= (row0 & 0xff)) && (v1 > (row1 & 0xff)) && (v1 > (row2 & 0xff))) {
              result = encodeFast(v1, x+1, y);
              goto store;
            }
          }
        }
      } else if (v2 > v3) {
        if ((v2 >= (row1 & 0xff)) && (v2 >= (row2 & 0xff)) && (v2 > (row3 & 0xff))) {
          row3 >>= 8;
          if (v2 > (row3 & 0xff)) {
            row3 >>= 8;
            if (v2 > (row3 & 0xff)) {
              result = encodeFast(v2, x, y+1);
              goto store;
            }
          }
        }
      } else {
        row3 >>= 8;
        if (v3 > (row3 & 0xff)) {
          row3 >>= 8;
          if (v3 > (row3 & 0xff)) {
            row1 >>= 24; row2 >>= 24; row3 >>= 8;
            if ((v3 >= (row1 & 0xff)) && (v3 > (row2 & 0xff)) && (v3 > (row3 & 0xff))) {
              result = encodeFast(v3, x+1, y+1);
              goto store;
            }
          }
        }
      }

      continue;

store:

      int bucket = (x-border) / bucketSize;
      int count = counts[bucket];

      if (logBucketSize == 0) {
        results.push_back(result);
      } else if (count == 0) {
        // technically case below handles count == 0, but is slightly slower.
        buckets[bucket][0] = result;
        counts[bucket] = 1;
      } else if (count < bucketLimit) {
        // forward insertion
        int i;
        for (i = count - 1; i >= 0 && result < buckets[bucket][i]; i -= 1) {
          // not an off by one error, count < bucket limit, therefore count-1+1 is valid.
          buckets[bucket][i+1] = buckets[bucket][i];
        }
        buckets[bucket][i+1] = result;
        counts[bucket] = count + 1;
      } else if (result > buckets[bucket][0]) {
        // backwards insertion if we are full but result is stronger
        int i;
        for (i = 1; i < bucketLimit && result > buckets[bucket][i]; i += 1) {
          buckets[bucket][i-1] = buckets[bucket][i];
        }
        buckets[bucket][i-1] = result;
      }
    }
  }

  if (logBucketSize != 0) {
    for (int b = 0; b < numBuckets; b += 1) {
      int count = counts[b];
      for (int i = 0; i < count; i += 1) {
        results.push_back(buckets[b][i]);
      }
    }
  }

  return results;
}

} /* namespace pislam */
#endif /* PISLAM_FAST_H_ */
