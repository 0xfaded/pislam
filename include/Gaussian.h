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

#ifndef PISLAM_GAUSSIAN_BLUR_H__
#define PISLAM_GAUSSIAN_BLUR_H__

#include "arm_neon.h"

#include <stdint.h>

#define PISLAM_ALL_D_REGS \
   "d0",  "d1",  "d2",  "d3",  "d4",  "d5",  "d6",  "d7", \
   "d8",  "d9", "d10", "d11", "d12", "d13", "d14", "d15", \
  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", \
  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"

namespace pislam {

template <int vstep>
void gaussian5x5_hstore(const int width, const int height,
    const uint8_t img[][vstep], uint8_t *hstore);

/// Convolve a single channel image with a 5x5 gaussian kernel.
///
/// Image may be of any dimension greater than 16x16, but must be padded
/// to a multiple of 16 columns and multiple of 8 rows.
/// Destination image must be the same size as input.
///
/// img and out may be same pointer, in which case blur is done in place.
/// 
/// Running time for a 640x480 image is 0.7ms on raspberry pi.
template <int vstep>
void gaussian5x5(const int width, const int height,
    uint8_t img[][vstep], uint8_t out[][vstep]) {
  // The below applies the seperable gaussian filter
  // 1/16 * 1/16 * [1 4 6 4 1] * [1 4 6 4 1].T
  //
  // It is implemented using only vrhadd (vector round halving add).
  //
  // Consider the pixel values [a b c d e].
  // The convolution with the filter at c is
  //
  // 1/16 * [1*(a+e) + 4*(b+d) + 6*e]
  //
  // This can be rewritten as
  //
  // a+e                       In the comments the below terminology is used
  // --- + c
  //  2                          - short delta: the (a+e)+c part
  // ------- + c                 - long delta : the (b+d)   part
  //    2          b+d
  // ----------- + ---
  //      2         2
  // -----------------
  //         2
  //        
  //    
  // The implementation works on 8x16 blocks (8 rows, 16 cols)
  // and uses only a single pass, however four columns from the
  // previously computed block must be stored in hstore.
  //
  // Each block is first convolved vertically, then transposed,
  // convolved horizontally, and finally untransposed.
  //
  // During the vectical convolution, the register layout is as below:
  //
  //  q4- q7: last four rows from previous block
  //  q8-q16: current eight rows
  // 
  // During the horizontal phase, the register layout is as below:
  //   q0: {col  0, col  1}
  //   q1: {col  2, col  3}
  //   q2: {col  4, col  5}
  //   q3: {col  6, col  7}
  //   q4: {col  8, col  9}
  //   q5: {col 10, col 11}
  //   q6: {col 12, col 13}
  //   q7: {col 14, col 15}
  //
  //  q10: {col -4, col -3}
  //  q11: {col -2, col -1}
  //
  // The implementation for processing multiple-of-block-size images
  // is actualy quite straight forward. Most of the LOC is actually
  // dedicated to handling odd sized images.
  //
  // A 5x5 computation requires us to look behind 2 pixels
  // and ahead two pixels. I have implemented this using unaligned
  // reads, which are costing a 20% preformance hit.
  // The alternative is to pass the entire block between adjacent
  // block computations, which I have not experimented with.
  //
  // Finally, since hstore is the only state between adjacent
  // computations, this mechanism could be exposed through the api.
  // This would allow image pyramids to be computed in blocks,
  // benefiting from caching.
  // 
  // PSS. There is no real reason for this function to be templated.
  //      I did so only for consistency and to keep the library header only.

  // if width % 16 == 1, subtract a block and compute a single column of
  // pixels at the very end using hstore.
  int vblocks = (height + 7) / 8;
  int hblocks = (width + 14) / 16;

  uint8_t *hstore = new uint8_t[vblocks * 64]();
  gaussian5x5_hstore<vstep>(width, height, img, hstore);

  int step = 2*vstep;

  int i = 0;
  for (; i < hblocks; i += 1) {
    // unaligned accesses are cheap compared to the monstrosity
    // required to handle the horizontal read ahead.
    uint8_t *in0 = &img[0][i*16+2];
    uint8_t *in1 = &img[1][i*16+2];
    uint8_t *out0 = &out[0][i*16];
    uint8_t *out1 = &out[1][i*16];

    uint8_t *hstore_ptr = hstore;

    // first two rows
    asm volatile(
      // load two rows and reflect them to mimic the
      // tail of a previous block.
      "vld1.8       {d4,d5}, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       {d6,d7}, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"

      // This reflected load will be loaded twice,
      "vld1.8       {d0,d1}, [%[in0]]\n\t"
      "vmov         q1, q3\n\t"

      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);

    bool vfix = false;
    bool vfix2 = false;
    int j = height % 8 == 1 ? 1 : 0;
    for (; j < vblocks-1; j += 1) {
      asm volatile(
        // next blocks
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d10,d11}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       {d12,d13}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d14,d15}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       {d16,d17}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d18,d19}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"

        // last two rows need to be reloaded, dont increment pointers
        "vld1.8       {d20,d21}, [%[in0]]\n\t"
        "vld1.8       {d22,d23}, [%[in1]]\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);

vlast:
      asm volatile(
        // long delta
        "vrhadd.u8     q12, q0, q4\n\t"
        "vrhadd.u8     q13, q1, q5\n\t"
        "vrhadd.u8     q14, q2, q6\n\t"
        "vrhadd.u8     q15, q3, q7\n\t"

        "vrhadd.u8     q12, q12, q2\n\t"
        "vrhadd.u8     q13, q13, q3\n\t"
        "vrhadd.u8     q14, q14, q4\n\t"
        "vrhadd.u8     q15, q15, q5\n\t"

        "vrhadd.u8     q12, q12, q2\n\t"
        "vrhadd.u8     q13, q13, q3\n\t"
        "vrhadd.u8     q14, q14, q4\n\t"
        "vrhadd.u8     q15, q15, q5\n\t"

        // short delta
        "vrhadd.u8     q0, q1, q3\n\t"
        "vrhadd.u8     q1, q2, q4\n\t"
        "vrhadd.u8     q2, q3, q5\n\t"
        "vrhadd.u8     q3, q4, q6\n\t"

        // combine
        "vrhadd.u8     q0, q0, q12\n\t"
        "vrhadd.u8     q1, q1, q13\n\t"
        "vrhadd.u8     q2, q2, q14\n\t"
        "vrhadd.u8     q3, q3, q15\n\t"

        // long delta
        "vrhadd.u8     q12, q4, q8\n\t"
        "vrhadd.u8     q13, q5, q9\n\t"
        "vrhadd.u8     q14, q6, q10\n\t"
        "vrhadd.u8     q15, q7, q11\n\t"

        "vrhadd.u8     q12, q12, q6\n\t"
        "vrhadd.u8     q13, q13, q7\n\t"
        "vrhadd.u8     q14, q14, q8\n\t"
        "vrhadd.u8     q15, q15, q9\n\t"

        "vrhadd.u8     q12, q12, q6\n\t"
        "vrhadd.u8     q13, q13, q7\n\t"
        "vrhadd.u8     q14, q14, q8\n\t"
        "vrhadd.u8     q15, q15, q9\n\t"

        // short delta
        "vrhadd.u8     q4, q5, q7\n\t"
        "vrhadd.u8     q5, q6, q8\n\t"
        "vrhadd.u8     q6, q7, q9\n\t"
        "vrhadd.u8     q7, q8, q10\n\t"

        // combine
        "vrhadd.u8     q4, q4, q12\n\t"
        "vrhadd.u8     q5, q5, q13\n\t"
        "vrhadd.u8     q6, q6, q14\n\t"
        "vrhadd.u8     q7, q7, q15\n\t"

        // load previous 4 columns
        "vld1.u8       {d20-d23}, [%[hstore]]\n\t"
        
        // transpose result for horizontal pass
        "vswp          d1, d8\n\t"
        "vswp          d3, d10\n\t"
        "vswp          d5, d12\n\t"
        "vswp          d7, d14\n\t"

        "vtrn.32       q0, q2\n\t"
        "vtrn.32       q1, q3\n\t"
        "vtrn.32       q4, q6\n\t"
        "vtrn.32       q5, q7\n\t"

        "vtrn.16       q0, q1\n\t"
        "vtrn.16       q2, q3\n\t"
        "vtrn.16       q4, q5\n\t"
        "vtrn.16       q6, q7\n\t"

        "vuzp.8        d0, d1\n\t"
        "vuzp.8        d2, d3\n\t"
        "vuzp.8        d4, d5\n\t"
        "vuzp.8        d6, d7\n\t"
        "vuzp.8        d8, d9\n\t"
        "vuzp.8        d10, d11\n\t"
        "vuzp.8        d12, d13\n\t"
        "vuzp.8        d14, d15\n\t"

        // q6-q7 become the new previous left rows
        "vst1.u8       {d12-d15}, [%[hstore]]!\n\t"

        : [hstore] "+r"(hstore_ptr) :: PISLAM_ALL_D_REGS);

      // this monstrosity has no place in my inner loop
      if (i == hblocks - 1) {
        goto hblock_fix;
      }
hlast:

      asm volatile(
        // long delta
        "vrhadd.u8     q12, q10, q0\n\t"
        "vrhadd.u8     q13, q11, q1\n\t"
        "vrhadd.u8     q14, q0, q2\n\t"
        "vrhadd.u8     q15, q1, q3\n\t"

        "vrhadd.u8     q12, q11\n\t"
        "vrhadd.u8     q13, q0\n\t"
        "vrhadd.u8     q14, q1\n\t"
        "vrhadd.u8     q15, q2\n\t"

        "vrhadd.u8     q12, q11\n\t"
        "vrhadd.u8     q13, q0\n\t"
        "vrhadd.u8     q14, q1\n\t"
        "vrhadd.u8     q15, q2\n\t"

        // short delta
        "vrhadd.u8     d20, d4, d6\n\t"
        "vrhadd.u8     d21, d21, d23\n\t"
        "vrhadd.u8     q11, q11, q0\n\t"
        "vrhadd.u8     q0, q0, q1\n\t"
        "vrhadd.u8     q1, q1, q2\n\t"

        // combine
        "vrhadd.u8     d24, d21\n\t"
        "vrhadd.u8     d25, d22\n\t"
        "vrhadd.u8     d26, d23\n\t"
        "vrhadd.u8     d27, d0\n\t"
        "vrhadd.u8     d28, d1\n\t"
        "vrhadd.u8     d29, d2\n\t"
        "vrhadd.u8     d30, d3\n\t"
        "vrhadd.u8     d31, d20\n\t"

        // long delta
        "vrhadd.u8     q0, q2, q4\n\t"
        "vrhadd.u8     q1, q3, q5\n\t"
        "vrhadd.u8     q10, q4, q6\n\t"
        "vrhadd.u8     q11, q5, q7\n\t"

        "vrhadd.u8     q0, q3\n\t"
        "vrhadd.u8     q1, q4\n\t"
        "vrhadd.u8     q10, q5\n\t"
        "vrhadd.u8     q11, q6\n\t"

        "vrhadd.u8     q0, q3\n\t"
        "vrhadd.u8     q1, q4\n\t"
        "vrhadd.u8     q10, q5\n\t"
        "vrhadd.u8     q11, q6\n\t"

        // short delta
        "vrhadd.u8     d4, d12, d14\n\t"
        "vrhadd.u8     d5, d5, d7\n\t"

        "vrhadd.u8     q7, q5, q6\n\t"
        "vrhadd.u8     q6, q4, q5\n\t"
        "vrhadd.u8     q5, q3, q4\n\t"

        // combine
        "vrhadd.u8     d0, d0, d5\n\t"
        "vrhadd.u8     d7, d23, d4\n\t"
        "vrhadd.u8     d1, d1, d10\n\t"
        "vrhadd.u8     d2, d2, d11\n\t"
        "vrhadd.u8     d3, d3, d12\n\t"
        "vrhadd.u8     d4, d20, d13\n\t"
        "vrhadd.u8     d5, d21, d14\n\t"
        "vrhadd.u8     d6, d22, d15\n\t"

        // restore previous two rows of image
        "vld1.8       {d20,d21}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d22,d23}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"

        // untranspose. Just to make everything even more confusing,
        // the top is now in q12-q15, and bottom in q0-q3.
        "vzip.8       d0, d1\n\t"
        "vzip.8       d2, d3\n\t"
        "vzip.8       d4, d5\n\t"
        "vzip.8       d6, d7\n\t"
        "vzip.8       d24, d25\n\t"
        "vzip.8       d26, d27\n\t"
        "vzip.8       d28, d29\n\t"
        "vzip.8       d30, d31\n\t"

        "vtrn.16      q12, q13\n\t"
        "vtrn.16      q14, q15\n\t"
        "vtrn.16      q0, q1\n\t"
        "vtrn.16      q2, q3\n\t"

        "vtrn.32      q12, q14\n\t"
        "vtrn.32      q13, q15\n\t"
        "vtrn.32      q0, q2\n\t"
        "vtrn.32      q1, q3\n\t"

        "vswp         d25, d0\n\t"
        "vswp         d27, d2\n\t"
        "vswp         d29, d4\n\t"
        "vswp         d31, d6\n\t"

        // store finished block
        "vst1.8       {d24,d25}, [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       {d26,d27}, [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       {d28,d29}, [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       {d30,d31}, [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       {d0,d1}, [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       {d2,d3}, [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       {d4,d5}, [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       {d6,d7}, [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"

        // shuffle registers for next loop iteration.
        "vmov         q0, q8\n\t"
        "vmov         q1, q9\n\t"
        "vmov         q2, q10\n\t"
        "vmov         q3, q11\n\t"

        : [out0] "+r"(out0), [out1] "+r"(out1),
          [in0] "+r"(in0), [in1] "+r"(in1),
          [hstore] "+r"(hstore_ptr)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);

      continue;

hblock_fix:
#define PISLAM_GAUSSIAN_REFLECT_COLS(c0, c1, c3, c4) \
        asm volatile(\
          "vmov         " #c0 ", " #c4 "\n\t" \
          "vmov         " #c1 ", " #c3 "\n\t" \
          ::: #c0, #c1)

      uint32_t remainder = width & 0xf;
      switch (remainder) {
        case 0:
          PISLAM_GAUSSIAN_REFLECT_COLS(d15, d14, d12, d11);
          goto hlast;
        case 1:
          // the diablocial case where the reflection straddles the border
          asm volatile(\
            "vmov         d15, d13\n\t"
            ::: "d15");
          goto hlast;
        case 2:
          PISLAM_GAUSSIAN_REFLECT_COLS(d1, d0, d22, d21);
          goto hlast;
        case 3:
          PISLAM_GAUSSIAN_REFLECT_COLS(d2, d1, d23, d22);
          goto hlast;
        case 4:
          PISLAM_GAUSSIAN_REFLECT_COLS(d3, d2, d0, d23);
          goto hlast;
        case 5:
          PISLAM_GAUSSIAN_REFLECT_COLS(d4, d3, d1, d0);
          goto hlast;                                 
        case 6:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d5, d4, d2, d1);
          goto hlast;                                 
        case 7:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d6, d5, d3, d2);
          goto hlast;                                 
        case 8:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d7, d6, d4, d3);
          goto hlast;
        case 9:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d8, d7, d5, d4);
          goto hlast;
        case 10:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d9, d8, d6, d5);
          goto hlast;
        case 11:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d10, d9, d7, d6);
          goto hlast;
        case 12:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d11, d10, d8, d7);
          goto hlast;
        case 13:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d12, d11, d9, d8);
          goto hlast;
        case 14:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d13, d12, d10, d9);
          goto hlast;
        case 15:                                  
          PISLAM_GAUSSIAN_REFLECT_COLS(d14, d13, d11, d10);
          goto hlast;
      }
    }

    if (vfix) {
      continue;
    }
    vfix = true;

    // handle odd number of rows by reflecting the last two valid rows.
    switch(height & 7) {
    case 0:
      // if we were given a multiple of 8 image,
      // we need to be careful of reading past the
      // very last two bytes of the image.
        asm volatile(
          "vld1.8       {d8,d9}, [%[in0]]\n\t"
          "add          %[in0], %[step]\n\t"
          "vld1.8       {d10,d11}, [%[in1]]\n\t"
          "add          %[in1], %[step]\n\t"
          "vld1.8       {d12,d13}, [%[in0]]\n\t"
          "add          %[in0], %[step]\n\t"
          "vld1.8       {d14,d15}, [%[in1]]\n\t"
          "add          %[in1], %[step]\n\t"
          "vld1.8       {d16,d17}, [%[in0]]\n\t"
          "vmov         q10, q8\n\t"
          "vmov         q11, q7\n\t"
          : [in0] "+r"(in0), [in1] "+r"(in1)
          : [step] "r"(step)
          : PISLAM_ALL_D_REGS);

      // if we are not on our last block, or we require
      // the second hfix, load full row.
      if (i != hblocks - 1 || (width % 16 == 1)) {
        asm volatile(
          "vld1.8       {d18,d19}, [%[in1]]\n\t"
          : [in0] "+r"(in0), [in1] "+r"(in1)
          : [step] "r"(step)
          : PISLAM_ALL_D_REGS);
      } else {
        asm volatile(
          "vld1.8       d18, [%[in1]]!\n\t"
          "vld1.32      d19[0], [%[in1]]!\n\t"
          "vld1.16      d19[2], [%[in1]]!\n\t"
          : [in0] "+r"(in0), [in1] "+r"(in1)
          : [step] "r"(step)
          : PISLAM_ALL_D_REGS);
      }
      goto vlast;

    case 1:
      // If someone passed a (rows % 8) == 1, they were mean.
      // Both sides of the block boundary need to be fixed.
      if (vfix2) {
        asm volatile(
          "vmov         q4, q0\n\t"
          ::: PISLAM_ALL_D_REGS);
      } else {
        asm volatile(
          "vld1.8       {d8,d9}, [%[in0]]\n\t"
          "add          %[in0], %[step]\n\t"
          "vld1.8       {d10,d11}, [%[in1]]\n\t"
          "add          %[in1], %[step]\n\t"
          "vld1.8       {d12,d13}, [%[in0]]\n\t"
          "add          %[in0], %[step]\n\t"
          "vld1.8       {d14,d15}, [%[in1]]\n\t"
          "add          %[in1], %[step]\n\t"
          "vld1.8       {d16,d17}, [%[in0]]\n\t"
          "add          %[in0], %[step]\n\t"
          "vld1.8       {d18,d19}, [%[in1]]\n\t"
          "vld1.8       {d20,d21}, [%[in0]]\n\t"
          "vmov         q11, q9\n\t"
          : [in0] "+r"(in0), [in1] "+r"(in1)
          : [step] "r"(step)
          : PISLAM_ALL_D_REGS);

        vfix = false;
        vfix2 = true;
      }
      goto vlast;

    case 2:
      asm volatile(
        "vmov         q4, q2\n\t"
        "vmov         q5, q1\n\t"
        ::: PISLAM_ALL_D_REGS);
      goto vlast;

    case 3:
      asm volatile(
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "vmov         q5, q3\n\t"
        "vmov         q6, q2\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
      goto vlast;

    case 4:
      asm volatile(
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "vld1.8       {d10,d11}, [%[in1]]\n\t"
        "vmov         q6, q4\n\t"
        "vmov         q7, q3\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
      goto vlast;

    case 5:
      asm volatile(
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d10,d11}, [%[in1]]\n\t"
        "vld1.8       {d12,d13}, [%[in0]]\n\t"
        "vmov         q7, q5\n\t"
        "vmov         q8, q4\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
      goto vlast;

    case 6:
      asm volatile(
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d10,d11}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       {d12,d13}, [%[in0]]\n\t"
        "vld1.8       {d14,d15}, [%[in1]]\n\t"
        "vmov         q8, q6\n\t"
        "vmov         q9, q5\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
      goto vlast;

    case 7:
      asm volatile(
        "vld1.8       {d8,d9}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d10,d11}, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       {d12,d13}, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       {d14,d15}, [%[in1]]\n\t"
        "vld1.8       {d16,d17}, [%[in0]]\n\t"
        "vmov         q9, q7\n\t"
        "vmov         q10, q6\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
      goto vlast;
    }
  }

  // for the split boundary hfix, use the remaining values in hstore
  // to compute the final column of the image.
  if (width % 16 == 1) {
    uint8_t *hstore_ptr = hstore;
    uint8_t *out0 = &out[0][hblocks*16];
    uint8_t *out1 = &out[1][hblocks*16];

    for (int j = 0; j < vblocks; j += 1) {
      asm volatile(
        // load previous four columns from hstore
        "vld1.8      {d8-d11}, [%[hstore]]!\n\t"

        // final layout is
        //  d8: n-2
        //  d9: n-1
        // d10: n
        // d11: n-1

        // long delta
        "vrhadd.u8    d12, d8, d8\n\t"

        // short delta
        "vrhadd.u8    d13, d9, d9\n\t"

        // long delta
        "vrhadd.u8    d12, d10\n\t"
        "vrhadd.u8    d12, d10\n\t"

        // combine
        "vrhadd.u8    d0, d12, d13\n\t"

        // assume storing byte by byte is faster than transposing
        "vst1.8       d0[0], [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       d0[1], [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       d0[2], [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       d0[3], [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       d0[4], [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       d0[5], [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"
        "vst1.8       d0[6], [%[out0]]\n\t"
        "add          %[out0], %[step]\n\t"
        "vst1.8       d0[7], [%[out1]]\n\t"
        "add          %[out1], %[step]\n\t"

        : [out0] "+r"(out0), [out1] "+r"(out1), [hstore] "+r"(hstore_ptr)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
    }
  }

  delete[] hstore;

  return;
}

/// Preform a vertical pass on the two leftmost pixels of image,
/// reflect them and store in hstore.
///
/// This method is currently intended for internal use,
/// but exposing the hstore mechanism would allow for the
/// convolution to be computed in blocks, offering pipelining
/// benefits.
template<int vstep>
void gaussian5x5_hstore(const int width, const int height,
    const uint8_t img[][vstep], uint8_t *hstore) {

  int vblocks = (height + 7) / 8;
  int step = 2*vstep;

  const uint8_t *in0 = &img[0][0];
  const uint8_t *in1 = &img[1][0];

  // first two rows
  asm volatile(
    // load two rows and reflect them to mimic the
    // tail of a previous block.
    "vld1.8       d2, [%[in0]]\n\t"
    "add          %[in0], %[step]\n\t"
    "vld1.8       d3, [%[in1]]\n\t"
    "add          %[in1], %[step]\n\t"

    // This reflected load will be loaded twice,
    "vld1.8       d0, [%[in0]]\n\t"
    "vmov         d1, d3\n\t"

    : [in0] "+r"(in0), [in1] "+r"(in1)
    : [step] "r"(step)
    : PISLAM_ALL_D_REGS);

  bool vfix = false;
  bool vfix2 = false;
  int j = height % 8 == 1 ? 1 : 0;
  for (; j < vblocks-1; j += 1) {

    asm volatile(
      // next blocks
      "vld1.8       d4, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d6, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d7, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d8, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d9, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d10, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d11, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);

vlast:
    asm volatile(
      // long delta
      "vrhadd.u8     d12, d0, d4\n\t"
      "vrhadd.u8     d13, d1, d5\n\t"
      "vrhadd.u8     d14, d2, d6\n\t"
      "vrhadd.u8     d15, d3, d7\n\t"

      "vrhadd.u8     d12, d12, d2\n\t"
      "vrhadd.u8     d13, d13, d3\n\t"
      "vrhadd.u8     d14, d14, d4\n\t"
      "vrhadd.u8     d15, d15, d5\n\t"

      "vrhadd.u8     d12, d12, d2\n\t"
      "vrhadd.u8     d13, d13, d3\n\t"
      "vrhadd.u8     d14, d14, d4\n\t"
      "vrhadd.u8     d15, d15, d5\n\t"

      // short delta
      "vrhadd.u8     d0, d1, d3\n\t"
      "vrhadd.u8     d1, d2, d4\n\t"
      "vrhadd.u8     d2, d3, d5\n\t"
      "vrhadd.u8     d3, d4, d6\n\t"

      // combine
      "vrhadd.u8     d0, d0, d12\n\t"
      "vrhadd.u8     d1, d1, d13\n\t"
      "vrhadd.u8     d2, d2, d14\n\t"
      "vrhadd.u8     d3, d3, d15\n\t"

      // long delta
      "vrhadd.u8     d12, d4, d8\n\t"
      "vrhadd.u8     d13, d5, d9\n\t"
      "vrhadd.u8     d14, d6, d10\n\t"
      "vrhadd.u8     d15, d7, d11\n\t"

      "vrhadd.u8     d12, d12, d6\n\t"
      "vrhadd.u8     d13, d13, d7\n\t"
      "vrhadd.u8     d14, d14, d8\n\t"
      "vrhadd.u8     d15, d15, d9\n\t"

      "vrhadd.u8     d12, d12, d6\n\t"
      "vrhadd.u8     d13, d13, d7\n\t"
      "vrhadd.u8     d14, d14, d8\n\t"
      "vrhadd.u8     d15, d15, d9\n\t"

      // short delta
      "vrhadd.u8     d4, d5, d7\n\t"
      "vrhadd.u8     d5, d6, d8\n\t"
      "vrhadd.u8     d6, d7, d9\n\t"
      "vrhadd.u8     d7, d8, d10\n\t"

      // combine
      "vrhadd.u8     d4, d4, d12\n\t"
      "vrhadd.u8     d5, d5, d13\n\t"
      "vrhadd.u8     d6, d6, d14\n\t"
      "vrhadd.u8     d7, d7, d15\n\t"

      // transpose
      "vtrn.32       d0, d4\n\t"
      "vtrn.32       d1, d5\n\t"
      "vtrn.32       d2, d6\n\t"
      "vtrn.32       d3, d7\n\t"

      "vtrn.16       d0, d2\n\t"
      "vtrn.16       d1, d3\n\t"

      "vtrn.8        d0, d1\n\t"
      "vtrn.8        d2, d3\n\t"

      // reflect
      "vswp          d0, d2\n\t"
      "vmov          d3, d1\n\t"

      "vst1.u8       {d0-d3}, [%[hstore]]!\n\t"

      // shuffle registers for next loop iteration.
      "vmov         d0, d8\n\t"
      "vmov         d1, d9\n\t"
      "vmov         d2, d10\n\t"
      "vmov         d3, d11\n\t"

      : [hstore] "+r"(hstore) :: PISLAM_ALL_D_REGS);
  }

  if (vfix) {
    return;
  }
  vfix = true;

  // handle odd number of rows by reflecting the last two valid rows.
  switch(height & 7) {
  case 0:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d6, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d7, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d8, [%[in0]]\n\t"
      "vld1.8       d9, [%[in1]]\n\t"
      "vmov         d10, d8\n\t"
      "vmov         d11, d7\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;

  case 1:
    // If someone passed a (rows % 8) == 1, they were mean.
    // Both sides of the block boundary need to be fixed.
    if (vfix2) {
      asm volatile(
        "vmov         d4, d0\n\t"
        ::: PISLAM_ALL_D_REGS);
    } else {
      asm volatile(
        "vld1.8       d4, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       d5, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       d6, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       d7, [%[in1]]\n\t"
        "add          %[in1], %[step]\n\t"
        "vld1.8       d8, [%[in0]]\n\t"
        "add          %[in0], %[step]\n\t"
        "vld1.8       d9, [%[in1]]\n\t"
        "vld1.8       d10, [%[in0]]\n\t"
        "vmov         d11, d9\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);

      vfix = false;
      vfix2 = true;
    }
    goto vlast;

  case 2:
    asm volatile(
      "vmov         d4, d2\n\t"
      "vmov         d5, d1\n\t"
      ::: PISLAM_ALL_D_REGS);
    goto vlast;

  case 3:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "vmov         d5, d3\n\t"
      "vmov         d6, d2\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;

  case 4:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "vmov         d6, d4\n\t"
      "vmov         d7, d3\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;

  case 5:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "vld1.8       d6, [%[in0]]\n\t"
      "vmov         d7, d5\n\t"
      "vmov         d8, d4\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;

  case 6:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d6, [%[in0]]\n\t"
      "vld1.8       d7, [%[in1]]\n\t"
      "vmov         d8, d6\n\t"
      "vmov         d9, d5\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;

  case 7:
    asm volatile(
      "vld1.8       d4, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d5, [%[in1]]\n\t"
      "add          %[in1], %[step]\n\t"
      "vld1.8       d6, [%[in0]]\n\t"
      "add          %[in0], %[step]\n\t"
      "vld1.8       d7, [%[in1]]\n\t"
      "vld1.8       d8, [%[in0]]\n\t"
      "vmov         d9, d7\n\t"
      "vmov         d10, d6\n\t"
      : [in0] "+r"(in0), [in1] "+r"(in1)
      : [step] "r"(step)
      : PISLAM_ALL_D_REGS);
    goto vlast;
  }
  return;
}

} /* namespace */

#endif /* PISLAM_GAUSSIAN_BLUR_H__ */
