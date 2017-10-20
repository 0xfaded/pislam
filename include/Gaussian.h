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

/// Convolve a single channel image with a 5x5 gaussian kernel.
///
/// Image may be of any dimension greater than 16x16, but must be padded
/// to a multiple of 8 bytes.
/// 
/// Running time is essentially the round trip time between main memory.
/// On the raspberry pi, a 640x480 image requires 1ms, or approximately
/// 4 clock cycles per pixel. A smaller image that fits in L1 cache (64x128)
/// requires approximately 2 clock cycles per pixel.
///
/// The algorithm may be further optimized in the future to better use
/// L1 cache for larger images.
template<int vstep>
void gaussian5x5(const int width, const int height, uint8_t img[][vstep]) {
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
  // The implementation works on 8x16 blocks (8 rows, 16 cols).
  // During the first pass, the register layout is as follows:
  //
  //  q4- q7: last four rows from previous block
  //  q8-q16: current block
  // 
  // During the first pass, the register layout is as follows:
  //   q8: {col  0, col  1}
  //   q9: {col  4, col  5}
  //  q10: {col  2, col  3}
  //  q11: {col  6, col  7}
  //  q12: {col  8, col  9}
  //  q13: {col 12, col 13}
  //  q14: {col 10, col 11}
  //  q15: {col 14, col 15}
  //
  // The transpose from rows (pass 1) to columns (pass 2) is
  // performed using interleaved writes and loads (vst4),
  // and a final unzip operation.
  //
  // The final result is transposed and has to be un-transposed again
  // in the usual way.
  //
  // These transpose operations require rows and columns to be mapped
  // to specific registers, which is why gcc intrinsics were not used.
  //
  // The implementation for processing multiple-of-block-size images
  // is actualy quite straight forward. Most of the LOC is actually
  // dedicated to handling odd sized images.
  //
  // Finally, L1 cache is very poorly utilized. Experiments show
  // that processing 128x64 pixel images, allowing both the image
  // and buffer to fit in L1 cache on a raspberry pi, is 1.85 times
  // faster than processing an 640x480 image. This can be fixed
  // by processing the image in blocks, but for now it is fast enough.
  // 
  // PSS. There is no real reason for this function to be templated.
  //      I did so only for consistency and to keep the library header only.

  int vblocks = (height + 7) / 8;
  int hblocks = (width + 15) / 16;

  uint8_t *hstore = new uint8_t[hblocks * 64]();
  int step = 2*vstep;

  for (int i = 0; i < hblocks; i += 1) {
    // unaligned accesses are cheap compared to the monstrosity
    // required to handle the horizontal read ahead.
    uint8_t *in0 = &img[0][i*16+2];
    uint8_t *in1 = &img[1][i*16+2];
    uint8_t *out0 = &img[0][i*16];
    uint8_t *out1 = &img[1][i*16];

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

    // Check for case where reflected rows straddle block boundary.
    int j = (height & 7) == 1 ? 1 : 0;
    bool vfix = false;
    bool vfix2 = false;
    bool hfix2 = false;
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
          ::: #c3, #c4)

      uint32_t remainder = width & 0xf;
      switch (remainder) {
        case 0:
          PISLAM_GAUSSIAN_REFLECT_COLS(d15, d14, d12, d11);
          goto hlast;
        case 1:
          // the diablocial case where the reflection straddles the border
          if (hfix2) {
            asm volatile(\
              "vmov         d0, d20\n\t"
              ::: "d0");
          } else {
            asm volatile(\
              "vmov         d15, d13\n\t"
              ::: "d13");
            hfix2 = !hfix2;
            i -= 1;
          }
          goto hlast;
        case 2:
          PISLAM_GAUSSIAN_REFLECT_COLS(d1, d0, d22, d21);
          goto hlast;
        case 3:
          PISLAM_GAUSSIAN_REFLECT_COLS(d2, d1, d23, d22);
          goto hlast;
        case 4:
          PISLAM_GAUSSIAN_REFLECT_COLS(d3, d2, d23, d0);
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
        "vld1.8       {d18,d19}, [%[in1]]\n\t"
        "vmov         q10, q8\n\t"
        "vmov         q11, q7\n\t"
        : [in0] "+r"(in0), [in1] "+r"(in1)
        : [step] "r"(step)
        : PISLAM_ALL_D_REGS);
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

        // go straight back to vlast for other side
        vfix2 = true;
        goto vlast;
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

  delete[] hstore;

  return;

#if 0
        if (vfix) {
          continue;
        }

    // handle odd number of rows by reflecting the last two valid rows.
    switch(height & 7) {
    case 0:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d12,d13}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d14,d15}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d16,d17}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d18,d19}, [%1]\n\t"
        "vmov         q10, q8\n\t"
        "vmov         q11, q7\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
            break;

    case 1:
      // If someone passed a (rows % 8) == 1, they were mean.
      // Both sides of the block boundary need to be fixed.
      if (vfix2) {
        asm volatile(
          "vmov         q4, q0\n\t"
          ::: PISLAM_ALL_D_REGS);
      } else {
        asm volatile(
          "vld1.8       {d8,d9}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vld1.8       {d10,d11}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vld1.8       {d12,d13}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vld1.8       {d14,d15}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vld1.8       {d16,d17}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vld1.8       {d18,d19}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vld1.8       {d20,d21}, [%0]\n\t"
          "vmov         q11, q9\n\t"
          : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
          : "r"(step)
          : PISLAM_ALL_D_REGS);

        // go straight back to vlast for other side
        vfix2 = true;
        goto vlast;
      }
      break;

    case 2:
      asm volatile(
        "vmov         q4, q2\n\t"
        "vmov         q5, q1\n\t"
        ::: PISLAM_ALL_D_REGS);
      break;

    case 3:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "vmov         q5, q3\n\t"
        "vmov         q6, q2\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
      break;

    case 4:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d10,d11}, [%1]\n\t"
        "vmov         q6, q4\n\t"
        "vmov         q7, q3\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
      break;

    case 5:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d12,d13}, [%0]\n\t"
        "vmov         q7, q5\n\t"
        "vmov         q8, q4\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
      break;

    case 6:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d12,d13}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d14,d15}, [%1]\n\t"
        "vmov         q8, q6\n\t"
        "vmov         q9, q5\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
      break;

    case 7:
      asm volatile(
        "vld1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d12,d13}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vld1.8       {d14,d15}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vld1.8       {d16,d17}, [%0]\n\t"
        "vmov         q9, q7\n\t"
        "vmov         q10, q6\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
      break;
    }

    vfix = true;
    goto vlast;
  }

  // horizontal pass
  for (int i = 0; i < vblocks; i += 1) {
    uint8_t *ptr1 = &img[i*8][0];
    uint8_t *ptr2 = &img[i*8+1][0];
    uint8_t *ptr3 = &buffer[i*128];
    bool hfix = false;

    asm volatile(
      "vld4.8       {d16,d18,d20,d22}, [%2]!\n\t"
      "vld4.8       {d17,d19,d21,d23}, [%2]!\n\t"

      "vuzp.32      q8, q9\n\t"
      "vuzp.32      q10, q11\n\t"

      // reflect into q15
      "vmov         d31, d17\n\t"
      "vmov         d30, d20\n\t"

      // compute first short delta
      "vrhadd.u8     d29, d17, d31\n\t"

      : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
      : "r"(step)
      : PISLAM_ALL_D_REGS);

    int j = 0;

    goto hload_skip;

    for (; j < width - 6; j += 16) {
      asm volatile(
        // load next block of columns, compute missing row 7
        "vld4.8       {d16,d18,d20,d22}, [%2]!\n\t"
        "vld4.8       {d17,d19,d21,d23}, [%2]!\n\t"

        "vuzp.32      q8, q9\n\t"
        "vuzp.32      q10, q11\n\t"

        // long delta
        "vrhadd.u8     q12, q8, q13\n\t"

        // short delta
        "vrhadd.u8     q14, q8, q15\n\t"

        // long delta
        "vrhadd.u8     q12, q12, q15\n\t"
        "vrhadd.u8     q12, q12, q15\n\t"

        // combine
        "vrhadd.u8     d14, d24, d15\n\t"
        "vrhadd.u8     d15, d25, d28\n\t"
        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);

hstore:

      asm volatile(
        // transpose
        "vswp         d1, d2\n\t"
        "vswp         d5, d6\n\t"
        "vswp         d9, d10\n\t"
        "vswp         d13, d14\n\t"

        "vtrn.32      q0, q4\n\t"
        "vtrn.32      q1, q5\n\t"
        "vtrn.32      q2, q6\n\t"
        "vtrn.32      q3, q7\n\t"

        "vtrn.16      q0, q2\n\t"
        "vtrn.16      q1, q3\n\t"
        "vtrn.16      q4, q6\n\t"
        "vtrn.16      q5, q7\n\t"

        "vtrn.8       q0, q1\n\t"
        "vtrn.8       q2, q3\n\t"
        "vtrn.8       q4, q5\n\t"
        "vtrn.8       q6, q7\n\t"

        // store
        "vst1.8       {d0,d1}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d2,d3}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d4,d5}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d6,d7}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d12,d13}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d14,d15}, [%1]\n\t"
        "add          %1, %3\n\t"

        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);

      ptr1 -= vstep*8-16;
      ptr2 -= vstep*8-16;

hload_skip:

      asm volatile(
        // compute first three rows

        // long delta
        "vrhadd.u8     q0, q10, q15\n\t"
        "vrhadd.u8     q2, q8, q9\n\t"
        "vrhadd.u8     q4, q10, q11\n\t"

        // short delta
        "vrhadd.u8     q1, q8, q10\n\t"

        // long delta
        "vrhadd.u8     q0, q0, q8\n\t"
        "vrhadd.u8     q2, q2, q10\n\t"
        "vrhadd.u8     q4, q4, q9\n\t"

        // short delta
        "vrhadd.u8     q3, q9, q10\n\t"

        // long delta
        "vrhadd.u8     q0, q0, q8\n\t"
        "vrhadd.u8     q2, q2, q10\n\t"
        "vrhadd.u8     q4, q4, q9\n\t"

        // short delta, store in d10 which is accessable to the next block
        "vrhadd.u8     q10, q9, q11\n\t"

        // combine
        "vrhadd.u8     d0, d0, d29\n\t"
        "vrhadd.u8     d1, d1, d2\n\t"
        "vrhadd.u8     d4, d4, d3\n\t"
        "vrhadd.u8     d5, d5, d6\n\t"
        "vrhadd.u8     d8, d8, d7\n\t"
        "vrhadd.u8     d9, d9, d20\n\t"

        "vld4.8        {d24,d26,d28,d30}, [%2]!\n\t"
        "vld4.8        {d25,d27,d29,d31}, [%2]!\n\t"

        "vuzp.32       q12, q13\n\t"
        "vuzp.32       q14, q15\n\t"

        // long delta
        "vrhadd.u8     q6, q9, q12\n\t"
        "vrhadd.u8     q1, q11, q14\n\t"
        "vrhadd.u8     q3, q12, q13\n\t"
        "vrhadd.u8     q5, q14, q15\n\t"

        "vrhadd.u8     q6, q6, q11\n\t"
        "vrhadd.u8     q1, q1, q12\n\t"
        "vrhadd.u8     q3, q3, q14\n\t"
        "vrhadd.u8     q5, q5, q13\n\t"

        "vrhadd.u8     q6, q6, q11\n\t"
        "vrhadd.u8     q1, q1, q12\n\t"
        "vrhadd.u8     q3, q3, q14\n\t"
        "vrhadd.u8     q5, q5, q13\n\t"

        // short delta
        // last row short delta gets passed to next loop iteration in q7
        "vrhadd.u8     q7, q13, q15\n\t"

        // compute remaining short deltas, q10 is holding long delta for q6
        "vrhadd.u8     q8, q11, q12\n\t"
        "vrhadd.u8     q9, q12, q14\n\t"
        "vrhadd.u8     q11, q13, q14\n\t"

        // combine
        "vrhadd.u8     d12, d12, d21\n\t"
        "vrhadd.u8     d13, d13, d16\n\t"
        "vrhadd.u8     d2, d2, d17\n\t"
        "vrhadd.u8     d3, d3, d18\n\t"
        "vrhadd.u8     d6, d6, d19\n\t"
        "vrhadd.u8     d7, d7, d22\n\t"
        "vrhadd.u8     d10, d10, d23\n\t"
        "vrhadd.u8     d11, d11, d14\n\t"

        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);

      ptr3 += (vblocks-1)*128;
    }

    // handle remainder columns
    uint32_t remainder = width & 0xf;
    if (remainder && (remainder <= 6)) {
      // This is the easy case, fire a second iteration with
      // columns of first block reflected appropriately.
      if (!hfix) {
        hfix = true;
        asm volatile(
          // load block
          "vld4.8       {d16,d18,d20,d22}, [%2]!\n\t"
          "vld4.8       {d17,d19,d21,d23}, [%2]!\n\t"

          "vuzp.32      q8, q9\n\t"
          "vuzp.32      q10, q11\n\t"
          : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
          : "r"(step)
          : PISLAM_ALL_D_REGS);

        if (remainder <= 4) {
          if (remainder <= 2) {
            if (remainder == 1) {
              asm volatile(
                "vmov         d17, d31\n\t"
                "vmov         d20, d30\n\t"
              ::: PISLAM_ALL_D_REGS);
            } else { // remainder == 2
              asm volatile(
                "vmov         d20, d16\n\t"
                "vmov         d21, d31\n\t"
              ::: PISLAM_ALL_D_REGS);
            }
          } else {
            if (remainder == 3) {
              asm volatile(
                "vmov         d21, d17\n\t"
                "vmov         d18, d16\n\t"
              ::: PISLAM_ALL_D_REGS);
            } else { // remainder == 4
              asm volatile(
                "vmov         d18, d20\n\t"
                "vmov         d19, d17\n\t"
              ::: PISLAM_ALL_D_REGS);
            }
          }
        } else {
          if (remainder == 5) {
            asm volatile(
              "vmov         d19, d21\n\t"
              "vmov         d22, d20\n\t"
            ::: PISLAM_ALL_D_REGS);
          } else { // remainder == 6
            asm volatile(
              "vmov         d22, d18\n\t"
              "vmov         d23, d21\n\t"
            ::: PISLAM_ALL_D_REGS);
          }
        }

        // compute row 7 as usual
        asm volatile (
          "vrhadd.u8     q12, q8, q13\n\t"
          "vrhadd.u8     q14, q8, q15\n\t"

          "vrhadd.u8     q12, q12, q15\n\t"
          "vrhadd.u8     q12, q12, q15\n\t"

          "vrhadd.u8     d14, d24, d15\n\t"
          "vrhadd.u8     d15, d25, d28\n\t"
          ::: PISLAM_ALL_D_REGS);

        goto hstore;
      } else { // hfix = false

hstore8:
        // only need to store first 8 columns
        asm volatile(
          "vtrn.32      q0, q4\n\t"
          "vtrn.32      q2, q6\n\t"

          "vtrn.16      q0, q2\n\t"
          "vtrn.16      q4, q6\n\t"

          "vtrn.8       d0, d1\n\t"
          "vtrn.8       d4, d5\n\t"
          "vtrn.8       d8, d9\n\t"
          "vtrn.8       d12, d13\n\t"

          // store
          "vst1.8       {d0}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vst1.8       {d1}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vst1.8       {d4}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vst1.8       {d5}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vst1.8       {d8}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vst1.8       {d9}, [%1]\n\t"
          "add          %1, %3\n\t"
          "vst1.8       {d12}, [%0]\n\t"
          "add          %0, %3\n\t"
          "vst1.8       {d13}, [%1]\n\t"
          "add          %1, %3\n\t"

          : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
          : "r"(step)
          : PISLAM_ALL_D_REGS);
      }
    } else {
      // Less fun cases. 
      // Directly compute reflected columns, then store appropriately.
      // The below macro takes the four last column registers,
      // and two output registers.
#define PISLAM_BLUR_COMPUTE_LAST(c0, c1, c2, c3, out0, out1, tmp) \
      asm volatile(\
        "vrhadd.u8    " #tmp  ", " #c2   ", " #c0  "\n\t"\
        "vrhadd.u8    " #out0 ", " #c3   ", " #c1  "\n\t"\
        "vrhadd.u8    " #out1 ", " #tmp  ", " #c0  "\n\t"\
        "vrhadd.u8    " #out0 ", " #out0 ", " #c1  "\n\t"\
        "vrhadd.u8    " #out1 ", " #out1 ", " #c1  "\n\t"\
        "vrhadd.u8    " #out0 ", " #out0 ", " #c1  "\n\t"\
        "vrhadd.u8    " #out0 ", " #out0 ", " #tmp "\n\t"\
        ::: #c0, #out0, #out1, #tmp)


      if (remainder <= 11) {
        if (remainder == 0) { // even case
          PISLAM_BLUR_COMPUTE_LAST(d31, d30, d27, d26, d14, d15, d16);
        } else {
          // these are the worst cases, since we need to reload lost values
          ptr3 -= vblocks*128;
          asm volatile(
            "vld4.8       {d16,d18,d20,d22}, [%2]!\n\t"
            "vld4.8       {d17,d19,d21,d23}, [%2]\n\t"

            "vuzp.32      q8, q9\n\t"
            "vuzp.32      q10, q11\n\t"
            : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
            : "r"(step)
            : PISLAM_ALL_D_REGS);
          
          if (remainder <= 9) {
            if (remainder == 7) {
              PISLAM_BLUR_COMPUTE_LAST(d22, d19, d18, d21, d9, d12, d16);
              goto hstore8;
            } else if (remainder == 8) {
              PISLAM_BLUR_COMPUTE_LAST(d23, d22, d19, d18, d12, d13, d16);
              goto hstore8;
            } else { // remainder == 9
              PISLAM_BLUR_COMPUTE_LAST(d24, d23, d22, d19, d13, d2, d16);
            }
          } else {
            if (remainder == 10) {
              PISLAM_BLUR_COMPUTE_LAST(d25, d24, d23, d22, d2, d3, d16);
            } else { // remainder == 11
              PISLAM_BLUR_COMPUTE_LAST(d28, d25, d24, d23, d3, d6, d16);
            }
          }
        }
      } else {
        if (remainder <= 13) {
          if (remainder == 12) { // even case
            PISLAM_BLUR_COMPUTE_LAST(d29, d28, d25, d24, d6, d7, d16);
          } else { // remainder == 13
            PISLAM_BLUR_COMPUTE_LAST(d26, d29, d28, d25, d7, d10, d16);
          }
        } else {
          if (remainder == 14) {
            PISLAM_BLUR_COMPUTE_LAST(d27, d26, d29, d28, d10, d11, d16);
          } else { // remainder == 15
            PISLAM_BLUR_COMPUTE_LAST(d30, d27, d26, d29, d11, d14, d16);
          }
        }
      }
      asm volatile(
        // transpose
        "vswp         d1, d2\n\t"
        "vswp         d5, d6\n\t"
        "vswp         d9, d10\n\t"
        "vswp         d13, d14\n\t"

        "vtrn.32      q0, q4\n\t"
        "vtrn.32      q1, q5\n\t"
        "vtrn.32      q2, q6\n\t"
        "vtrn.32      q3, q7\n\t"

        "vtrn.16      q0, q2\n\t"
        "vtrn.16      q1, q3\n\t"
        "vtrn.16      q4, q6\n\t"
        "vtrn.16      q5, q7\n\t"

        "vtrn.8       q0, q1\n\t"
        "vtrn.8       q2, q3\n\t"
        "vtrn.8       q4, q5\n\t"
        "vtrn.8       q6, q7\n\t"

        // store
        "vst1.8       {d0,d1}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d2,d3}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d4,d5}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d6,d7}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d8,d9}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d10,d11}, [%1]\n\t"
        "add          %1, %3\n\t"
        "vst1.8       {d12,d13}, [%0]\n\t"
        "add          %0, %3\n\t"
        "vst1.8       {d14,d15}, [%1]\n\t"
        "add          %1, %3\n\t"

        : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
        : "r"(step)
        : PISLAM_ALL_D_REGS);
    }
  }

  delete[] buffer;
#endif
}

} /* namespace */

#endif /* PISLAM_GAUSSIAN_BLUR_H__ */
