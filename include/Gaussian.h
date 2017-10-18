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

template<int vstep>
void gaussian5x5(const int width, const int height, uint8_t img[][vstep]) {

  int vblocks = (height + 7) / 8;
  int hblocks = (width + 15) / 16;

  int step = 2*vstep;

  uint8_t *buffer = new uint8_t[vblocks*hblocks*128];

  // vertical pass
  uint8_t *ptr3 = buffer;
  for (int i = 0; i < hblocks; i += 1) {
    uint8_t *ptr1 = &img[0][i*16];
    uint8_t *ptr2 = &img[1][i*16];

    bool vfix = false;

    // if height % 8 == 1, two fixes are required.
    bool vfix2 = false;

  // first two rows
  asm(
    // first block with reflected top two rows
    "vld1.8       {d4,d5}, [%0]\n\t"
    "add          %0, %3\n\t"
    "vld1.8       {d6,d7}, [%1]\n\t"
    "add          %1, %3\n\t"

    // this reflected load will be loaded twice,
    // but not the end of the world
    "vld1.8       {d0,d1}, [%0]\n\t"
    "vmov         q1, q3\n\t"

    : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
    : "r"(step)
    : PISLAM_ALL_D_REGS);

    // Check for case where reflected rows straddle block boundary.
    int j = (height & 7) == 1 ? 1 : 0;
    for (; j < vblocks-1; j += 1) {

  asm(
    // next two blocks
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
    "add          %0, %3\n\t"
    "vld1.8       {d22,d23}, [%1]\n\t"
    "add          %1, %3\n\t"
    : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
    : "r"(step)
    : PISLAM_ALL_D_REGS);

last_block:

  asm(
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

    "vrhadd.u8     q0, q1, q3\n\t"
    "vrhadd.u8     q1, q2, q4\n\t"
    "vrhadd.u8     q2, q3, q5\n\t"
    "vrhadd.u8     q3, q4, q6\n\t"

    "vrhadd.u8     q0, q0, q12\n\t"
    "vrhadd.u8     q1, q1, q13\n\t"
    "vrhadd.u8     q2, q2, q14\n\t"
    "vrhadd.u8     q3, q3, q15\n\t"

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

    "vrhadd.u8     q4, q5, q7\n\t"
    "vrhadd.u8     q5, q6, q8\n\t"
    "vrhadd.u8     q6, q7, q9\n\t"
    "vrhadd.u8     q7, q8, q10\n\t"

    "vrhadd.u8     q4, q4, q12\n\t"
    "vrhadd.u8     q5, q5, q13\n\t"
    "vrhadd.u8     q6, q6, q14\n\t"
    "vrhadd.u8     q7, q7, q15\n\t"

    "vst4.32      {d0,d2,d4,d6}, [%2]!\n\t"
    "vst4.32      {d8,d10,d12,d14}, [%2]!\n\t"
    "vst4.32      {d1,d3,d5,d7}, [%2]!\n\t"

    // shuffle registers for next loop iteration.
    // pipeline is stalled on stores regardless
    "vmov         q0, q8\n\t"
    "vmov         q1, q9\n\t"

    "vst4.32      {d9,d11,d13,d15}, [%2]!\n\t"

    "vmov         q2, q10\n\t"
    "vmov         q3, q11\n\t"

    : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
    : "r"(step)
    : PISLAM_ALL_D_REGS);

    }
    if (vfix) {
      continue;
    }

    switch(height & 7) {
      case 0:
  asm(
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
          asm(
            "vmov         q4, q0\n\t"
            ::: PISLAM_ALL_D_REGS);
        } else {
          asm(
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

          // go straight back to last_block for other side
          vfix2 = true;
          goto last_block;
        }
        break;

      case 2:
  asm(
    "vmov         q4, q2\n\t"
    "vmov         q5, q1\n\t"
    ::: PISLAM_ALL_D_REGS);
        break;

      case 3:
  asm(
    "vld1.8       {d8,d9}, [%0]\n\t"
    "vmov         q5, q3\n\t"
    "vmov         q6, q2\n\t"
    : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
    : "r"(step)
    : PISLAM_ALL_D_REGS);
        break;

      case 4:
  asm(
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
  asm(
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
  asm(
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
  asm(
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
      goto last_block;
  }

  // horizontal pass
  for (int i = 0; i < vblocks; i += 1) {
    uint8_t *ptr1 = &img[i*8][0];
    uint8_t *ptr2 = &img[i*8+1][0];
    uint8_t *ptr3 = &buffer[i*128];
    bool hfix = false;

  asm(
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
  asm(
    // load next block of columns, compute missing row 7, transpose and store
    "vld4.8       {d16,d18,d20,d22}, [%2]!\n\t"
    "vld4.8       {d17,d19,d21,d23}, [%2]!\n\t"

    "vuzp.32      q8, q9\n\t"
    "vuzp.32      q10, q11\n\t"

    "vrhadd.u8     q12, q8, q13\n\t"
    "vrhadd.u8     q14, q8, q15\n\t"

    "vrhadd.u8     q12, q12, q15\n\t"
    "vrhadd.u8     q12, q12, q15\n\t"

    "vrhadd.u8     d14, d24, d15\n\t"
    "vrhadd.u8     d15, d25, d28\n\t"
    : "+r"(ptr1), "+r"(ptr2), "+r"(ptr3)
    : "r"(step)
    : PISLAM_ALL_D_REGS);

hstore:

  asm(
    // transpose everything
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

    ptr1 -= step*4-16;
    ptr2 -= step*4-16;

hload_skip:

  asm(
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

    "vrhadd.u8     d0, d0, d29\n\t"
    "vrhadd.u8     d1, d1, d2\n\t"
    "vrhadd.u8     d4, d4, d3\n\t"
    "vrhadd.u8     d5, d5, d6\n\t"
    "vrhadd.u8     d8, d8, d7\n\t"
    "vrhadd.u8     d9, d9, d20\n\t"

    "vld4.8       {d24,d26,d28,d30}, [%2]!\n\t"
    "vld4.8       {d25,d27,d29,d31}, [%2]!\n\t"

    "vuzp.32      q12, q13\n\t"
    "vuzp.32      q14, q15\n\t"

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

    // last row short delta gets passed to next loop iteration in q7
    "vrhadd.u8     q7, q13, q15\n\t"

    // compute remaining short deltas, q10 is holding long delta for q6
    "vrhadd.u8     q8, q11, q12\n\t"
    "vrhadd.u8     q9, q12, q14\n\t"
    "vrhadd.u8     q11, q13, q14\n\t"

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

    // last block
    uint32_t remainder = width & 0xf;
    if (remainder && (remainder <= 6)) {
      // This is the easy case, fire a second iteration with
      // columns of first block reflected appropriately.
      if (!hfix) {
        hfix = true;
        asm(
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
              asm(
                "vmov         d17, d31\n\t"
                "vmov         d20, d30\n\t"
              ::: PISLAM_ALL_D_REGS);
            } else { // remainder == 2
              asm(
                "vmov         d20, d16\n\t"
                "vmov         d21, d31\n\t"
              ::: PISLAM_ALL_D_REGS);
            }
          } else {
            if (remainder == 3) {
              asm(
                "vmov         d21, d17\n\t"
                "vmov         d18, d16\n\t"
              ::: PISLAM_ALL_D_REGS);
            } else { // remainder == 4
              asm(
                "vmov         d18, d20\n\t"
                "vmov         d19, d17\n\t"
              ::: PISLAM_ALL_D_REGS);
            }
          }
        } else {
          if (remainder == 5) {
            asm(
              "vmov         d19, d21\n\t"
              "vmov         d22, d20\n\t"
            ::: PISLAM_ALL_D_REGS);
          } else { // remainder == 6
            asm(
              "vmov         d22, d18\n\t"
              "vmov         d23, d21\n\t"
            ::: PISLAM_ALL_D_REGS);
          }
        }

        // compute row 7 as usual
        asm (
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
        asm(
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
#define PISLAM_BLUR_COMPUTE_LAST(c0, c1, c2, c3, out0, out1, tmp) asm(\
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
          asm(
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
      asm(
        // transpose everything
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
}

} /* namespace */

#endif /* PISLAM_GAUSSIAN_BLUR_H__ */
