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

#ifndef PISLAM_BILINEAR_H__
#define PISLAM_BILINEAR_H__

#include "arm_neon.h"

namespace pislam {

/// Reduce image size by 7/8, performing bilinear interpolation.
/// This ratio is significantly larger than 5/6 as used by pislam,
/// but can be combined with 13/16 reductions to produce a very
/// similar image pyramid.
///
/// Image must be padded to multiple of 8 bytes in both dimensions.
///
/// Round down when computing output dimensions.
/// For example, a 39x29 image scales to a 34x25.
///
/// img and out may be same pointer, in which case scaling is
/// done in place.
///
/// Cost on raspberry pi is essentially round trip memory time,
/// or 0.5 ms for a 640x480 image.
template <int vstep>
void bilinear7_8(const int width, const int height,
    uint8_t img[][vstep], uint8_t out[][vstep]) {

  // gcc is able to interleave the loop body extremely nicely.
  // The measured cost is basically a memory round trip

  uint8x8_t filter0   = {238, 201, 165, 128,  91,  55,   18,   0};
  uint8x8_t filter1   = {  0,  18,  55,  91, 128, 165,  201, 238};
  uint16x4_t filterw0 = {238, 201, 165, 128};
  uint16x4_t filterw1 = {91,  55,   18,   0};

  uint8_t *out_ptr;
  int y, yy;
  for (y = 0, yy = 0; y < height; y += 8, yy += 7) {
    out_ptr = &out[yy][0];
    for (int x = 0; x < width; x += 8) {
      uint8x8_t r0 = vld1_u8(&img[y+0][x]);
      uint8x8_t r1 = vld1_u8(&img[y+1][x]);
      uint8x8_t r2 = vld1_u8(&img[y+2][x]);
      uint8x8_t r3 = vld1_u8(&img[y+3][x]);
      uint8x8_t r4 = vld1_u8(&img[y+4][x]);
      uint8x8_t r5 = vld1_u8(&img[y+5][x]);
      uint8x8_t r6 = vld1_u8(&img[y+6][x]);
      uint8x8_t r7 = vld1_u8(&img[y+7][x]);

      uint16x8_t h00 = vmull_u8(r0, filter0);
      uint16x8_t h01 = vmull_u8(r1, filter0);
      uint16x8_t h02 = vmull_u8(r2, filter0);
      uint16x8_t h03 = vmull_u8(r3, filter0);
      uint16x8_t h04 = vmull_u8(r4, filter0);
      uint16x8_t h05 = vmull_u8(r5, filter0);
      uint16x8_t h06 = vmull_u8(r6, filter0);
      uint16x8_t h07 = vmull_u8(r7, filter0);
              
      uint16x8_t h10 = vmull_u8(r0, filter1);
      uint16x8_t h11 = vmull_u8(r1, filter1);
      uint16x8_t h12 = vmull_u8(r2, filter1);
      uint16x8_t h13 = vmull_u8(r3, filter1);
      uint16x8_t h14 = vmull_u8(r4, filter1);
      uint16x8_t h15 = vmull_u8(r5, filter1);
      uint16x8_t h16 = vmull_u8(r6, filter1);
      uint16x8_t h17 = vmull_u8(r7, filter1);
               
      // roll filter 1 results
      h10 = vextq_u16(h10, h10, 1);
      h11 = vextq_u16(h11, h11, 1);
      h12 = vextq_u16(h12, h12, 1);
      h13 = vextq_u16(h13, h13, 1);
      h14 = vextq_u16(h14, h14, 1);
      h15 = vextq_u16(h15, h15, 1);
      h16 = vextq_u16(h16, h16, 1);
      h17 = vextq_u16(h17, h17, 1);

      uint16x8_t h0 = vaddq_u16(h00, h10);
      uint16x8_t h1 = vaddq_u16(h01, h11);
      uint16x8_t h2 = vaddq_u16(h02, h12);
      uint16x8_t h3 = vaddq_u16(h03, h13);
      uint16x8_t h4 = vaddq_u16(h04, h14);
      uint16x8_t h5 = vaddq_u16(h05, h15);
      uint16x8_t h6 = vaddq_u16(h06, h16);
      uint16x8_t h7 = vaddq_u16(h07, h17);

      h0 = vrshrq_n_u16(h0, 8);
      h1 = vrshrq_n_u16(h1, 8);
      h2 = vrshrq_n_u16(h2, 8);
      h3 = vrshrq_n_u16(h3, 8);
      h4 = vrshrq_n_u16(h4, 8);
      h5 = vrshrq_n_u16(h5, 8);
      h6 = vrshrq_n_u16(h6, 8);
      h7 = vrshrq_n_u16(h7, 8);

      // compute results
      uint16x8_t v00 = vmulq_lane_u16(h0, filterw0, 0);
      uint16x8_t v01 = vmulq_lane_u16(h1, filterw0, 1);
      uint16x8_t v02 = vmulq_lane_u16(h2, filterw0, 2);
      uint16x8_t v03 = vmulq_lane_u16(h3, filterw0, 3);
      uint16x8_t v04 = vmulq_lane_u16(h4, filterw1, 0);
      uint16x8_t v05 = vmulq_lane_u16(h5, filterw1, 1);
      uint16x8_t v06 = vmulq_lane_u16(h6, filterw1, 2);
                                                 
      uint16x8_t v10 = vmulq_lane_u16(h1, filterw1, 2);
      uint16x8_t v11 = vmulq_lane_u16(h2, filterw1, 1);
      uint16x8_t v12 = vmulq_lane_u16(h3, filterw1, 0);
      uint16x8_t v13 = vmulq_lane_u16(h4, filterw0, 3);
      uint16x8_t v14 = vmulq_lane_u16(h5, filterw0, 2);
      uint16x8_t v15 = vmulq_lane_u16(h6, filterw0, 1);
      uint16x8_t v16 = vmulq_lane_u16(h7, filterw0, 0);

      uint8x8_t b0 = vraddhn_u16(v00, v10);
      uint8x8_t b1 = vraddhn_u16(v01, v11);
      uint8x8_t b2 = vraddhn_u16(v02, v12);
      uint8x8_t b3 = vraddhn_u16(v03, v13);
      uint8x8_t b4 = vraddhn_u16(v04, v14);
      uint8x8_t b5 = vraddhn_u16(v05, v15);
      uint8x8_t b6 = vraddhn_u16(v06, v16);

      vst1_u8(&out_ptr[0*vstep], b0);
      vst1_u8(&out_ptr[1*vstep], b1);
      vst1_u8(&out_ptr[2*vstep], b2);
      vst1_u8(&out_ptr[3*vstep], b3);
      vst1_u8(&out_ptr[4*vstep], b4);
      vst1_u8(&out_ptr[5*vstep], b5);
      vst1_u8(&out_ptr[6*vstep], b6);

      out_ptr += 7;
    }
  }
}

/// Reduce image size by 13/16, performing bilinear interpolation.
/// This ratio is 1/40th away from 5/6, the ratio used by pislam.
///
/// Image must be padded to multiple of 16 bytes in both dimensions.
///
/// Round down when computing output dimensions.
/// For example, a 39x29 image scales to a 31x23.
///
/// img and out may be same pointer, in which case scaling is
/// done in place.
///
/// Cost on raspberry pi is more expensive than 7_8, costing
/// 0.7 ms for a 640x480 image.
template <int vstep>
void bilinear13_16(const int width, const int height,
    uint8_t img[][vstep], uint8_t out[][vstep]) {

  // TODO[carl]: These filter banks are hogging eight D registers.
  // If they can be freed gcc will have more freedom to interleave
  // operations.
  uint8x8_t filter0l  = {226, 167, 108,  49,   0, 246, 187, 128};
  uint8x8_t filter0h  = { 69,  10,   0, 207, 138,  89,  30,   0};
  uint8x8_t filter1l  = {  0,  30,  89, 138, 207,   0,  10,  69};
  uint8x8_t filter1h  = {128, 187, 246,   0,  49, 108, 167, 226};

  uint16x4_t filterw0 = {226, 167, 108,  49};
  uint16x4_t filterw1 = {246, 187, 128,  69};
  uint16x4_t filterw2 = { 10, 207, 138,  89};
  uint16x4_t filterw3 = { 30,   0,   0,   0};

  uint8_t *out_ptr;
  int y, yy;
  for (y = 0, yy = 0; y < height; y += 16, yy += 13) {
    out_ptr = &out[yy][0];
    for (int x = 0; x < width; x += 16) {
      // It is too ridiculous to list these all out.
      // gcc manages to reorder these and interleave appropriately
      // with only one register spill.
#define PISLAM_BILINEAR_13_16_H(n) \
      uint8x16_t r ## n = vld1q_u8(&img[y+0x ## n][x]); \
      uint16x8_t h0 ## n ## l = vmull_u8(vget_low_u8 (r ## n), filter0l); \
      uint16x8_t h0 ## n ## h = vmull_u8(vget_high_u8(r ## n), filter0h); \
      uint16x8_t h1 ## n ## l = vmull_u8(vget_low_u8 (r ## n), filter1l); \
      uint16x8_t h1 ## n ## h = vmull_u8(vget_high_u8(r ## n), filter1h); \
      h1 ## n ## l = vextq_u16(h1 ## n ## l, h1 ## n ## h, 1); \
      h1 ## n ## h = vextq_u16(h1 ## n ## h, h1 ## n ## h, 1); \
      uint16x8_t h ## n ## l = vaddq_u16(h0 ## n ## l, h1 ## n ## l); \
      uint16x8_t h ## n ## h = vaddq_u16(h0 ## n ## h, h1 ## n ## h); \
      h ## n ## l = vrshrq_n_u16(h ## n ## l, 8); \
      h ## n ## h = vrshrq_n_u16(h ## n ## h, 8)

      PISLAM_BILINEAR_13_16_H(0);
      PISLAM_BILINEAR_13_16_H(1);
      PISLAM_BILINEAR_13_16_H(2);
      PISLAM_BILINEAR_13_16_H(3);
      PISLAM_BILINEAR_13_16_H(4);
      PISLAM_BILINEAR_13_16_H(5);
      PISLAM_BILINEAR_13_16_H(6);
      PISLAM_BILINEAR_13_16_H(7);
      PISLAM_BILINEAR_13_16_H(8);
      PISLAM_BILINEAR_13_16_H(9);
      PISLAM_BILINEAR_13_16_H(a);
      PISLAM_BILINEAR_13_16_H(b);
      PISLAM_BILINEAR_13_16_H(c);
      PISLAM_BILINEAR_13_16_H(d);
      PISLAM_BILINEAR_13_16_H(e);
      PISLAM_BILINEAR_13_16_H(f);

      // compute results
      uint16x8_t v00l = vmulq_lane_u16(h0l, filterw0, 0);
      uint16x8_t v00h = vmulq_lane_u16(h0h, filterw0, 0);
      uint16x8_t v01l = vmulq_lane_u16(h1l, filterw0, 1);
      uint16x8_t v01h = vmulq_lane_u16(h1h, filterw0, 1);
      uint16x8_t v02l = vmulq_lane_u16(h2l, filterw0, 2);
      uint16x8_t v02h = vmulq_lane_u16(h2h, filterw0, 2);
      uint16x8_t v03l = vmulq_lane_u16(h3l, filterw0, 3);
      uint16x8_t v03h = vmulq_lane_u16(h3h, filterw0, 3);
      uint16x8_t v04l = vmulq_lane_u16(h5l, filterw1, 0);
      uint16x8_t v04h = vmulq_lane_u16(h5h, filterw1, 0);
      uint16x8_t v05l = vmulq_lane_u16(h6l, filterw1, 1);
      uint16x8_t v05h = vmulq_lane_u16(h6h, filterw1, 1);
      uint16x8_t v06l = vmulq_lane_u16(h7l, filterw1, 2);
      uint16x8_t v06h = vmulq_lane_u16(h7h, filterw1, 2);
      uint16x8_t v07l = vmulq_lane_u16(h8l, filterw1, 3);
      uint16x8_t v07h = vmulq_lane_u16(h8h, filterw1, 3);
      uint16x8_t v08l = vmulq_lane_u16(h9l, filterw2, 0);
      uint16x8_t v08h = vmulq_lane_u16(h9h, filterw2, 0);
      uint16x8_t v09l = vmulq_lane_u16(hbl, filterw2, 1);
      uint16x8_t v09h = vmulq_lane_u16(hbh, filterw2, 1);
      uint16x8_t v0al = vmulq_lane_u16(hcl, filterw2, 2);
      uint16x8_t v0ah = vmulq_lane_u16(hch, filterw2, 2);
      uint16x8_t v0bl = vmulq_lane_u16(hdl, filterw2, 3);
      uint16x8_t v0bh = vmulq_lane_u16(hdh, filterw2, 3);
      uint16x8_t v0cl = vmulq_lane_u16(hel, filterw3, 0);
      uint16x8_t v0ch = vmulq_lane_u16(heh, filterw3, 0);
                                                 
      uint16x8_t v10l = vmulq_lane_u16(h1l, filterw3, 0);
      uint16x8_t v10h = vmulq_lane_u16(h1h, filterw3, 0);
      uint16x8_t v11l = vmulq_lane_u16(h2l, filterw2, 3);
      uint16x8_t v11h = vmulq_lane_u16(h2h, filterw2, 3);
      uint16x8_t v12l = vmulq_lane_u16(h3l, filterw2, 2);
      uint16x8_t v12h = vmulq_lane_u16(h3h, filterw2, 2);
      uint16x8_t v13l = vmulq_lane_u16(h4l, filterw2, 1);
      uint16x8_t v13h = vmulq_lane_u16(h4h, filterw2, 1);
      uint16x8_t v14l = vmulq_lane_u16(h6l, filterw2, 0);
      uint16x8_t v14h = vmulq_lane_u16(h6h, filterw2, 0);
      uint16x8_t v15l = vmulq_lane_u16(h7l, filterw1, 3);
      uint16x8_t v15h = vmulq_lane_u16(h7h, filterw1, 3);
      uint16x8_t v16l = vmulq_lane_u16(h8l, filterw1, 2);
      uint16x8_t v16h = vmulq_lane_u16(h8h, filterw1, 2);
      uint16x8_t v17l = vmulq_lane_u16(h9l, filterw1, 1);
      uint16x8_t v17h = vmulq_lane_u16(h9h, filterw1, 1);
      uint16x8_t v18l = vmulq_lane_u16(hal, filterw1, 0);
      uint16x8_t v18h = vmulq_lane_u16(hah, filterw1, 0);
      uint16x8_t v19l = vmulq_lane_u16(hcl, filterw0, 3);
      uint16x8_t v19h = vmulq_lane_u16(hch, filterw0, 3);
      uint16x8_t v1al = vmulq_lane_u16(hdl, filterw0, 2);
      uint16x8_t v1ah = vmulq_lane_u16(hdh, filterw0, 2);
      uint16x8_t v1bl = vmulq_lane_u16(hel, filterw0, 1);
      uint16x8_t v1bh = vmulq_lane_u16(heh, filterw0, 1);
      uint16x8_t v1cl = vmulq_lane_u16(hfl, filterw0, 0);
      uint16x8_t v1ch = vmulq_lane_u16(hfh, filterw0, 0);

      // storing is a pain since there are holes in the vectors.
      // gcc actually makes a mess of this, but its not the
      // worst thing in the code base.
      //
      // TODO: measure time for storing bytes individually
      // and see if this mess is actually faster.
#define PISLAM_BILINEAR_13_16_V(n) \
      uint16x4_t v0 ## n ## ll = vget_low_u16(v0 ## n ## l); \
      uint16x4_t v0 ## n ## lh = vget_high_u16(v0 ## n ## l); \
      uint16x4_t v1 ## n ## ll = vget_low_u16(v1 ## n ## l); \
      uint16x4_t v1 ## n ## lh = vget_high_u16(v1 ## n ## l); \
      v0 ## n ## lh = vreinterpret_u16_u64( \
          vshr_n_u64(vreinterpret_u64_u16(v0 ## n ## lh), 16)); \
      v1 ## n ## lh = vreinterpret_u16_u64( \
          vshr_n_u64(vreinterpret_u64_u16(v1 ## n ## lh), 16)); \
      v0 ## n ## l = vcombine_u16(v0 ## n ## ll, v0 ## n ## lh); \
      v1 ## n ## l = vcombine_u16(v1 ## n ## ll, v1 ## n ## lh); \
      uint8x8_t b ## n ## l = vraddhn_u16(v0 ## n ## l, v1 ## n ## l); \
      uint8x8_t b ## n ## h = vraddhn_u16(v0 ## n ## h, v1 ## n ## h); \
      b ## n ## l = vreinterpret_u8_u64(vsli_n_u64( \
        vreinterpret_u64_u8(b ## n ## l), \
        vreinterpret_u64_u8(b ## n ## h), 56)); \
      vst1_u8(&out_ptr[(0x ## n)*vstep], b ## n ## l); \
      vst1_lane_u8(&out_ptr[(0x ## n)*vstep + 8], b ## n ## h, 1); \
      b ## n ## h = vreinterpret_u8_u64(vshr_n_u64( \
        vreinterpret_u64_u8(b ## n ## h), 24)); \
      vst1_lane_u32((uint32_t *)&out_ptr[(0x ## n)*vstep + 9], \
          vreinterpret_u32_u8(b ## n ## h), 0)

      PISLAM_BILINEAR_13_16_V(0);
      PISLAM_BILINEAR_13_16_V(1);
      PISLAM_BILINEAR_13_16_V(2);
      PISLAM_BILINEAR_13_16_V(3);
      PISLAM_BILINEAR_13_16_V(4);
      PISLAM_BILINEAR_13_16_V(5);
      PISLAM_BILINEAR_13_16_V(6);
      PISLAM_BILINEAR_13_16_V(7);
      PISLAM_BILINEAR_13_16_V(8);
      PISLAM_BILINEAR_13_16_V(9);
      PISLAM_BILINEAR_13_16_V(a);
      PISLAM_BILINEAR_13_16_V(b);
      PISLAM_BILINEAR_13_16_V(c);

      out_ptr += 13;
    }
  }
}

} /* namespace pislam */
#endif /* PISLAM_BILINEAR_H__ */
