#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "../include/Bilinear.h"
#include "TestUtil.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

class BilinearTest: public ::testing::TestWithParam<::testing::tuple<int, int>> {};

#define RSHR(a, n) (((a) >> (n)) + (((a) >> ((n)-1))&1))

static void reference7_8(const int vstep, const int width,
    const int height, uint8_t *m);
static void reference13_16(const int vstep, const int width,
    const int height, uint8_t *m);

TEST_P(BilinearTest, spiral7_8) {
  constexpr size_t vstep = 64;

  size_t width = ::testing::get<0>(GetParam());
  size_t height = ::testing::get<1>(GetParam());

  uint8_t spiral[vstep*vstep];
  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];

  test_util::fill_spiral(vstep, width, height, vstep/3, vstep/3, spiral);

  std::copy(spiral, spiral+height*vstep, a);
  std::copy(spiral, spiral+height*vstep, b);

  reference7_8(vstep, width, height, a);
  pislam::bilinear7_8<vstep>(width, height,
      (uint8_t (*)[vstep])b, (uint8_t (*)[vstep])b);

  size_t out_height = height * 7 / 8;
  size_t out_width = width * 7 / 8;

#if 0
  test_util::print_buffer(vstep, width, height, spiral, 3);
  test_util::print_buffer(vstep, width, height, a, 3);
  test_util::print_buffer(vstep, width, height, b, 3);

  for (size_t i = 0; i < out_height; i += 1) {
    for (size_t j = 0; j < out_width; j += 1) {
      if (a[i*vstep+j] != b[i*vstep+j]) {
        std::cout << i << ", " << j << std::endl;
      }
    }
  }
#else
  for (size_t i = 0; i < out_height; i += 1) {
    for (size_t j = 0; j < out_width; j += 1) {
      ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
#endif
}

TEST_P(BilinearTest, random7_8) {
  constexpr size_t vstep = 64;

  size_t width = ::testing::get<0>(GetParam());
  size_t height = ::testing::get<1>(GetParam());

  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];


  test_util::fill_random(vstep, width, height, a);
  std::copy(a, a+vstep*vstep, b);

  reference7_8(vstep, width, height, a);
  pislam::bilinear7_8<vstep>(width, height,
      (uint8_t (*)[vstep])b, (uint8_t (*)[vstep])b);

  size_t out_height = height * 7 / 8;
  size_t out_width = width * 7 / 8;
  for (size_t i = 0; i < out_height; i += 1) {
    for (size_t j = 0; j < out_width; j += 1) {
      ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
}

TEST_P(BilinearTest, spiral13_16) {
  constexpr size_t vstep = 64;

  size_t width = ::testing::get<0>(GetParam());
  size_t height = ::testing::get<1>(GetParam());

  uint8_t spiral[vstep*vstep];
  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];

  test_util::fill_spiral(vstep, width, height, vstep/3, vstep/3, spiral);

  std::copy(spiral, spiral+height*vstep, a);
  std::copy(spiral, spiral+height*vstep, b);

  reference13_16(vstep, width, height, a);
  pislam::bilinear13_16<vstep>(width, height,
      (uint8_t (*)[vstep])b, (uint8_t (*)[vstep])b);

  size_t out_height = height * 13 / 16;
  size_t out_width = width * 13 / 16;
  for (size_t i = 0; i < out_height; i += 1) {
    for (size_t j = 0; j < out_width; j += 1) {
      ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
}

TEST_P(BilinearTest, random13_16) {
  constexpr size_t vstep = 64;

  size_t width = ::testing::get<0>(GetParam());
  size_t height = ::testing::get<1>(GetParam());

  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];


  test_util::fill_random(vstep, width, height, a);
  std::copy(a, a+vstep*vstep, b);

  reference13_16(vstep, width, height, a);
  pislam::bilinear13_16<vstep>(width, height,
      (uint8_t (*)[vstep])b, (uint8_t (*)[vstep])b);

  size_t out_height = height * 13 / 16;
  size_t out_width = width * 13 / 16;
  for (size_t i = 0; i < out_height; i += 1) {
    for (size_t j = 0; j < out_width; j += 1) {
      ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    DimensionTest,
    BilinearTest,
    Combine(Range(1, 48), Range(1, 48)));
    //Combine(Values(34), Values(34)));

void reference7_8(const int vstep, const int width,
    const int height, uint8_t *m) {

  int filter[] = {238, 201, 165, 128,  91,  55,   18};

  int i, j, oi, oj;
  for (i = 0, oi = 0; i < height; i += 8, oi += 7) {
    for (j = 0, oj = 0; j < width; j += 8, oj += 7) {
      for (int y = 0; y < 7; y += 1) {
        for (int x = 0; x < 7; x += 1) {
          int p00 = m[vstep*(i+y  )+(j+x  )];
          int p01 = m[vstep*(i+y  )+(j+x+1)];
          int p10 = m[vstep*(i+y+1)+(j+x)];
          int p11 = m[vstep*(i+y+1)+(j+x+1)];

          int h0 = RSHR(p00 * filter[x] + p01 * filter[6-x], 8);
          int h1 = RSHR(p10 * filter[x] + p11 * filter[6-x], 8);

          int b = RSHR(h0 * filter[y] + h1 * filter[6-y], 8);

          m[vstep*(oi+y)+(oj+x)] = b;
        }
      }
    }
  }
}

static int map13(int i) {
  if (i > 3) {
    i += 1;
  }
  if (i > 9) {
    i += 1;
  }
  return i;
}

void reference13_16(const int vstep, const int width,
    const int height, uint8_t *m) {

  int filter[] = {226, 167, 108, 49, 246, 187, 128, 69, 10, 207, 138, 89, 30};

  int i, j, oi, oj;
  for (i = 0, oi = 0; i < height; i += 16, oi += 13) {
    for (j = 0, oj = 0; j < width; j += 16, oj += 13) {
      for (int y = 0; y < 13; y += 1) {
        for (int x = 0; x < 13; x += 1) {
          int p00 = m[vstep*(i+map13(y)  )+(j+map13(x)  )];
          int p01 = m[vstep*(i+map13(y)  )+(j+map13(x)+1)];
          int p10 = m[vstep*(i+map13(y)+1)+(j+map13(x))];
          int p11 = m[vstep*(i+map13(y)+1)+(j+map13(x)+1)];

          int h0 = RSHR(p00 * filter[x] + p01 * filter[12-x], 8);
          int h1 = RSHR(p10 * filter[x] + p11 * filter[12-x], 8);

          int b = RSHR(h0 * filter[y] + h1 * filter[12-y], 8);

          m[vstep*(oi+y)+(oj+x)] = b;
        }
      }
    }
  }
}

} /* namespace */
