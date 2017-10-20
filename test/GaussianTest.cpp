#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "../include/Gaussian.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

#define RHADD(a, b) ((a >> 1) + (b >> 1) + ((a|b)&1))

class GaussianTest: public ::testing::TestWithParam<::testing::tuple<int, int>> {};

static void reference(const int vstep, const int width,
    const int height, uint8_t *m);

TEST_P(GaussianTest, spiral) {
  constexpr size_t vstep = 640;

  size_t width = 640; //::testing::get<0>(GetParam());
  size_t height = 480; // ::testing::get<1>(GetParam());

  uint8_t spiral[vstep*vstep];
  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];

  /*
  std::fill(spiral,spiral+vstep*vstep,0);

  float phi = (1 + sqrtf(5)) / 2;
  for (float theta = 0; theta < 20; theta += 0.01) {
    float r = powf(phi, theta*M_2_PI);
    float x = r*cosf(theta);
    float y = r*sinf(theta);

    int i = y + vstep / 3;
    int j = x + vstep / 3;
    
    if (0 <= i && i < int(vstep) && 0 <= j && j < int(vstep)) {
      spiral[i*vstep+j] = 0xff;
    }

    i = -y + vstep / 3;
    j = -x + vstep / 3;
    
    if (0 <= i && i < int(vstep) && 0 <= j && j < int(vstep)) {
      spiral[i*vstep+j] = 0xff;
    }
  }

  */
  /*
  std::fill(spiral,spiral+vstep*vstep,0);

  for (int i= 0; i < vstep; i += 1) {
    spiral[i*vstep + i] = 255;
  }
  */

  /*
  std::copy(spiral, spiral+height*vstep, a);
  std::copy(spiral, spiral+height*vstep, b);
  */

  //reference(vstep, width, height, a);
  for (int i = 0; i < 100000; i ++)
  pislam::gaussian5x5<vstep>(width, height, (uint8_t (*)[vstep])b);


#if 0
  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)spiral[i*vstep+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)a[i*vstep+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)b[i*vstep+j] << " ";
    }
    std::cout << std::endl;
  }

  /*
  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      if (a[i*vstep+j] != b[i*vstep+j]) {
        std::cout << i << ", " << j << std::endl;
      }
    }
  }
  */
#else
  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      //ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
#endif
}

TEST_P(GaussianTest, random) {
  constexpr size_t vstep = 64;

  size_t width = ::testing::get<0>(GetParam());
  size_t height = ::testing::get<1>(GetParam());

  uint8_t a[vstep*vstep];
  uint8_t b[vstep*vstep];

  std::fill(a, a+vstep*vstep, 0);

  std::mt19937_64 rng;
  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      b[i*vstep+j] = rng();
    }
  }

  std::copy(a, a+vstep*vstep, b);

  reference(vstep, width, height, a);
  pislam::gaussian5x5<vstep>(width, height, (uint8_t (*)[vstep])b);


  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      ASSERT_EQ(a[i*vstep+j], b[i*vstep+j]);
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    DimensionTest,
    GaussianTest,
    //Combine(Range(16, 33), Range(16, 33)));
    Combine(Values(32), Values(32)));

static void reference(const int vstep, const int width,
    const int height, uint8_t *m) {

  // vertical pass
  for (int j = 0; j < width; j += 1) {
    uint8_t a, b, c, d, e;
    a = m[2*vstep+j];
    b = m[1*vstep+j];
    c = m[0*vstep+j];
    d = m[1*vstep+j];
    for (int i = 0; i < height; i += 1) {
      if (i == height - 2) {
        e = c;
      } else if (i == height - 1) {
        e = a;
      } else {
        e = m[(i+2)*vstep+(j)];
      }

      uint8_t x = RHADD(a, e);
      uint8_t y = RHADD(b, d);
      x = RHADD(x, c);
      x = RHADD(x, c);

      m[(i)*vstep+(j)] = RHADD(x, y);

      a = b; b = c; c = d; d = e;
    }
  }

  // horizontal pass
  for (int i = 0; i < height; i += 1) {
    uint8_t a, b, c, d, e;
    a = m[i*vstep+2];
    b = m[i*vstep+1];
    c = m[i*vstep+0];
    d = m[i*vstep+1];
    for (int j = 0; j < width; j += 1) {
      if (j == width - 2) {
        e = c;
      } else if (j == width - 1) {
        e = a;
      } else {
        e = m[(i)*vstep+(j+2)];
      }

      uint8_t x = RHADD(a, e);
      uint8_t y = RHADD(b, d);
      x = RHADD(x, c);
      x = RHADD(x, c);

      m[(i)*vstep+(j)] = RHADD(x, y);

      a = b; b = c; c = d; d = e;
    }
  }
}

} /* namespace */
