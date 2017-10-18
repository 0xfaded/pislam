#include <Eigen/Core>
#include <cmath>

#include "gtest/gtest.h"
#include "../include/GaussianBlur.h"

namespace {

#define RHADD(a, b) ((a >> 1) + (b >> 1) + ((a|b)&1))

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXu8;

static void reference(const int width, const int height, uint8_t *m) {
  // vertical pass
  for (int j = 0; j < width; j += 1) {
    uint8_t a, b, c, d, e;
    a = m[2*width+j];
    b = m[1*width+j];
    c = m[0*width+j];
    d = m[1*width+j];
    for (int i = 0; i < height; i += 1) {
      if (i == height - 2) {
        e = c;
      } else if (i == height - 1) {
        e = a;
      } else {
        e = m[(i+2)*width+(j)];
      }

      uint8_t x = RHADD(a, e);
      uint8_t y = RHADD(b, d);
      x = RHADD(x, c);
      x = RHADD(x, c);

      m[(i)*width+(j)] = RHADD(x, y);

      a = b; b = c; c = d; d = e;
    }
  }

  // horizontal pass
  for (int i = 0; i < height; i += 1) {
    uint8_t a, b, c, d, e;
    a = m[i*width+2];
    b = m[i*width+1];
    c = m[i*width+0];
    d = m[i*width+1];
    for (int j = 0; j < width; j += 1) {
      if (j == width - 2) {
        e = c;
      } else if (j == width - 1) {
        e = a;
      } else {
        e = m[(i)*width+(j+2)];
      }

      if (i == 5) {
        std::cout << (int)a << " " << (int)b << " " << (int)c << " " << (int)d << " " << (int)e << std::endl;
      }

      uint8_t x = RHADD(a, e);
      uint8_t y = RHADD(b, d);
      x = RHADD(x, c);
      x = RHADD(x, c);

      m[(i)*width+(j)] = RHADD(x, y);

      a = b; b = c; c = d; d = e;
    }
  }
}


TEST(GaussianBlurTest, spiral) {
  constexpr size_t width = 64;
  constexpr size_t height = 64;

  uint8_t spiral[height*width];
  uint8_t a[height*width];
  uint8_t b[height*width];

  std::fill(spiral,spiral+width*height,0);

  float phi = (1 + sqrtf(5)) / 2;
  for (float theta = 0; theta < 20; theta += 0.01) {
    float r = powf(phi, theta*M_2_PI);
    float x = r*cosf(theta);
    float y = r*sinf(theta);

    int i = y + height / 2;
    int j = x + width / 2;
    
    if (0 <= i && i < int(height) && 0 <= j && j < int(width)) {
      spiral[i*width+j] = 0xff;
      spiral[(height-1-i)*width+(width-1-j)] = 0xff;
    }
  }

  std::copy(spiral, spiral+height*width, a);
  std::copy(spiral, spiral+height*width, b);

  reference(width, height, a);
  pislam::gaussian5x5<width>(width, height, (uint8_t (*)[width])b);


  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)spiral[i*width+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)a[i*width+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      std::cout << std::setw(3) << (int)b[i*width+j] << " ";
    }
    std::cout << std::endl;
  }

  for (size_t i = 0; i < height; i += 1) {
    for (size_t j = 0; j < width; j += 1) {
      if (a[i*width+j] != b[i*width+j]) {
        std::cout << i << ", " << j << std::endl;
      }
    }
  }

  /*
  std::cout << a.cast<int>();
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << b.cast<int>();
  std::cout << std::endl;
  */
}

} /* namespace */
