#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

namespace test_util {

void fill_spiral(int vstep, int width, int height, int cx, int cy,
    uint8_t *buffer) {

  std::fill(buffer,buffer+vstep*height,0);

  float phi = (1 + sqrtf(5)) / 2;
  for (float theta = 0; theta < 20; theta += 0.01) {
    float r = powf(phi, theta*M_2_PI);
    float x = r*cosf(theta);
    float y = r*sinf(theta);

    int i = y + cy;
    int j = x + cx;
    
    if (0 <= i && i < int(vstep) && 0 <= j && j < int(vstep)) {
      buffer[i*vstep+j] = 0xff;
    }

    i = -y + cy;
    j = -x + cx;
    
    if (0 <= i && i < int(vstep) && 0 <= j && j < int(vstep)) {
      buffer[i*vstep+j] = 0xff;
    }
  }
}

void fill_random(int vstep, int width, int height, uint8_t *buffer) {

  std::mt19937_64 rng;
  for (int i = 0; i < height; i += 1) {
    for (int j = 0; j < width; j += 1) {
      buffer[i*vstep+j] = rng();
    }
  }
}

void print_buffer(int vstep, int width, int height, uint8_t *buffer, int fw) {
  for (int i = 0; i < height; i += 1) {
    for (int j = 0; j < width; j += 1) {
      std::cout << std::setw(fw) << (int)buffer[i*vstep+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

} /* namespace test_util */
