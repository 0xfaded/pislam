#ifndef PISLAM_TEST_UTIL_H__
#define PISLAM_TEST_UTIL_H__

#include <cmath>
#include <random>

namespace test_util {

void fill_spiral(int vstep, int width, int height, int cx, int cy,
    uint8_t *buffer);

void fill_random(int vstep, int width, int height, uint8_t *buffer);

void print_buffer(int vstep, int width, int height, uint8_t *buffer, int fw);

} /* namespace test_util */

#endif /* PISLAM_TEST_UTIL_H__ */
