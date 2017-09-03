#ifndef PISLAM_UTIL_H_
#define PISLAM_UTIL_H_

#include <cstdint>

namespace pislam {

static inline uint32_t encodeFast(uint32_t score, uint32_t x, uint32_t y) {
  return (score << 24) | (x << 12) | y;
}

static inline uint32_t rencodeFastScore(uint32_t score, uint32_t encoded) {
  return (score << 24) | (encoded & 0xffffff);
}

static inline uint32_t decodeFastX(uint32_t encoded) {
  return (encoded >> 12) & 0xfff;
}

static inline uint32_t decodeFastY(uint32_t encoded) {
  return encoded & 0xfff;
}

static inline uint32_t decodeFastScore(uint32_t encoded) {
  return encoded >> 24;
}

} /* namespace pislam */
#endif /* PISLAM_FAST_H_ */
