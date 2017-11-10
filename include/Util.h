#ifndef PISLAM_UTIL_H_
#define PISLAM_UTIL_H_

#include <cstdint>

namespace pislam {

static inline uint32_t encodeFast(uint32_t score, uint32_t x, uint32_t y) {
  return (score << 24) | (x << 12) | y;
}

static inline uint32_t reencodeFastScore(uint32_t score, uint32_t encoded) {
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

static inline uint32_t encodeOrb(uint32_t octave, uint32_t orientation,
    uint32_t fast) {

  return (octave << 29) | (orientation << 24) | (fast & 0xffffff);
}

static inline uint32_t decodeOrbOctave(uint32_t encoded) {
  return (encoded >> 29);
}

static inline uint32_t decodeOrbOrientation(uint32_t encoded) {
  return (encoded >> 24) & 0x1f;
}

// scale x and y by scale/65536
static inline uint32_t scaleKeypoint(uint32_t encoded, uint32_t scale) {
  // mvScaleFactors < 0x10
  // point.[xy] < 0x1000
  // leaves 0x10000 for precision
  uint32_t x = (scale * decodeFastX(encoded)) >> 16;
  uint32_t y = (scale * decodeFastY(encoded)) >> 16;
  return (encoded & 0xff000000) | (x << 12) | (y);
}

} /* namespace pislam */
#endif /* PISLAM_FAST_H_ */
