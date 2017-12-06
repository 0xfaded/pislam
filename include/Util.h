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
