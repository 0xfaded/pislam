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
