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

#include "Fast.h"
#include "Util.h"
#include "Orb.h"

#include <png.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <cassert>

void write_png_file(const char* file_name, int width, int height, int depth, uint8_t *buf);
uint8_t *read_png_file(const char *filename, uint32_t *width, uint32_t *height);

template<int vstep>
void paintPoint(uint8_t img[][vstep], int x, int y);

static int pyramidLevels[16] = {
    640, 480,
    533, 400,
    444, 333,
    370, 278,
    309, 231,
    257, 193,
    214, 161,
    179, 134
};

#define IMG_W 640

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./demo pyramid.png" << std::endl;
    return 1;
  }

  const char *fname = argv[1];

  uint32_t pyramidHeight = 0;
  for (size_t i = 1; i < sizeof(pyramidLevels)/sizeof(*pyramidLevels); i += 2) {
    pyramidHeight += pyramidLevels[i];
  }

  uint32_t width, height;
  uint8_t (*img)[IMG_W] = (uint8_t (*)[IMG_W])read_png_file(fname, &width, &height);

  assert(width == IMG_W && "Image width does not match compiled width");
  assert(height == pyramidHeight && "Image height does not match compiled pyramid height");

  uint8_t out[pyramidHeight][IMG_W];

  std::vector<uint32_t> points;
  std::vector<uint32_t> descriptors;

  std::clock_t begin = std::clock();

  uint32_t pyramidRow = 0;
  for (size_t i = 0; i < sizeof(pyramidLevels)/sizeof(*pyramidLevels); i += 2) {
    uint32_t levelWidth = pyramidLevels[i];
    uint32_t levelHeight = pyramidLevels[i+1];

    uint8_t (*imgPtr)[IMG_W] = &img[pyramidRow];
    uint8_t (*outPtr)[IMG_W] = &out[pyramidRow];

    pislam::fastDetect<IMG_W, 16>(levelWidth, levelHeight, imgPtr, outPtr, 20);
    pislam::fastScoreHarris<IMG_W, 16>(levelWidth, levelHeight, imgPtr, 1 << 15, outPtr);

    size_t oldSize = points.size();
    pislam::fastExtract<IMG_W, 16>(levelWidth, levelHeight, outPtr, points);

    // Adjust y coordinate to match position in image pyramid.
    for (auto p = points.begin() + oldSize; p < points.end(); ++ p) {
      uint32_t x = pislam::decodeFastX(*p);
      uint32_t y = pislam::decodeFastY(*p) + pyramidRow;
      uint32_t score = pislam::decodeFastScore(*p);
      *p = pislam::encodeFast(score, x, y);
    }

    pyramidRow += levelHeight;
  }
  pislam::orbCompute<IMG_W, 8>(img, points, descriptors);

  std::clock_t end = std::clock();

  for (uint32_t point: points) {
    uint32_t x = pislam::decodeFastX(point);
    uint32_t y = pislam::decodeFastY(point);
    paintPoint(img, x, y);
  }

  write_png_file("out.png", IMG_W, height, 1, (uint8_t *)img);

  std::cout << "CPU  Time: " << (end - begin) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  std::cout << points.size() << " features" << std::endl;

  return 0;
}

template<int vstep>
void paintPoint(uint8_t img[][vstep], int x, int y) {
  img[y-5][x] = 0;
  img[y-4][x] = 0;
  img[y+4][x] = 0;
  img[y+5][x] = 0;

  img[y][x-5] = 0;
  img[y][x-4] = 0;
  img[y][x+4] = 0;
  img[y][x+5] = 0;
}

void abort_(const char * s, ...) {
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}

uint8_t *read_png_file(const char *filename, uint32_t *width, uint32_t *height) {
  png_byte color_type;
  png_byte bit_depth;
  png_bytep *row_pointers;

  FILE *fp = fopen(filename, "rb");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  png_read_info(png, info);

  *width     = png_get_image_width(png, info);
  *height    = png_get_image_height(png, info);
  color_type = png_get_color_type(png, info);
  bit_depth  = png_get_bit_depth(png, info);

  // Read any color_type into 8bit depth, RGBA format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt

  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);

  row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
  for(uint32_t y = 0; y < *height; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }

  png_read_image(png, row_pointers);

  fclose(fp);

  uint8_t *out = (uint8_t *)malloc((*width)*(*height));
  for (uint32_t y = 0; y < *height; y += 1) {
    for (uint32_t x = 0; x < *width; x += 1) {
      out[y*(*width)+x] = row_pointers[y][4*x];
    }
    free(row_pointers[y]);
  }
  free(row_pointers);
  return out;
}

void write_png_file(const char* file_name, int width, int height, int depth, uint8_t *buf) {
    int y;

    png_byte color_type = PNG_COLOR_TYPE_GRAY;
    png_byte bit_depth = 8;

    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(*row_pointers) * height);

    for (y = 0; y < height; y += 1) {
        row_pointers[y] = buf + y * width * depth;
    }

    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp)
        abort_("[write_png_file] File %s could not be opened for writing", file_name);


    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[write_png_file] png_create_write_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[write_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during init_io");

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing header");

    png_set_IHDR(png_ptr, info_ptr, width, height,
            bit_depth, color_type, PNG_INTERLACE_NONE,
           PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing bytes");

    png_write_image(png_ptr, row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during end of write");

    png_write_end(png_ptr, NULL);

    /* cleanup heap allocation */
    free(row_pointers);

    fclose(fp);
}
