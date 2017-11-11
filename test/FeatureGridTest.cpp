#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "../include/Util.h"
#include "../include/Fast.h"
#include "TestUtil.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

class FeatureGridTest: public ::testing::TestWithParam<::std::tuple<int, int, int, int>> {};

TEST(FeatureGridTest, GridReduce) {
  constexpr int bucketLimit = 5;
  constexpr int border = 16;
  constexpr int logBucketSize = 4;

  pislam::FeatureGrid<bucketLimit, logBucketSize, border> grid(640, 480);

  const int maxPerFourCell = 8;
  const int minPerFourCell = 4;
  const int step = 2;
  const int totalDesiredFeatures = 1000;

  std::mt19937_64 rng;
  for (size_t i = 0; i < grid.hBuckets * grid.vBuckets; i += 1) {
    pislam::FeatureBucket<bucketLimit> &bucket = grid.buckets.get()[i];
    bucket.count = rng() % (bucketLimit+1);
    for (uint32_t j = 0; j < bucket.count; j += 1) {
      bucket[j] = rng();
    }
    std::sort(&bucket[0], &bucket[bucket.count]);
  }

  pislam::FeatureGrid<bucketLimit, logBucketSize, border> referenceGrid =
    grid.Clone();

  uint32_t count = grid.GridReduce(minPerFourCell, maxPerFourCell,
    step, totalDesiredFeatures);

  if (count != totalDesiredFeatures) {
    // ignore odd row/column
    for (size_t y = 0; y < (grid.vBuckets & (~1)); y += 1) {
      for (size_t x = 0; x < (grid.hBuckets & (~1)); x += 1) {
        EXPECT_LE(grid.Row(y)[x].count, minPerFourCell);
      }
    }
  }

  uint32_t referenceCount = 0;
  for (size_t y = 0; y < grid.vBuckets; y += 2) {
    for (size_t x = 0; x < grid.hBuckets; x += 2) {
      std::vector<uint32_t> features;
      std::vector<uint32_t> original;

      features.insert(features.end(), &grid.Row(y)[x][0],
          &grid.Row(y)[x][grid.Row(y)[x].count]);
      features.insert(features.end(), &grid.Row(y+1)[x][0],
          &grid.Row(y+1)[x][grid.Row(y+1)[x].count]);
      features.insert(features.end(), &grid.Row(y)[x+1][0],
          &grid.Row(y)[x+1][grid.Row(y)[x+1].count]);
      features.insert(features.end(), &grid.Row(y+1)[x+1][0],
          &grid.Row(y+1)[x+1][grid.Row(y+1)[x+1].count]);

      original.insert(original.end(), &referenceGrid.Row(y)[x][0],
          &referenceGrid.Row(y)[x][referenceGrid.Row(y)[x].count]);
      original.insert(original.end(), &referenceGrid.Row(y+1)[x][0],
          &referenceGrid.Row(y+1)[x][referenceGrid.Row(y+1)[x].count]);
      original.insert(original.end(), &referenceGrid.Row(y)[x+1][0],
          &referenceGrid.Row(y)[x+1][referenceGrid.Row(y)[x+1].count]);
      original.insert(original.end(), &referenceGrid.Row(y+1)[x+1][0],
          &referenceGrid.Row(y+1)[x+1][referenceGrid.Row(y+1)[x+1].count]);

      ASSERT_LE(features.size(), original.size());

      std::sort(features.begin(), features.end());
      std::sort(original.begin(), original.end());

      // assert that the kept features are the best features
      for (size_t i = 0; i < features.size(); i += 1) {
        EXPECT_EQ(features.rbegin()[i], original.rbegin()[i]);
      }

      referenceCount += features.size();
    }
  }

  // add ignored edges to count
  if (grid.vBuckets % 2) {
    for (size_t x = 0; x < grid.hBuckets; x += 1) {
      referenceCount += grid.Row(grid.vBuckets-1)[x].count;
    }
  }
  if (grid.hBuckets % 2) {
    for (size_t y = 0; y < grid.vBuckets; y += 1) {
      referenceCount += grid.Row(y)[grid.hBuckets-1].count;
    }
  }
  if (grid.vBuckets % 2 && grid.hBuckets % 2) {
    referenceCount -= grid.Row(grid.vBuckets-1)[grid.hBuckets-1].count;
  }

  EXPECT_EQ(count, referenceCount);
}

TEST(FeatureGridTest, GetFeaturesInArea) {
  constexpr int bucketLimit = 5;
  constexpr int logBucketSize = 4;
  constexpr int border = 16;
  constexpr int bucketSize =
    pislam::FeatureGrid<bucketLimit, logBucketSize, border>::bucketSize;

  const int numFeatures = 1000;
  const int numTests = 10000;

  pislam::FeatureGrid<bucketLimit, logBucketSize, border> grid(640, 480);

  std::mt19937_64 rng;
  for (int i = 0; i < numFeatures; i += 1) {
    uint32_t x = rng() % 608 + border;
    uint32_t y = rng() % 448 + border;

    uint32_t feature = pislam::encodeFast(0, x, y);

    x = (x - border) / bucketSize;
    y = (y - border) / bucketSize;

    pislam::FeatureBucket<bucketLimit> &bucket = grid.Row(y)[x];
    if (bucket.count < bucketLimit) {
      bucket[bucket.count] = feature;
      bucket.count += 1;
    } else {
      i -= 1;
    }
  }

  std::vector<uint32_t> features;
  grid.ExtractAndIndex(features);

  for (int test = 0; test < numTests; test += 1) {
    std::vector<uint32_t> indices;
    std::vector<uint32_t> reference;

    int x = rng() % 640;
    int y = rng() % 480;

    // bias tests towards small radii
    int r = (rng() % 8) * (rng() % 8) * (rng() % 8);

    int x0 = x-r;
    int y0 = y-r;
    int x1 = x+r;
    int y1 = y+r;

    int index = 0;
    for (uint32_t feature : features) {
      int fx = pislam::decodeFastX(feature);
      int fy = pislam::decodeFastY(feature);

      if ((x0 <= fx && fx <= x1) && (y0 <= fy && fy <= y1)) {
        reference.push_back(index);
      }

      index += 1;
    }

    grid.GetFeaturesInArea(x, y, r, indices);

    std::sort(indices.begin(), indices.end());
    std::sort(reference.begin(), reference.end());

    for (size_t i = 0; i < indices.size(); i += 1) {
      ASSERT_EQ(indices[i], reference[i]);
    }
  }
}

TEST(FeatureGridTest, FindEmptySlices) {
  constexpr int bucketLimit = 5;
  constexpr int logBucketSize = 4;
  constexpr int border = 16;
  constexpr int bucketSize =
    pislam::FeatureGrid<bucketLimit, logBucketSize, border>::bucketSize;

  const int numFeatures = 16;

  pislam::FeatureGrid<bucketLimit, logBucketSize, border> grid(640, 480);

  std::mt19937_64 rng;
  for (int i = 0; i < numFeatures; i += 1) {
    uint32_t x = rng() % 608 + 16;
    uint32_t y = rng() % 448 + 16;

    uint32_t feature = pislam::encodeFast(0, x, y);

    x = (x - border) / bucketSize;
    y = (y - border) / bucketSize;

    pislam::FeatureBucket<bucketLimit> &bucket = grid.Row(y)[x];
    if (bucket.count < bucketLimit) {
      bucket[bucket.count] = feature;
      bucket.count += 1;
    } else {
      i -= 1;
    }
  }

  std::vector<pislam::FeatureGrid<bucketLimit, logBucketSize, border>> slices;

  slices = grid.FindEmptySlices(3);

  ASSERT_NE(slices.size(), 0);

  size_t notEmptyLeft = 0;
  for (const auto &slice : slices) {
    size_t notEmptyWidth = slice.hOffset - notEmptyLeft;
    if (notEmptyWidth >= 3) {
      size_t count = 0;
      for (size_t y = 0; y < grid.vBuckets; y += 1) {
        for (size_t x = notEmptyLeft; x < slice.hOffset; x += 1) {
          count += grid.Row(y)[x].count;
        }
      }
      EXPECT_NE(count, 0);
    } else {
    }
    size_t count = 0;
    for (size_t y = 0; y < grid.vBuckets; y += 1) {
      for (size_t x = slice.hOffset; x < slice.hOffset + slice.hBuckets; x += 1) {
        count += grid.Row(y)[x].count;
      }
    }
    EXPECT_EQ(count, 0);
  }
  
  const auto &slice = slices.back();
  size_t notEmptyWidth = slice.hOffset - notEmptyLeft;
  if (notEmptyWidth >= 3) {
    size_t count = 0;
    for (size_t y = 0; y < grid.vBuckets; y += 1) {
      for (size_t x = notEmptyLeft; x < slice.hOffset; x += 1) {
        count += grid.Row(y)[x].count;
      }
    }
    EXPECT_NE(count, 0);
  }
}

} /* namespace */
