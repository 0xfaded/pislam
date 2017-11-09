#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "../include/Fast.h"
#include "TestUtil.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

class FeatureGridTest: public ::testing::TestWithParam<::std::tuple<int, int, int, int>> {};

TEST(FeatureGridTest, GridReduce) {
  constexpr int bucketLimit = 5;

  pislam::FeatureGrid<bucketLimit, 16, 16> grid(640, 480);

  const int maxPerFourCell = 8;
  const int minPerFourCell = 4;
  const int step = 2;
  const int totalDesiredFeatures = 1000;

  std::mt19937_64 rng;
  for (size_t i = 0; i < grid.numBuckets; i += 1) {
    pislam::FeatureBucket<bucketLimit> &bucket = grid.buckets[i];
    bucket.count = rng() % (bucketLimit+1);
    for (uint32_t j = 0; j < bucket.count; j += 1) {
      bucket[j] = rng();
    }
    std::sort(&bucket[0], &bucket[bucket.count]);
  }

  pislam::FeatureGrid<bucketLimit, 16, 16> referenceGrid(grid);

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
} /* namespace */
