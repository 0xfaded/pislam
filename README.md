PiSlam
=====

Real-time feature extraction on the Raspberry Pi and other ARM processors supporting NEON.

License
---
As of August 2022, PiSlam is licensed under the MIT License. If you find this project useful, I would appreciate an email to carlchatfield@gmail.com,

Thank you :)

Design Goals
---

PiSlam aims to extract ORB features with the following metrics:

 * Extract and process 1000 ORB descriptors per frame at 30fps.
 * Operate on an VGA 640x480 image stream.
 * Use an 8 level, 1.2 scale reduction pyramid. Approx 750k total pixels.
 * Use 256 bits per descriptor.

Currently, the released code is only the SLAM frontend, i.e. the ORB extraction code.
The performance section shows that these above targets have been achieved.

Usage
---

Images should be prepared by applying a Gaussian blur and externally computing
the image pyramid. The Raspberry Pi GPU is better suited for this task and the
code is therefore at this time not included in this release. A 5x5 or 7x7 kernel
works well.

This code extracts FAST points from a single level of the pyramid.

```
  // 2210 is the pyramid height
  uint8_t img[2210][640] = ...; // load pyramid from elsehere
  uint8_t out[2210][640] = {0};

  std::vector<uint32_t> keypoints;
  std::vector<uint32_t> descriptors;

  pislam::fastDetect<640, 16>(width, height, &img[y], &out[y], 20);
  pislam::fastScoreHarris<640, 16>(width, height, &img[y], 1 << 15, &out[y]);
  pislam::fastExtract<640, 16, 4, 3>(width, height, &out[y], keypoints);
  pislam::orbCompute<640, 8>(img, keypoints, descriptors);
```

Each of the above functions is well documented in the source code. Template parameters have
been used for `vstep` and `border` width, allowing gcc to use constant
offsets. Because ARM instructions permit only immediate relative addresses between
-4096 to 4095, try to keep vstep small. For context, if gcc cannot use
constant offsets for the ORB descriptor pattern, the code size will double.
Performance will more than half due to bad instruction caching.

To compute the whole pyramid, the above will need to be executed in a loop.
For example, the below code works if the pyramid is vertically stacked.

```
  // 2210 is the pyramid height
  uint8_t img[2210][640] = ...; // load pyramid from elsehere
  uint8_t out[2210][640] = {0};

  std::vector<uint32_t> keypoints;
  std::vector<uint32_t> descriptors;

  int y = 0;
  for (int level = 0; level < 8; level += 1) {
    int oldSize = keypoints.size(); // Iterator to correct y offsets

    int width =  pyramidLevels[level].width;
    int height = pyramidLevels[level].height;

    pislam::fastDetect<640, 16>(width, height, &img[y], &out[y], 20);
    pislam::fastScoreHarris<640, 16>(width, height, &img[y], 1 << 15, &out[y]);
    pislam::fastExtract<640, 16, 4, 3>(width, height, &out[y], keypoints);

    for (auto it = result.begin() + oldSize; it < result.end(); ++it) (*it) += y;

    y += height;
  }
  pislam::orbCompute<640, 8>(img, keypoints, descriptors);
```

Performance
---

Four separate steps comprise the ORB extraction process.

 1. FAST keypoint detection
 2. Harris score filtering
 3. Non-max suppression
 4. ORB computation

Unlike other FAST implementations which try to detect non-features and abort quickly,
PiSlam uses a pure SIMD implementation. Therefore the running time is constant and
easy to quantify at 8.4ms on a Raspberry Pi 3, making it the most expensive stage.

All other stages are dependent on the number of points detected, but empirical
results show that the frontend can comfortably handle up to 2000 ORB features
on a single core.

Performance compromises include the following:

 * Discrete ORB angles at 12 degree intervals, where OpenCV is exact.
 * Harris score computed from 6x6 patch rather than 7x7.
 * Approximate vectorized atan2 function, average error less than half a degree.

The charts below were produced using 200 frames from Sample 3 of the New College SLAM data sets.
<http://www.robots.ox.ac.uk/NewCollegeData/index.php?n=Main.Downloads>. 
Images are first scaled up to VGA format, and then blurred using a 5x5 kernel.
The pyramids are then computed and used as input to the test program.

![Frame Execution Time](doc/frame_times.png?raw=true "Frame Execution Time")
![Stage Execution Time](doc/stage_times.png?raw=true "Stage Execution Time")

For comparison, to extract 1000 ORB points the OpenCV detection implementation
(everything before the ORB descriptors) requires 50-60ms.
The ORB descriptor implementation has a large overhead,
since OpenCV will copy the image and blur it using 7x7 kernel.
PiSlam skips this step since input images are assumed to be blurred.
After subtracting the constant overhead, OpenCV requires 10ms to compute the 1000 descriptors.
Therefore our implementation is 3-4 times faster depending on how one measures.

Further, other tests (not included in this release) have shown that feature matching
can be achieved in under 20 ms per frame using [flann](https://github.com/mariusmuja/flann).
This is promising, but bit counts are computed using lookup tables leaving room for
improvement.

Demo
---

The below image was produced by computing the ORB features in four keyframes, and
then tracing the paths of best matches within a 20 pixel radius.
Images used are the same sequence that generated the performance graphs.

[![Demo](doc/out005.png?raw=true)](https://www.youtube.com/watch?v=zv2aqzKjivA "ORB feature extraction demo")

Note that foreground points are lost because the demo does not compensate for scale changes.

Tests
---

Due to the haphazard development cycle of my side projects, the tests and benchmarks
are not yet organized. They are provided only as reference for how to use the library.

Sorry.
