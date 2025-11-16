"""
Speed Calculation Configuration

To calibrate for your video:
1. Pause video at a frame showing the road clearly
2. Identify 4 points forming a quadrilateral on the road
3. Measure the actual dimensions of that area
4. Update SOURCE and TARGET values below
"""

import numpy as np

# SOURCE: 4 points in image coordinates [x, y]
# These should form a quadrilateral on the road surface
# Default values - MUST BE CALIBRATED for your specific video!
SOURCE = np.array([
    [1252, 787],    # Top-left corner
    [2298, 803],    # Top-right corner
    [5039, 2159],   # Bottom-right corner
    [-550, 2159]    # Bottom-left corner
])

# TARGET: Real-world dimensions in meters
TARGET_WIDTH = 25    # Width of the road area (meters)
TARGET_HEIGHT = 250  # Length of the road area (meters)

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH-1, 0],
    [TARGET_WIDTH-1, TARGET_HEIGHT-1],
    [0, TARGET_HEIGHT-1]
])
