# Correlate the stage position to the points in a low magnification SEM image. Therefore, by defining a pixel position on the SEM image, the required stage x and y position to bring the point under the electron beam will be calculated.

import sys

sys.path.append(
    r"C:\Users\Zhou Xu\OneDrive - Monash University\2018-2019 MCEM\02 Projects\13 EBSD n ECCI\20230213 tutorial manuscript\supplementary\openECCI_master"
)

from pathlib import Path
from openECCI import stagecomputation

# Load the SEM image
data_path = Path(
    r"C:\Users\Zhou Xu\OneDrive - Monash University\2018-2019 MCEM\02 Projects\13 EBSD n ECCI\20230213 tutorial manuscript\supplementary\openECCI-data\data\fcc_fe"
)
data_file = data_path / r"01_steel overview.tif"

# pixel position on the image
x = 100
y = 100

[stage_x, stage_y] = stagecomputation.pixel_pos_to_stage_coord(
    data_file,
    pixel_x=x,
    pixel_y=y,
    stage_mode="absolute",
)

print(f"Stage position: x {stage_x*1e3:.5f}mm, y {stage_y*1e3:.5f}mm")
