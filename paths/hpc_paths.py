import os
import pathlib

data_path = pathlib.Path("/home/haoyu/work/code/medical/SEGA_MW_2023/data")

### Data Paths ###
sega_path = data_path / "SEGA"

### RAW Data Paths ###
raw_sega_path = sega_path / "RAW"

### Parsed Data Paths ###
parsed_sega_path = sega_path / "PARSED"

### Training Paths ###
project_path = pathlib.Path("/home/haoyu/work/code/medical/SEGA_MW_2023/project")
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"