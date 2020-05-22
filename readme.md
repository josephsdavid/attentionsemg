# sEMG attention

Repository for the paper *insert title here*

## Reproduction 

We provide three methods to reproduce the environment used in this paper:

### Nix

To reproduce down to CUDA version and compiler level, please use the reproducible package manager [nix](https://nixos.org/nix/). The environment is available as a distributable binary using [cachix](https://cachix.org/). To pull the binaries to your system, install cachix, and run 

```bash
cachix use tf-envs
```

Then, regardless of cachix usage (if you are not using cachix, you will have to wait for tensorflow to compile to the specifications used in this project), run

```bash
nix-shell
```

And the environment will be reproduced!

### Pip

To install the requirements for this project, please run

```
pip install -r requirements.txt
```

### Docker

Carson will make this!

## Get the data

From the home directory of this project:

```bash
cd data
# always inspect scripts before downloading!
cat load_nina.sh
./load_nina.sh
```

Alternatively, the data is available for download (after the creation of a login) [here](ninapro.hevs.ch/DB5_DoubleMyo) and [here](https://zenodo.org/record/1000116#.XhkdcCVMHDs).


## Using the code

For the sake of thoroughness, we will demonstrate how to load the data and generate the training, validation, and test datasets used in this project:

```python
import numpy as np
from data import dataset, ma_batch
from generator import generator

# load the data, takes about 12 GB in memory
data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
# split by exercise repetition
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

# create generators, these load the data, in the case of train, augment the data
# and lazily apply the moving axis. To include IMU data, add `imu = True`, and 
# to not do the moving average transformation described in the paper, add `ma = False`
train = generator(data, list(train_reps))
validation = generator(data, list(val_reps), augment=False)
test = generator(data, [test_reps][0], augment=False)

# if you want the full data as a numpy array, it is stored in the X and y attributes
test_x = np.moveaxis(ma_batch(test.X, test.ma_len), -1, 0)
test_y = test.y

# get all the data:
from typing import Iterable
def get_arrays(g: generator) -> Iterable[np.ndarray]:
	return np.moveaxis(ma_batch(g.X, g.ma_len), -1, 0), g.y

(train_x, train_y), (val_x, val_y) = (get_arrays(g) for g in [train, validation])
```

To reproduce the models described in the paper:

blah

