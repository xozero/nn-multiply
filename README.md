# nn-multiply

This repository contains code for training neural networks to predict the multiplication of two numbers.

To run the experiments yourself, use `main.py`:

```
usage: main.py [-h] [--gendata] [--train] [-c CFG_ID] [-n [NRUNS]]

optional arguments:
  -h, --help            show this help message and exit
  --gendata
  --train
  -c CFG_ID, --cfg_id CFG_ID
                        cfg_id
  -n [NRUNS], --nruns [NRUNS]
```

To configure experiments, edit `simulations.json`. The `CFG_ID` in the above command usage is the key in the config. To find all keys in the config, run `python3 config.py`.

Example commands for generating data for configuration `7_high2_2000` and for training a model afterwards:

```
python3 main.py --gendata -c 7_high2_2000
python3 main.py --train -c 7_high2_2000
```

The generated data for all experiments configured in the checked in `simulations.json` can be found in `datasets`, it is all checked in. Thus, the `--gendata` step can be omitted for those.

A blog post with details can be found [here](https://blog.xa0.de/post/Multiplying-large-numbers-with-Neural-Networks/).
