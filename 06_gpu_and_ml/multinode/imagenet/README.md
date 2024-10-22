# Training Resnet50

- [wandb dashboard](https://wandb.ai/nathan-modal-labs/resnet50-training?nw=nwusernathanmodal)
- [wandb benchmarks dashboard](https://wandb.ai/nathan-modal-labs/resnet50-training-benchmark)

## Downloading Data

You can experiment with `convert_to_webdataset.ipynb` locally to figure out how you want to download the data.

The download script I used was `modal run download.py`.

## Training

1. Modify `training/config.py` as needed.
2. Run `modal run -d modal_train.py`.
