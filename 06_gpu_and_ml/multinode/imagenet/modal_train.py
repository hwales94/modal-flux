import os
import modal
import training.config as config

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "unzip", "vim", "git", "htop")
    .pip_install("torch", "torchvision", "nvidia-dali-cuda120")
    # needed until they update pypi
    .pip_install("git+https://github.com/thecodingwizard/webdataset.git")
    .pip_install("wandb")
    .pip_install("tqdm")
)
app = modal.App(
    f"resnet-{config.run_name}",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/data": modal.Volume.from_name("imagenet"),
    },
)


@app.function(
    gpu=modal.gpu.H100(count=config.gpus_per_node),
    timeout=60 * 60 * 24,
    cloud="oci",
    # The CPU reservation here was important for me. Otherwise the dataloader could be slow.
    cpu=min(10 * config.gpus_per_node + 4, 64),
    mounts=[modal.Mount.from_local_dir("training", remote_path="/root/training")],
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
@modal.experimental.grouped(size=config.nodes)
def train_resnet():
    from torch.distributed.run import parse_args, run

    if config.nodes > 1:
        world_size = int(os.environ.get("MODAL_WORLD_SIZE", 1))
        container_rank = os.environ["MODAL_CONTAINER_RANK"]
        main_addr = os.environ["MODAL_MAIN_I6PN"]
        assert config.nodes == world_size

    run(
        parse_args(
            (
                ["--standalone"]
                if config.nodes == 1
                else [
                    f"--node_rank={container_rank}",
                    f"--master_addr={main_addr}",
                ]
            )
            + [
                f"--nnodes={config.nodes}",
                f"--nproc-per-node={config.gpus_per_node}",
                "--master_port=1234",
                "training/train.py",
            ]
        )
    )


@app.local_entrypoint()
def main():
    if config.runtime == "runc":
        assert os.environ["MODAL_FUNCTION_RUNTIME"] == "runc"
    else:
        assert "MODAL_FUNCTION_RUNTIME" not in os.environ
    train_resnet.remote()
