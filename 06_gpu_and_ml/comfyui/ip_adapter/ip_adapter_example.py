import subprocess

import modal
from comfyui.comfy_base_image import image

image = (  # build up a Modal Image to run ComfyUI, step by step
    image.run_commands(  # download the WAS Node Suite custom node pack
        "comfy node install ComfyUI_IPAdapter_plus"
    )
    .run_commands("apt install -y wget")
    .run_commands(  # the Unified Model Loader node requires these two models to be named a specific way, so we use wget instead of the usual comfy model download command
        "wget -q -O /root/comfy/ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    )
    .run_commands(
        "wget -q -O /root/comfy/ComfyUI/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors, https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
    )
    .run_commands(  # download the IP-Adapter model
        "comfy --skip-prompt model download --url https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors --relative-path models/ipadapter"
    )
    .run_commands(  # download the Civitai model
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/128713 --relative-path models/checkpoints --set-civitai-api-token $CIVIT_AI_TOKEN",
        secrets=[modal.Secret.from_name("civitai-token")],
    )
)

app = modal.App(name="example-ip-adapter", image=image)


# Run ComfyUI as an interactive web server
@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A10G",
)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
