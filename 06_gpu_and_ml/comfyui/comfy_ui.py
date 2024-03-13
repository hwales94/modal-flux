# ---
# lambda-test: false
# ---
#
# # Run ComfyUI
#
# This example shows you how to run a ComfyUI workspace with `modal serve`.
#
# If you're unfamiliar with how ComfyUI works we recommend going through Scott Detweiler's
# [tutorials on Youtube](https://www.youtube.com/watch?v=AbB33AxrcZo).
#
# ![example comfyui workspace](./comfyui-hero.png)

import pathlib

import modal

# ## Define container image
#
# Fun with ComfyUI begins with pre-trained model checkpoints.
# The checkpoint downloaded below is [huggingface.co/dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0), but others can be used.
# The ComfyUI repository has other recommendations listed in this file:
# [notebooks/comfyui_colab.ipynb](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb).
#
# This download function is run as the final image building step, and takes around 10 seconds to download
# the ~2.0 GiB model checkpoint.


def download_checkpoints():
    import httpx
    from tqdm import tqdm

    with open("/checkpoints.txt") as f:
        checkpoints = [line.strip() for line in f.readlines()]

    for url in checkpoints:
        checkpoints_directory = "/root/models/checkpoints"
        local_filename = url.split("/")[-1]
        local_filepath = pathlib.Path(checkpoints_directory, local_filename)
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        print(f"downloading {url} ...")
        with httpx.stream("GET", url, follow_redirects=True) as stream:
            total = int(stream.headers["Content-Length"])
            with open(local_filepath, "wb") as f, tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = stream.num_bytes_downloaded
                for data in stream.iter_bytes():
                    f.write(data)
                    progress.update(
                        stream.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = stream.num_bytes_downloaded


def download_plugins():
    import subprocess
    import json

    with open("/plugins.json") as f:
        plugins = json.load(f)
        for plugin in plugins:
            url = plugin["url"]
            name = url.split("/")[-1]
            command = f"cd /root/custom_nodes && git clone {url}"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Repository {url} cloned successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning repository: {e.stderr}")
            if plugin.get("requirements"):
                pip_command = f"cd /root/custom_nodes/{name} && pip install -r {plugin['requirements']}"
            try:
                subprocess.run(pip_command, shell=True, check=True)
                print(f"Requirements for {url} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error installing requirements: {e.stderr}")


# Pin to a specific commit from https://github.com/comfyanonymous/ComfyUI/commits/master/
# for stability. To update to a later ComfyUI version, change this commit identifier.
comfyui_commit_sha = "a38b9b3ac152fb5679dad03813a93c09e0a4d15e"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Here we place the latest ComfyUI repository code into /root.
    # Because /root is almost empty, but not entirely empty
    # as it contains this comfy_ui.py script, `git clone` won't work.
    # As a workaround we `init` inside the non-empty directory, then `checkout`.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
    )
    # Use fork of https://github.com/valohai/asgiproxy with bugfixes.
    .pip_install(
        "git+https://github.com/modal-labs/asgiproxy.git@ef25fe52cf226f9a635e87616e7c049e451e2bd8",
        "httpx",
        "tqdm",
    )
    .copy_local_file("checkpoints.txt")
    .run_function(download_checkpoints)
    .copy_local_file("plugins.json")
    .run_function(download_plugins)
)
stub = modal.Stub(name="example-comfy-ui", image=image)

# ## Spawning ComfyUI in the background
#
# Inside the container, we will run the ComfyUI server and execution queue in a background subprocess using
# `subprocess.Popen`. Here we define `spawn_comfyui_in_background()` to do this and then poll until the server
# is ready to accept connections.

HOST = "127.0.0.1"
PORT = "8188"


def spawn_comfyui_in_background():
    import socket
    import subprocess

    process = subprocess.Popen(
        [
            "python",
            "main.py",
            "--dont-print-server",
            "--multi-user",
            "--port",
            PORT,
        ]
    )

    # Poll until webserver accepts connections before running inputs.
    while True:
        try:
            socket.create_connection((HOST, int(PORT)), timeout=1).close()
            print("ComfyUI webserver ready!")
            break
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"comfyui main.py exited unexpectedly with code {retcode}"
                )


# ## Wrap it in an ASGI app
#
# Finally, Modal can only serve apps that speak the [ASGI](https://modal.com/docs/guide/webhooks#asgi) or
# [WSGI](https://modal.com/docs/guide/webhooks#wsgi) protocols. Since the ComfyUI server uses `aiohttp`,
# which [does not support either](https://github.com/aio-libs/aiohttp/issues/2902), we run a separate ASGI
# app using the `asgiproxy` package that proxies requests to the ComfyUI server.


@stub.function(
    gpu="any",
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Restrict to 1 container because we want to our ComfyUI session state
    # to be on a single container.
    concurrency_limit=1,
    # Extend idle timeout of container so that you can switch tabs, read a
    # quick article, and come back to resume work on same container.
    container_idle_timeout=10 * 5,
    # Timeout requests after 60s
    timeout=60,
)
@modal.asgi_app()
def web():
    from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig
    from asgiproxy.context import ProxyContext
    from asgiproxy.simple_proxy import make_simple_proxy_app

    spawn_comfyui_in_background()

    config = type(
        "Config",
        (BaseURLProxyConfigMixin, ProxyConfig),
        {
            "upstream_base_url": f"http://{HOST}:{PORT}",
            "rewrite_host_header": f"{HOST}:{PORT}",
        },
    )()
    return make_simple_proxy_app(ProxyContext(config))
