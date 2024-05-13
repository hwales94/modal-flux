import json
import os
import pathlib
import subprocess
import time
import urllib.request
import uuid

import httpx
import websocket
from tqdm import tqdm

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def download_to_comfyui(url, path):
    model_directory = "/root/" + path
    local_filename = url.split("/")[-1]
    local_filepath = pathlib.Path(model_directory, local_filename)
    local_filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {url} ... to {model_directory}")

    if url.endswith(".git"):
        download_custom_node(url, model_directory)

    else:
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


def download_custom_node(url, path):
    subprocess.run(["git", "clone", url], cwd=path)

    # Pip install requirements.txt if it exists in the custom node
    repo_name = url.split("/")[-1].split(".")[0]
    repo_path = f"{path}/{repo_name}"
    if os.path.isfile(f"{repo_path}/requirements.txt"):
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"], cwd=repo_path
        )


def connect_to_local_server():
    ws = websocket.WebSocket()
    while True:
        try:
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            print("Connection established!")
            break
        except ConnectionRefusedError:
            print("Server still standing up...")
            time.sleep(1)
    return ws


# ComfyUI specific helpers, adpated from: https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example_ws_images.py
def queue_prompt(workflow_json):
    # confusingly here, "prompt" in that code actually refers to the workflow_json, so just renaming it here for clarity
    p = {"prompt": workflow_json, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    resp = json.loads(urllib.request.urlopen(req).read())
    print(f"Queued workflow {resp['prompt_id']}")
    return json.loads(urllib.request.urlopen(req).read())


def get_images(ws, workflow_json):
    prompt_id = queue_prompt(workflow_json)["prompt_id"]
    output_images = []
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            print(message)
            if message["type"] == "executing":
                data = message["data"]
                if data["prompt_id"] == prompt_id:
                    if data["node"] is None:
                        break  # Execution is done
                    else:
                        current_node = data["node"]
        else:
            if workflow_json.get(current_node):
                if (
                    workflow_json.get(current_node).get("class_type")
                    == "SaveImageWebsocket"
                ):
                    output_images.append(
                        out[8:]
                    )  # parse out header of the image byte string

    return output_images
