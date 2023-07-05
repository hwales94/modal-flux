# ---
# args: ["--message", "what's up?"]
# ---
"""Single-page application that lets you talk to a transformer chatbot.

This is a complex example demonstrating an end-to-end web application backed by
serverless web handlers and GPUs. The user visits a single-page application,
written using Solid.js. This interface makes API requests that are handled by a
Modal function running on the GPU.

The weights of the model are cached in a shared volume, so they don't need to be
downloaded again as long as the app is running.

Chat history tensors are saved in a `modal.Dict` distributed dictionary.
"""

import uuid
from pathlib import Path
from typing import Optional, Tuple

import fastapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from modal import Dict, Image, Mount, NetworkFileSystem, Stub, asgi_app

assets_path = Path(__file__).parent / "chatbot_spa"
stub = Stub("example-web-spa")

stub.cache = NetworkFileSystem.new()
stub.chat_histories = Dict.new()

gpu_image = Image.debian_slim()
gpu_image = gpu_image.pip_install(
    "torch", find_links="https://download.pytorch.org/whl/cu116"
)
gpu_image = gpu_image.pip_install("transformers")
stub.gpu_image = gpu_image

if stub.is_inside(stub.gpu_image):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-large", cache_dir="/cache"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-large", cache_dir="/cache"
    )


@stub.function(
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")]
)
@asgi_app()
def transformer():
    app = fastapi.FastAPI()

    @app.post("/chat")
    def chat(body: dict = fastapi.Body(...)):
        message = body["message"]
        chat_id = body.get("id")
        id, response = generate_response.call(message, chat_id)
        return JSONResponse({"id": id, "response": response})

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


@stub.function(
    gpu="any", image=stub.gpu_image, network_file_systems={"/cache": stub.cache}
)
def generate_response(
    message: str, id: Optional[str] = None
) -> Tuple[str, str]:
    chat_histories = stub.app.chat_histories  # Load the queue object.

    new_input_ids = tokenizer.encode(
        message + tokenizer.eos_token, return_tensors="pt"
    )
    if id is not None:
        chat_history = chat_histories[id]
        bot_input_ids = torch.cat([chat_history, new_input_ids], dim=-1)
    else:
        id = str(uuid.uuid4())
        bot_input_ids = new_input_ids

    chat_history = model.generate(
        bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(
        chat_history[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    chat_histories[id] = chat_history
    return id, response


@stub.local_entrypoint()
def test_response(message: str):
    _, response = generate_response.call(message)
    print(response)
