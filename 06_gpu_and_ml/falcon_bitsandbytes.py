# ---
# integration-test: false
# ---
# # Run Falcon-40B with bitsandbytes

# In this example, we download the full-precision weights of the Falcon-40B LLM but load it in 4-bit using
# Tim Dettmer's [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library. This enables it to fit
# into a single GPU (A100 40GB).
#
# Due to the current limitations of the library, the inference speed is a little over 2 token/second and due
# to the sheer size of the model, the cold start time on Modal is around 2 minutes.
#
# For faster cold start at the expense of inference, check out
# [Running Falcon-40B with AutoGPTQ](/docs/guide/ex/falcon_gptq).
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import Image, Stub, gpu, method, web_endpoint

# Spec for an image where falcon-40b-instruct is cached locally
def download_falcon_40b():
    from huggingface_hub import snapshot_download

    model_name = "tiiuae/falcon-40b-instruct"
    snapshot_download(model_name)


image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "peft @ git+https://github.com/huggingface/peft.git",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "accelerate @ git+https://github.com/huggingface/accelerate.git",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
    )
    .run_function(download_falcon_40b)
)

stub = Stub(image=image, name="example-falcon-bnb")

# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](/docs/guide/lifecycle-functions) and the __enter__` method.
#
# Within the [@stub.cls](/docs/reference/modal.Stub#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](/pricing). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# We load the model in 4-bit using the `bitsandbytes` library.
#
# The rest is just using the [pipeline()](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# abstraction from the `transformers` library. Refer to the documentation for more parameters and tuning.
@stub.cls(
    gpu=gpu.A100(), # Use A100s
    timeout=60 * 10, # 10 minute timeout on inputs
    container_idle_timeout=60 * 5, # Keep runner alive for 5 minutes
)
class Falcon40B_4bit:
    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

        model_name = "tiiuae/falcon-40b-instruct"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True, # Model is downloaded to cache dir
            device_map="auto",
            quantization_config=nf4_config,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True, device_map="auto"
        )
        tokenizer.bos_token_id = 1

        self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @method()
    def generate(self, prompt: str):
        from threading import Thread
        from transformers import TextIteratorStreamer
        from transformers import GenerationConfig

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        input_ids = input_ids.to(self.model.device)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            attention_mask=tokenized.attention_mask,
            output_scores=True,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        for new_text in streamer:
            print(new_text, end="")
            yield new_text
        
        thread.join()


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_bitsandbytes.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-bnb-get.modal.run/?question=How%20do%20planes%20work?).
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)

@stub.function(timeout=60 * 10)
@web_endpoint()
def get(question: str):
    from fastapi.responses import StreamingResponse
    from itertools import chain

    model = Falcon40B_4bit()
    return StreamingResponse(
        chain(
            ("Loading model (100GB). This usually takes around 110s ...\n\n"),
            model.generate.call(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )
