import modal

app = modal.App(
    "example-generators-async"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
async def run_async():
    async for r in f.remote_gen.aio(10):
        print(r)
