import modal

stub = modal.Stub(image=modal.Image.debian_slim().pip_install(["123abc"]))


@stub.function
def run():
    pass


if __name__ == "__main__":
    try:
        with stub.run():
            run()
    except modal.exception.RemoteError:
        pass
