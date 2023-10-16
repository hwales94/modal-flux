# ---
# lambda-test: false
# ---

import modal
from modal import wsgi_app

stub = modal.Stub(
    "example-web-flask-stream",
    image=modal.Image.debian_slim().pip_install("flask"),
)


@stub.function()
def generate_rows():
    """
    This creates a large CSV file, about 10MB, which will be streaming downloaded
    by a web client.
    """
    for i in range(10_000):
        line = ",".join(str((j + i) * i) for j in range(128))
        yield f"{line}\n"


@stub.function()
@wsgi_app()
def flask_app():
    from flask import Flask

    web_app = Flask(__name__)

    # These web handlers follow the example from
    # https://flask.palletsprojects.com/en/2.2.x/patterns/streaming/

    @web_app.route("/")
    def generate_large_csv():
        # Run the function locally in the web app's container.
        return generate_rows.local(), {"Content-Type": "text/csv"}

    @web_app.route("/remote")
    def generate_large_csv_in_container():
        # Run the function remotely in a separate container,
        # which will stream back results to the web app container,
        # which will stream back to the web client.
        #
        # This is less efficient, but demonstrates how web serving
        # containers can be separated from and cooperate with other
        # containers.
        return generate_rows.remote(), {"Content-Type": "text/csv"}

    return web_app
