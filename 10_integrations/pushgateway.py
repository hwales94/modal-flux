# ---
# cmd: ["modal", "deploy", "10_integrations/pushgateway.py"]
# ---
# # Publish custom metrics with Prometheus Pushgateway
#
# This example shows how to publish custom metrics to a Prometheus instance with Modal.
# Due to a Modal container's ephemeral nature, it's not a good fit for a traditional
# scraping-based Prometheus setup. Instead, we'll use a [Prometheus Pushgateway](https://github.com/prometheus/pushgateway)
# to collect and store metrics from our Modal container. We can run the Pushgateway in Modal
# as a separate process and have our application push metrics to it.
#
# ![Prometheus Pushgateway diagram](./pushgateway_diagram.png)
#
# ## Install Prometheus Pushgateway
#
# Since the official Prometheus pushgateway image does not have Python installed, we'll
# use a custom image that includes Python to push metrics to the Pushgateway. Pushgateway
# ships a single binary, so it's easy to get it into a Modal container.

import os
import subprocess

import modal
from modal import web_endpoint

PUSHGATEWAY_VERSION = "1.9.0"

gw_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "tar")
    .run_commands(
        f"wget https://github.com/prometheus/pushgateway/releases/download/v{PUSHGATEWAY_VERSION}/pushgateway-{PUSHGATEWAY_VERSION}.linux-amd64.tar.gz",
        f"tar xvfz pushgateway-{PUSHGATEWAY_VERSION}.linux-amd64.tar.gz",
        f"cp pushgateway-{PUSHGATEWAY_VERSION}.linux-amd64/pushgateway /usr/local/bin/",
        f"rm -rf pushgateway-{PUSHGATEWAY_VERSION}.linux-amd64 pushgateway-{PUSHGATEWAY_VERSION}.linux-amd64.tar.gz",
        "mkdir /pushgateway",
    )
)

# ## Start the Pushgateway
#
# We'll start the Pushgateway as a separate Modal app. This way, we can run the Pushgateway
# in the background and have our main app push metrics to it. We'll use the `web_server`
# decorator to expose the Pushgateway's web interface. Note that we must set `concurrency_limit=1`
# as the Pushgateway is a single-process application. If we spin up multiple instances, they'll
# conflict with each other.
#
# This is an example configuration, but a production-ready configuration will differ in two respects:
# 1. You should set up authentication for the Pushgateway. Modal has built-in support
#    for this with [our authentication helpers](https://modal.com/docs/guide/webhooks#authentication).
# 2. The Pushgateway should listen on a [custom domain](https://modal.com/docs/guide/webhook-urls#custom-domains).
#    This will allow you to configure Prometheus to scrape metrics from a predictable URL rather than
#    the autogenerated URL Modal assigns to your app.

gw_app = modal.App(
    "pushgateway-example",
    image=gw_image,
)


@gw_app.function(keep_warm=1, concurrency_limit=1)
@modal.web_server(9091)
def run_pushgateway():
    subprocess.Popen("/usr/local/bin/pushgateway")


# ## Push metrics to the Pushgateway
#
# Now that we have the Pushgateway running, we can push metrics to it. We'll use the `prometheus_client`
# library to create a simple counter and push it to the Pushgateway. This example is a simple counter,
# but you can push any metric type to the Pushgateway.
#
# Note that we use the `grouping_key` argument to distinguish between different instances of the same
# metric. This is useful when you have multiple instances of the same app pushing metrics to the Pushgateway.
# Without this, the Pushgateway will overwrite the metric with the latest value.

client_image = modal.Image.debian_slim().pip_install(
    "prometheus-client==0.20.0"
)
app = modal.App(
    "client-example",
    image=client_image,
)

with client_image.imports():
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        delete_from_gateway,
        push_to_gateway,
    )


@app.cls(keep_warm=3)
class ExampleClientApplication:
    @modal.enter()
    def init(self):
        self.registry = CollectorRegistry()
        self.web_url = run_pushgateway.web_url
        self.instance_id = os.environ["MODAL_TASK_ID"]
        self.counter = Counter(
            "hello_counter",
            "This is a counter",
            registry=self.registry,
        )

    # We must explicitly clean up the metric when the app exits so Prometheus doesn't
    # keep stale metrics around.
    @modal.exit()
    def cleanup(self):
        delete_from_gateway(
            self.web_url,
            job="hello",
            grouping_key={"instance": self.instance_id},
        )

    @web_endpoint()
    def hello(self):
        self.counter.inc()
        push_to_gateway(
            self.web_url,
            job="hello",
            grouping_key={"instance": self.instance_id},
            registry=self.registry,
        )
        return f"Hello world from {self.instance_id}!"


app.include(gw_app)

# Now, we can deploy the `client-example` app and see the metrics in the Pushgateway's web interface.

# ```shell
# $ modal deploy pushgateway.py
# ✓ Created objects.
# ├── 🔨 Created mount /Users/example/projects/scratch/pushgateway/pushgateway.py
# ├── 🔨 Created web function run_pushgateway => https://modal-labs-example--client-example-run-pushgateway.modal.run
# ├── 🔨 Created function ExampleClientApplication.*.
# └── 🔨 Created web function ExampleClientApplication.hello =>
#     https://modal-labs-example--client-example-exampleclientappli-4c6f64.modal.run (label truncated)
# ✓ App deployed! 🎉
#
# View Deployment: https://modal.com/modal-labs/example/apps/deployed/client-example
# ```
#
# Go to both the client application and Pushgateway URLs to see the metrics being pushed.
#
# ## Hooking up Prometheus
#
# Now that we have metrics in the Pushgateway, we can configure Prometheus to scrape them. This
# is as simple as adding a new job to your Prometheus configuration. Here's an example configuration
# snippet:
#
# ```yaml
# scrape_configs:
# - job_name: 'pushgateway'
#   honor_labels: true # required so that the instance label is preserved
#   static_configs:
#   - targets: ['modal-labs-example--client-example-run-pushgateway-dev.modal.run']
# ```
#
# Note that the target will be different if you have a custom domain set up for the Pushgateway,
# and you may need to configure authentication.
#
# Once you've added the job to your Prometheus configuration, Prometheus will start scraping metrics
# from the Pushgateway. You can then use Grafana or another visualization tool to create dashboards
# and alerts based on these metrics!
#
# ![Grafana example](./pushgateway_grafana.png)
