# ---
# lambda-test: false
# ---

import time
from datetime import datetime, timezone

import modal

stub = modal.Stub("example-say-hello-cron")


@stub.function(schedule=modal.Period(seconds=10))
def say_hello():
    start_time = datetime.now(timezone.utc)
    for i in range(10):
        print(f"Message #{i} from invocation at {start_time}")
        time.sleep(1.5)
