import time

def sample_job(x):
    print(f"Running sample job with input: {x}")
    time.sleep(1)
    return x * 2
