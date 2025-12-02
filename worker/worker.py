import time
import redis
from rq import Queue
from rq.job import Job
from jobs import sample_job   # import from module, NOT __main__

# Redis connection
redis_url = "redis://redis:6379"
redis_conn = redis.from_url(redis_url)

def main():
    q = Queue('default', connection=redis_conn)

    # Enqueue test job
    job: Job = q.enqueue(sample_job, 2)
    print(f"Enqueued test job: {job.id}")

    # Poll for job result
    for _ in range(10):
        job.refresh()
        if job.is_finished:
            print(f"Job finished! Result: {job.result}")
            break
        elif job.is_failed:
            print(f"Job failed: {job.exc_info}")
            break
        else:
            print("Waiting for job to finish...")
        time.sleep(1)

    print("Worker Sprint 0 demo finished.")

if __name__ == "__main__":
    main()
