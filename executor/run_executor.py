import time
import json
import sys

def main():
    print(json.dumps({"status":"executor_ready"}))
    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()

