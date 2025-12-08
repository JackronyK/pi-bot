import json
import math

# Given values from the problem description
a = 1
r = 1/7

# Calculate the sum of the infinite geometric series using the formula S = a / (1 - r)
# This formula is valid when |r| < 1, which is true for r = 1/7.
S = a / (1 - r)

# Prepare the result in a dictionary
result = {
    "S": S
}

# Print the JSON result to stdout
print(json.dumps(result))