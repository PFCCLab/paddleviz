import subprocess
import os

# 执行控制台命令
command = "python ./test.py"
# result = subprocess.run(command, shell=True, capture_output=True, text=True)

with open("output.txt", "w") as f:
    subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)