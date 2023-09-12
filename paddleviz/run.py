import os
import sys
import subprocess

# 执行控制台命令
if len(sys.argv) < 2:
    print("Pleace input the name of model's file")
    sys.exit()


file_name = sys.argv[1]
command = "python " + file_name

with open("output.txt", "w") as f:
    subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)