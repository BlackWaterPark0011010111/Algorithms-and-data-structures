import os

conda_check = os.system("conda --version")
if conda_check == 0:
    print("conda is installed!")
else:
    print("conda not found/please install it.")

env_name = "my_env" 
python_version = "3.10" 
