import os

conda_check = os.system("conda --version")
if conda_check == 0:
    print("conda is installed!")
else:
    print("conda not found/please install it.")

env_name = "my_env" 
python_version = "3.10" 

print(f"creating environment '{env_name}' with Python {python_version}...")
os.system(f"conda create --name {env_name} python={python_version} -y")

print(f"activate your environment using: conda activate {env_name}")

packages = ["numpy", "pandas", "matplotlib"]
print("installing packages")
for package in packages:
    os.system(f"conda install {package} -y")
    print(f"Installed {package}")

installed_packages = {}
for package in packages:
    try:
        exec(f"import {package}")
        installed_packages[package] = "installed"
    except ImportError:
        installed_packages[package] = "not Found"

print("\nPackage Installation Status:")
for pkg, status in installed_packages.items():
    print(f"{pkg}: {status}")

import numpy as np
print("creating a simple NumPy array...")
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)

import pandas as pd
print("creating a simple Pandas DataFrame...")
data = {"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 22]}
df = pd.DataFrame(data)
print(df)

import matplotlib.pyplot as plt
print("plotting a simple line graph...")
x = [1, 2, 3, 4]
y = [10, 20, 15, 25]
plt.plot(x, y, label="simple Line")
plt.legend()
plt.show()

print("listing installed packages...")
os.system("conda list")

print(f"removing environment '{env_name}'...")
os.system(f"conda remove --name {env_name} --all -y")

envs_list = os.popen("conda env list").read()
print("\nAvailable Conda environments:")
print(envs_list)

print("practice Problems:") 