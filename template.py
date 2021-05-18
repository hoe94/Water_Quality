import os

folder = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src"
]

for folder_ in folder:
    os.makedirs(folder_, exist_ok = True)
    with open(os.path.join(folder_, ".gitkeep"), 'w')as f:
        pass

files = [
    "config.yaml",
    "dvc.yaml",
    os.path.join("src", "__init__.py")
]

for file_ in files:
    with open(file_, "w") as f:
       pass 
