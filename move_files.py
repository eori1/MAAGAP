import os
import shutil

base_dir = r"c:\Users\ASUS\Desktop\Tisis"
backend_dir = os.path.join(base_dir, "backend")

if not os.path.exists(backend_dir):
    os.makedirs(backend_dir)

items = os.listdir(base_dir)

# Items to ignore from moving
ignore_list = ["backend", ".git", ".gitignore", "move_files.py"]

for item in items:
    if item in ignore_list:
        continue
    
    src = os.path.join(base_dir, item)
    dst = os.path.join(backend_dir, item)
    
    try:
        shutil.move(src, dst)
        print(f"Moved {item} to backend/")
    except Exception as e:
        print(f"Failed to move {item}: {e}")
