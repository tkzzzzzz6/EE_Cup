import os

# 获取当前工作目录
root_dir = os.getcwd()

# 遍历所有子目录和文件
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower() == "desktop.ini":
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"无法删除 {file_path}: {e}")