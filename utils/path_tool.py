# 为整个工程提供统一的绝对路径

import os

def get_project_root() -> str:
    '''
    获取工程所在的根目录
    '''
    # 当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取工程的根目录，先获取文件所在的文件夹的绝对路径
    current_dir = os.path.dirname(current_file)
    # 获取文件夹根目录
    project_dir = os.path.dirname(current_dir)

    return project_dir

def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径吗，得到绝对路径
    """
    project_root = get_project_root()
    # print(os.path.join(project_root,relative_path))
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    print(get_abs_path("model.py"))
