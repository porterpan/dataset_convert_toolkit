import os

def check_folder(folder_name):
    '''
    检查文件夹是否存在, 如果不存在，创建文件夹
    '''
    if not os.path.exists(folder_name):        
        os.makedirs(folder_name)
        print(f"文件夹'{folder_name}'已创建。")
    else:
        print(f"文件夹'{folder_name}'已存在。")