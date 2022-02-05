import os
import shutil
from tkinter import *
from tkinter import filedialog

def chooseFolder(dest):
    root = Tk()
    root.withdraw()
    src = filedialog.askdirectory()

    print("Copying files from")
    print(os.listdir(src))
    print("into")
    print(os.listdir(dest))

    shutil.copytree(src, dest)

    print()
    print("Done!")

