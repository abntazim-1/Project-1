from setuptools import find_packages,setup
from typing import List

HYPEN_DOT_E = "-e ."
def get_requirement(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines
        requirments=[req.replace("\n","") for req in requirments]
        if HYPEN_DOT_E in requirments:
            requirments.remove(HYPEN_DOT_E)

    return requirments


setup (
    
name = "mlproject", 
version = "1.0",
author = "Tazim",
author_email = "abdullah_tazim@outlook.com",
packages = find_packages(),
install_requires = get_requirement("requirments.txt")


)