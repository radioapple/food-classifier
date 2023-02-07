import requests
from pathlib import Path
  
def import_module(link: str, file_name: str):
  """File name must end with '.py'"""
  
  if Path(file_name).is_file():
    print(f"{file_name} already exists, skipped download.")
  else:
    print(f"Downloaded {file_name}.")
    request = requests.get(link)
    with open(file_name, "wb") as f:
      f.write(request.content)
  return

def import_model_data(links_and_file_names, directory_name: str = None):
  if directory_name:
    dir_path = Path(directory_name)

    # Create directory
    if dir_path.is_dir():
      print(f"Directory {dir_path} already exists, skipping creation")
    else:
      print(f"Creating directory {dir_path}...")
      dir_path.mkdir(parents = True, exist_ok = True)
      print(f"{dir_path} directory was created.")
  else:
    pass
  
  for link, file_name in links_and_file_names:
    import_model_data_helper(link, file_name, directory_name)
  
  return

def import_model_data_helper(link, file_name, directory_name: str = None):
  if directory_name:
    file_name = Path(directory_name) / file_name

  if file_name.is_file():
    print(f"{file_name} already exists. Skipped creation/download.")
  else:
    print("Downloading...")
    request = requests.get(link)
    with open(file_name, "wb") as f:
      f.write(request.content)
    print(f"Downloaded {file_name}.")
    
if __name__ == '__main__':
    data_loading_modules_link = "https://raw.githubusercontent.com/radioapple/food-classifier/main/Modules/data_loading_modules.py"
    data_loading_modules_file_name = "data_loading_modules.py"
    
    training_and_plotting_modules_link = "https://raw.githubusercontent.com/radioapple/food-classifier/main/Modules/training_and_plotting_modules.py"
    training_and_plotting_modules_file_name = "training_and_plotting_modules.py"
    
    import_module(data_loading_modules_link, data_loading_modules_file_name)
    import_module(training_and_plotting_modules_link, training_and_plotting_modules_file_name)