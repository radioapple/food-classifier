from pathlib import Path

# ===== For importing full dataset =====
def import_full_dataset(directory_name: str = None, 
                        print_steps: bool = False,
                        return_data_path: bool = False):
    """
    Loads in Food101 dataset. If <directory_name>
    is given, the dataset is loaded into that directory instead.
    
    Parameters
    ----------
    directory_name : str, optional
        Directory to load dataset into. The default is None.
    print_steps : bool, optional
        Print steps such as creating the directory or extracting
        the data. The default is False.
    return_data_path : bool, optional
        Function returns path to the imported data if True, and None
        otherwise. The default is False.

    Returns
    -------
    str or Path or None.
        Returns path to the imported dataset, if <return_data_path>
        is True, and None otherwise.
    """
    # import modules
    import requests
    from pathlib import Path
    
    # Creating directory, if name is given
    if directory_name:
        dir_path = Path(directory_name)
        data_path = dir_path / "food101.tar.gz"
        
        if dir_path.is_dir():
            if print_steps:
                print("Directory already exists, skipping creation.")
        else:
            if print_steps:
                print("Creating directory...")
            dir_path.mkdir(exist_ok = True)
            if print_steps:
                print(f"Directory '{dir_path}' was created.")
                
    else:
        dir_path = None
        data_path = "food101.tar.gz"
        
    # Writing data to directory
    link_to_data = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    with open(data_path, "wb") as f:
        request = requests.get(link_to_data)
        if print_steps:
            print(f"Downloading '{link_to_data}'...")
        f.write(request.content)
        if print_steps:
            print("Download finished.")
            
    if return_data_path:
        return data_path
    else:
        return


# ===== For unzipping a zip file =====
def extract_zipfile(file_path: str or Path, 
                    directory_name: str = None, 
                    print_steps: bool = False, 
                    return_data_path: bool = False):
    """
    Unzips the file given by <file_name>. You can give a directory name
    to store the zip file's contents in if you wish.

    Parameters
    ----------
    file_path : str or Path
        Path of the zip file, including extension.
    directory_name : str, optional
        Directory you wish to load the zipfile's data into. The default is None,
        and will simply load data into whichever directory the file executing this
        function is in.
    print_steps : bool, optional
        Prints steps if True. The default is False.
    return_data_path : bool, optional
        Function returns path to the imported data if True, and None
        otherwise. The default is False.

    Returns
    -------
    str or Path or None.
        Returns path to the imported dataset, if <return_data_path>
        is True, and None otherwise.
    """
    import zipfile
    from pathlib import Path
            
    # --- Create directory, if name given ---
    if directory_name:
        dir_path = Path(directory_name)
        
        # Create directory, if it doesn't already exist
        if dir_path.is_dir():
            if print_steps:
                print(f"Directory {dir_path} already exists. Skipping creation.")
        else:
            if print_steps:
                print(f"Creating directory {dir_path}...")
            dir_path.mkdir(parents = True, exist_ok = True)
            if print_steps:
                print(f"Directory {dir_path} created.")
    else:
        dir_path = None
    
    # --- Unzip ---
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        data_path = zip_ref.namelist()[0]
        if print_steps:
            print(f"Unzipping {file_path}...")
        zip_ref.extractall(dir_path)
        if print_steps:
            print("Done!")
    
    # --- Return --
    if return_data_path:
        if dir_path:
            return dir_path / data_path
        else:
            return data_path
    else:
        return


# ===== For extracting data from a tar file ====
def extract_tarfile(file_path: str or Path,
                    directory_name: str = None,
                    print_steps: bool = False,
                    return_data_path: bool = False):
    """
    Extracts the tar file given by <file_path>.

    Parameters
    ----------
    file_path : str or Path
        Tar file's path, including extension.
    directory_name: str
        The directory you wish to extract the tar file into. If it doesn't
        exist yet, it will be created for you.
    print_steps : bool, optional
        Prints steps if True. The default is False.
    return_data_path : bool, optional
        Function returns path to the imported data if True, and None
        otherwise. The default is False.

    Returns
    -------
    str or Path or None.
        Returns path to the imported dataset, if <return_data_path>
        is True, and None otherwise.
    """
    import tarfile
    from pathlib import Path
    # --- Create directory, if name given ---
    if directory_name:
        dir_path = Path(directory_name)
        
        # Create directory, if it doesn't already exist
        if dir_path.is_dir():
            if print_steps:
                print(f"Directory {dir_path} already exists. Skipping creation.")
        else:
            if print_steps:
                print(f"Creating directory {dir_path}...")
            dir_path.mkdir(parents = True, exist_ok = True)
            if print_steps:
                print(f"Directory {dir_path} created.")
    else:
        dir_path = None
    
    # --- Extract tar file ---
    if print_steps:
        print(f"Extracting {file_path}...")
    tar = tarfile.open(file_path)
    if dir_path:
        tar.extractall(dir_path)
        data_path = dir_path / (tar.get_names()[0])
    else:
        tar.extractall()
        data_path = Path(tar.get_names()[0])
    tar.close()
    if print_steps:
        print(f"Finished extracting {file_path}.")
    
    # --- Return ---
    if return_data_path:
        return data_path
    else:
        return


# ===== For Getting Directory Paths for Food101 Dataset =====
def get_directory_paths(dataset_dir_name: str or Path):
    """Returns all directory path names in the Food101 dataset.
    
    Parameters
    ----------
    dataset_dir_name: str
        The name of the directory in which the food101 directories are
        held (i.e. where the images and meta folders are).
        
    Returns 
    -------
    Path, Path
        Returns 2 path objects corresponding to the path to the images directory,
        and meta directory respectively.
    """
    if type(dataset_dir_name) is str:
        from pathlib import Path
        dataset_dir_name = Path(dataset_dir_name)
        
    # Creating main directory paths
    images_path = dataset_dir_name / "images"
    meta_path = dataset_dir_name / "meta"
    
    return images_path, meta_path


# ===== Extracts data from the 'Meta' directory in the Food101 dataset =====
def get_meta_data(meta_path: str or Path):
    """
    Returns dict containing all data from the 'Meta' folder in the Food101
    dataset.
    
    Parameters
    ----------
    meta_path : str or Path
        Path to the "Meta" directory in the Food101 dataset.

    Returns
    -------
    Dict.
        The dictionary contains data from each file in the 'Meta' directory
        in the Food101 dataset (excluding the README file). The keys of the
        dict describe what the data is as well as the type of object the data
        is stored in.
    """
    # Import needed modules
    import json
    
    if type(meta_path) is str:
        from pathlib import Path
        meta_path = Path(meta_path)
        
    # Getting list of files in the 'food-101/meta' directory
    meta_filenames_list = sorted(list(meta_path.glob("*")))[1:] # ignoring the README file
    
    # Compile data into an array
    meta_files_data = []
    for file_name in meta_filenames_list:
      if str(file_name)[-5:] == ".json":
        data = json.load(open(file_name))
        meta_files_data.append(data)
      elif str(file_name)[-4:] == ".txt":
        with open(str(file_name), 'r') as f:
          data = [line.strip() for line in f.readlines()]
        meta_files_data.append(data)
    
    # Based the order off of order that file names appear in meta_filenames_list
    lst_keys = ['class_names', 'labels','test_data_paths_dict', 'test_data_paths_lst', \
        'train_data_paths_dict', 'train_data_paths_lst']
    meta_data_dict = {}
    meta_data_dict.update(zip(lst_keys, meta_files_data))
    
    return meta_data_dict

    
# ===== Rearrange Files into "images/train/class_name/image_name.jpg" format =====
def rearrange_files(images_path: str or Path, 
                    class_names: list, 
                    train_data_paths_dict: dict, 
                    test_data_paths_dict: dict,
                    percent: float = None,
                    print_steps: bool = False):
    """
    Rearranges files inside the Food101 dataset into images/train/class_name/image_name.jpg
    format as the files are originally in the images/class_name/image_name.jpg
    format.

    Parameters
    ----------
    images_path : str or Path
        Path to the 'images' folder in the Food101 dataset.
    class_names : list[str]
        List of all class names.
    train_data_paths_dict : dict
        Dict containing all training image paths.
    test_data_paths_dict : dict
        Dict containing all test image paths.
    percent : float, optional
        <percent> is the percentage of the total data you want (must be between 0 and 1). 
        This is if you only want some subset of the Food101 class. The default is
        None, which just means that it takes the entire dataset.
    print_steps : bool, optional
        Set to True if you want to see which directories are being created and all
        the moves, and False otherwise. The default is False.
        

    Returns
    -------
    Path, Path, Dict[str: Path], Dict[str: Path].
        Returns the training data path, testing data path, a dict of paths
        inside the training directory, and a dict of paths inside the testing directory,
        respectively. The keys of the dict are class names and the values are their 
        corresponding directory paths (e.g. {'ramen': '../images/train/ramen'}).
    """
    # Import modules
    import shutil
    import random
    
    if type(images_path) is str:
        from pathlib import Path
        images_path = Path(images_path)
        
    # Train and test directories
    train_path = images_path / "train"
    test_path = images_path / "test"
    
    # Class directories inside train and test directories
    classes_paths = {class_name: images_path / class_name for class_name in class_names}
    train_paths = {class_name: train_path / class_name for class_name in class_names}
    test_paths = {class_name: test_path / class_name for class_name in class_names}
    train_test_paths = [train_paths, test_paths]
    
    # Create train and test directories, along with directories for each class inside them
    if print_steps:
        print("=== Creating Directories ===")
        
    for paths in train_test_paths:  # gets [train_paths, test_paths]
        for class_name in sorted(paths):
            if paths[class_name].is_dir():
                if print_steps:
                    print(f"Directory '{paths[class_name]}' already exists, skipping creation.")
            else:
                paths[class_name].mkdir(parents = True, exist_ok = True)
                if print_steps:
                    print(f"Directory '{paths[class_name]}' created.")

    # Move images into train and test folders instead
    if print_steps:
        print("\n=== Moving Files ===")
        
    for i, paths in enumerate(train_test_paths): # gets [train_paths, test_paths]
        if i == 0: # if we're in the images/train directory
            arr = train_data_paths_dict
            destination_dir_path = train_path
        else: # we're in the images/test directory
            arr = test_data_paths_dict
            destination_dir_path = test_path

        for class_name in sorted(paths):
            if percent:
                subset_length = int(percent * len(arr[class_name]))
                sub_arr = random.sample(arr[class_name], k = subset_length)
            else:
                sub_arr = arr[class_name]
                
                
            for image_path in sub_arr: # looks through training or testing image paths
                source = str(images_path / image_path) + ".jpg"
                destination = str(destination_dir_path / image_path) + ".jpg"
                shutil.move(source, destination) # moves image from source to train/test directories
                if print_steps:
                    print(f"File '{source}' was moved to '{destination}'.")
    
    # Delete the empty left over directories
    uneccessary_folder = images_path.parent.parent.absolute() / "__MACOSX"
    if uneccessary_folder.is_dir():
        shutil.rmtree(uneccessary_folder)
        
    for x in sorted(classes_paths):
        path = classes_paths[x]
        shutil.rmtree(path)
        
    return train_path, test_path, *train_test_paths


# ===== Get All Image Paths =====
def get_image_paths_list(images_path: str or Path):
    """
    Returns all image paths, assuming that files are now in
    'images/train/class_name/image_name.jpg' and
    'images/test/class_name/image_name.jpg' format.
    
    Parameters
    ----------
    images_path : str or Path
        Path to 'images' directory.
    print_info : bool, optional
        Prints information about how many images there are in total, if set
        to True. The default is False.

    Returns
    -------
    image_paths_list : list
        A list containing paths to all images in the dataset.
    """
    if type(images_path) is str:
        from pathlib import Path
        images_path = Path(images_path)
    
    image_paths_list = list(images_path.glob("*/*/*.jpg"))
    
    return image_paths_list


# ===== Print information about the dataset after everything else =====
def print_info(class_names: list,
               train_test_paths: list,
               image_paths_list: list):
    """
    Prints information about how many classes there are,
    number of training images per class, number of test images
    per class, total number of train images, total number of 
    test images, and total number of images overall.

    Parameters
    ----------
    class_names : list
        List containing all class names.
    train_test_paths : list[dict[str: Path]]
        List whose first element is a dict containing all train paths
        and second element is a dict containing all test paths. The keys
        of the dict are the class names and the values are their
        corresponding directory paths (e.g.{'ramen': '.../images/train/ramen'}).
    image_paths_list : list
        List containing all image paths.

    Returns
    -------
    None.

    """
    import os
    # Variables to print
    num_classes = len(class_names)

    num_train_images_per_class = len(os.listdir(train_test_paths[0][class_names[0]]))
    num_test_images_per_class = len(os.listdir(train_test_paths[1][class_names[0]]))

    num_train_images_tot = num_classes * num_train_images_per_class
    num_test_images_tot = num_classes * num_test_images_per_class

    # Printing...
    print(f"Number of classes: {num_classes}")

    print(f"\nNumber of train images per class: { num_test_images_per_class }")
    print(f"Number of test images per class: { num_train_images_per_class }")

    print(f"\nNumber of train images in total: { num_train_images_tot }")
    print(f"Number of test images in total: { num_test_images_tot }")

    print(f"\nNumber of images in total: {len(image_paths_list)}")
    
    return




# ===== Function for Loading, Extracting, and Getting all Values =====
def load_and_extract_data(file_path: str or Path = None,
              directory_name: str or Path = None,
              uploaded: bool = False,
              percent: float = None,
              print_steps_data_loading: bool = False,
              print_steps_rearranging: bool = False,
              get_info: bool = True):
    """
    Loads and extracts Food101 data. Returns all important paths, arrays,
    and dicts.
    
    If data has already been uploaded and you wish to use that data, set
    <uploaded> to True and provide the uploaded data's path. Otherwise,
    if data isn't uploaded or you wish to import and work with the full
    Food101 dataset, set <uploaded> to False.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to the data file you wish to extract from, or if data is already
        extracted, this would be the path to the directory in which the data
        is contained.The default is None. If <uploaded> is True, a value for
        this parameter must be provided. If <uploaded> is False, the function
        will load in the full dataset instead.
    directory_name : str or Path, optional
        Directory that you wish to import the data into. If it doesn't exist,
        it will be created. The default is None.
    uploaded : bool, optional
        True if data has already been uploaded, False otherwise. The default is
        False. If <uploaded> is True however, then <file_path> to the uploaded data
        must be provided.
    percent : float, optional
        Must be between 0 and 1, and non-zero. Specifies what percent of the data you would 
        like to look at. The function then chooses that specific percent from 
        each class to create a subset. The default is None, and will simply look
        at the full dataset.
    print_steps_data_loading : bool, optional
        Prints throughout the data importing and extracting process if set to True. 
        The default is False.
    print_steps_rearranging : bool, optional
        Prints throughout the dataset directories format changing process if set 
        to True. The default is False.
    get_info : bool, optional
        Prints description of loaded in data if set to True. The default is True.

    Raises
    ------
    Exception
        Raises exception if <file_path> isn't provided, but <uploaded> is set to
        True.
        Raises exception if <percent> isn't None or between 0 and 1 and non-zero.

    Returns
    -------
    result : dict
        Returns a dict object containing all important variables, arrays, and dicts.
    """
    # ------- Exceptions -------
    # --- Check that file name is given if data has been uploaded ---
    if file_path is None and uploaded is True:
        raise Exception("File path needs to be specified if data has been uploaded.")
    # --- Check that <percent> is appropriate ---
    if percent is not None:
        if not (0 < percent <= 1):
            raise Exception("Percent needs to be between 0 and 1, and non-zero.")
        if percent == 1:
            percent = None
    
    # ------- Importing and Extracting Files -------
    # ---- If not Uploaded, Load in Data ----
    if not uploaded:
        loaded_data_path = import_full_dataset(directory_name = directory_name, 
                                               print_steps=print_steps_data_loading,
                                               return_data_path = True)
        data_path = extract_tarfile(file_path = loaded_data_path, 
                                    directory_name = directory_name,
                                    print_steps = print_steps_data_loading,
                                    return_data_path = True)
    # ---- If Uploaded ----
    else:
        # Extract from Zip file
        if str(file_path)[-4:] == ".zip":
            data_path = extract_zipfile(file_path = file_path,
                                        directory_name = directory_name,
                                        print_steps = print_steps_data_loading,
                                        return_data_path = True)
        # Or Extract from Tarfile
        elif str(file_path)[-7:] == ".tar.gz":
            data_path = extract_tarfile(file_path = file_path,
                                        directory_name = directory_name,
                                        print_steps = print_steps_data_loading,
                                        return_data_path = True)
        # Or if data is already extracted
        else:
            data_path = file_path
            
    # ------- Get Directory Paths -------
    images_path, meta_path = get_directory_paths(dataset_dir_name = data_path)
    
    # ------- Get Meta Data -------
    meta_data = get_meta_data(meta_path)
    
    # ------- Rearrange Files -------
    train_path, test_path, train_paths, test_paths = \
        rearrange_files(images_path = images_path, 
                        class_names = meta_data['class_names'], 
                        train_data_paths_dict = meta_data['train_data_paths_dict'], 
                        test_data_paths_dict = meta_data['test_data_paths_dict'],
                        percent = percent,
                        print_steps = print_steps_rearranging)
    
    # ------- Get all image paths -------
    image_paths_list = get_image_paths_list(images_path = images_path)
    
    # ------- Print information about dataset -------
    if get_info:
        print_info(class_names = meta_data['class_names'], 
                   train_test_paths = [train_paths, test_paths], 
                   image_paths_list = image_paths_list)
    
    # ------- Return all important data -------
    result_keys = ['data_path', 'images_path', 'meta_path', 'train_path', 'test_path', \
                   'meta_data', 'train_paths', 'test_paths', 'image_paths_list']
    results_lst = [data_path, images_path, meta_path, train_path, test_path, \
                   meta_data, train_paths, test_paths, image_paths_list]
    result = dict(zip(result_keys, results_lst))
    
    return result