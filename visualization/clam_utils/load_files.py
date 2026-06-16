import os
from sklearn.model_selection import train_test_split

def load_files(dataset_path,ext=".h5"):
    """Return {'train', 'plot'} file lists from the dataset's training/ and testing/ dirs."""
    dataset = {}
    dataset['train'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "training")) for file in files if file.endswith(ext)]
    dataset['plot'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "testing")) for file in files if file.endswith(ext)]

    return dataset

def Get_train_valid_Path(Train_set, train_percentage=0.9):
    train_labels = [int(1) if "tumor" in os.path.splitext(os.path.basename(path))[0] else int(0) for path in Train_set]
    Model_Train, Model_Val, y_train, y_test = train_test_split(Train_set, train_labels, test_size=1-train_percentage,
                                                        random_state=12321, stratify=train_labels)

    return Model_Train, Model_Val


