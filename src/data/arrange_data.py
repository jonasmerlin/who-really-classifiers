import argparse, os, shutil, warnings, glob
import scipy.io as sio
import git
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("set", help="IMDB or Wiki Data?")
parser.add_argument("feature", help="The feature we want to create the dataset for. (e.g. gender)")
parser.add_argument("num_training_images", help="The number of per class images you want to train on.", type=int)
parser.add_argument("num_validation_images", help="The number of per class images you want to validate on.", type=int)
parser.add_argument("-c", "--clean", help="remove corrupt images from wiki data", action="store_true")
args = parser.parse_args()

dataset = args.set

num_training_images = args.num_training_images
num_validation_images = args.num_validation_images

# https://stackoverflow.com/a/41920796
def get_git_root(path):
        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        print(git_root)
        return git_root

git_root                            = get_git_root(os.path.abspath(__file__))

package_directory                   = os.path.dirname(os.path.abspath(__file__))
data_directory                      = os.path.join(git_root, "datasets/")

wiki_directory                      = os.path.join(data_directory, "wiki/")
imdb_directory                      = os.path.join(data_directory, "imdb_crop/")

if dataset == "wiki":
    dataset_directory               = wiki_directory
    meta_data_path                  = os.path.join(data_directory, "wiki/wiki.mat")
    corrupted_directory             = os.path.join(data_directory, "wiki/corrupted")
if dataset == "imdb":
    dataset_directory               = imdb_directory
    meta_data_path                  = os.path.join(data_directory, "imdb_crop/imdb.mat")
    corrupted_directory             = os.path.join(data_directory, "imdb_crop/corrupted")

gender_data_directory               = os.path.join(data_directory, "gender-data")
gender_data_train_directory         = os.path.join(data_directory, "gender-data/train")
gender_data_train_female_directory  = os.path.join(data_directory, "gender-data/train/female")
gender_data_train_male_directory    = os.path.join(data_directory, "gender-data/train/male")
gender_data_val_directory           = os.path.join(data_directory, "gender-data/val")
gender_data_val_female_directory    = os.path.join(data_directory, "gender-data/val/female")
gender_data_val_male_directory      = os.path.join(data_directory, "gender-data/val/male")

if args.clean:
    # Some of the images in the set are corrupt. Here we filter those out.
    # Some of the corrupt images throw a warning. To programmatically catch those we elevate them to exceptions.
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        try:
            os.mkdir(corrupted_directory)
        except:
            print("Error while creating corrupted subdirectory in wiki. Subdirectory most likely already exists.")

        with os.scandir(dataset_directory) as it:
            for entry in it:
                if entry.is_dir() and entry.path != corrupted_directory:
                    print("Going through " + entry.path + " in search for weird entries.")
                    for file in glob.glob(entry.path + "/*.jpg"):
                        try:
                            im = Image.open(file)
                            if os.stat(file).st_size < 200:
                                print("Moving " + file + " for being weird.")
                                shutil.move(file, corrupted_directory)
                        except:
                            print("Moving " + file + " for having weird exif data.")
                            shutil.move(file, corrupted_directory)

# Here we load the image meta data from the matlab file.
# The list hierarchy is rather deep and a bit convoluted and
# I arrived at the feature variables through trial and error on the REPL.
meta_data = sio.loadmat(meta_data_path)
paths = meta_data[dataset][0][0][2][0]
names = meta_data[dataset][0][0][4][0]
gender = meta_data[dataset][0][0][3][0]

def create_gender_dirs():
    shutil.rmtree(gender_data_directory, ignore_errors=True)
    try:
        os.mkdir(gender_data_directory)
        os.mkdir(gender_data_train_directory)
        os.mkdir(gender_data_val_directory)
        os.mkdir(gender_data_train_female_directory)
        os.mkdir(gender_data_train_male_directory)
        os.mkdir(gender_data_val_female_directory)
        os.mkdir(gender_data_val_male_directory)
    except:
        print("Error while creating gender-data directories.")

if args.feature == "gender":

    create_gender_dirs()

    # We enumerate the image paths to get the corresponding index.
    # Because the data is ordered, the gender can be found at that same index in the gender list.
    # Depending on which gender we find, we copy the image over the corresponding folder.  
    num_train_female = 0
    num_val_female = 0
    num_train_male = 0
    num_val_male = 0

    for index, image in enumerate(paths):
        image_path = os.path.join(dataset_directory, image[0])
        if os.path.exists(image_path):
            if gender[index] == 1.0:
                if num_train_male < num_training_images:
                    shutil.copy2(image_path, gender_data_train_male_directory)
                    num_train_male += 1
                elif num_val_male < num_validation_images:
                    shutil.copy2(image_path, gender_data_val_male_directory)
                    num_val_male += 1    
            if gender[index] == 0.0:
                if num_train_female < num_training_images:
                    shutil.copy2(image_path, gender_data_train_female_directory)
                    num_train_female += 1
                elif num_val_female < num_validation_images:
                    shutil.copy2(image_path, gender_data_val_female_directory)
                    num_val_female += 1
