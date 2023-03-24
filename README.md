# Dictionary-Learner
**Program that deploys K-SVD and Orthogonal Matching Pursuit to perform dictionary learning from black-and-white images**



**ENVIRONMENT INSTALLATION:**
- Clone/fork project
- Install the appropriate version of conda for your OS
- Navigate to directory
- Enter: conda env create -f environment.yml



**REPO OUTLINE:**

This program allows you to create novel sparse representations of images through K-SVD and OMP dictionary learning

Directories:

- **images**: folder for saving images that the program can easily access
- **encodings**: folder holds text representation (of numpy matrices) of sparse encodings (original image, dictionary, sparse encoding)

Files:

- **main.py**: access point of the entire project
- **dict_learner.py**: object responsible for running dictionary learning algorithms and holding data
- **image_utils.py**: utility functions for easily handling image data



**GETTING STARTED**

All features can be accessed through the starting command: python main.py

There are three main features of the program accessed through the flag: --func

- **train**: creates a new spare representation and dictionary from an input image then plots the recreated and original images
- **train_save**: performs the actions of train, then saves the state of your DictionaryLearner object (original image, dictionary, spare representation) in text files in the encodings directory
- **load_show**: uploades state from saved text files in encodings and plots original image and sparse representation



**FUNCTIONALITY**

The three central features can be fine tuned with a number of flags

Flags to use with all three functions:

- --pic_size [int]: size of plots when original image and sparse representation are plotted

Flags to use with --func train or --func train_save:

- --pat_size [int]: patch sized used when augmenting original image for dictionary learning
- --atoms [int]: number of atoms in the learned dictionary
- sparse [int]: sparsity of each column in the sparse representation 
- --iter [int]: number of iterations K-SVD will be run
- --image [str]: name of file in images folder (do not need to include ".png") than can be used for dictionary learning
- --path [str]: path to any image on computer that can also be used for dictionary learning

Flags to use with just --func train_save:

- --save_state [str]: prefix used to name saved state text files that will be created in the encodings folder

Flags to use with just --func load_show:

- --load_state [str]: prefix of files in encodings folder that you wish to load the state of
