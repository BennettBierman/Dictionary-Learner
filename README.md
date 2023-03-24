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
- **encodings**: folder holds text representations (of numpy matrices) of sparse encodings (original image, dictionary, sparse encoding)

Files:

- **main.py**: access point of the project
- **dict_learner.py**: object responsible for running dictionary learning algorithms and maintaining data
- **image_utils.py**: utility functions for easily handling image data



**GETTING STARTED**

All features can be accessed through the starting command: python main.py

There are three main features of the program that are accessed through the flag: --func

- **train**: creates a new spare representation and dictionary from an input image, plots the recreated and original images, and prints information about the accuracy and sparsity of the representation. The accuracy is calculated by dividing the norm of the residual by the norm of the image we are representing, and subtracting this fraction from one.
- **train_save**: performs the actions of train, then saves the state of your DictionaryLearner object (original image, dictionary, spare representation) in text files in the encodings directory. Given a prefix "x", files are saved as x_image, x_dictionary, and x_sparse in this folder
- **load_show**: uploades state from saved text files in encodings and plots original image and sparse representation



**FUNCTIONALITY**

The three central features can be fine tuned with a number of flags

Flags to use with all three functions:

- --pic_size [int]: size of plots when original image and sparse representation are plotted

Flags to use with --func train or --func train_save:

- --pat_size [int]: patch size used when augmenting original image for dictionary learning
- --atoms [int]: number of atoms in the learned dictionary
- sparse [int]: sparsity of each column in the sparse representation 
- --iter [int]: number of iterations K-SVD run
- --image [str]: name of file in images folder (do not need to include ".png" or file's path) that can be used for dictionary learning
- --path [str]: path to any image on computer that can also be used for dictionary learning

Flags to use with just --func train_save:

- --save_state [str]: prefix used to name saved state text files that will be created in the encodings folder

Flags to use with just --func load_show:

- --load_state [str]: prefix of files in encodings folder that you wish to load the state of

**FUTURE IMPROVEMENTS**

There are two central improvements I would like to make. 

First, I would like to add dictionary learning to colored images, not just black and white ones. 

Secondly, I would like to reimplement my K-SVD algorithm using the QR Decomposition approach. This should speed up the runtime of the dictionary update step
