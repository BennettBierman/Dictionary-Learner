# Dictionary-Learner
**Program that deploys K-SVD and Orthogonal Matching Pursuit to perform dictionary learning from black-and-white images**


**ENVIRONMENT INSTALLATION:**
- Clone/fork project
- Install the appropriate version of conda for your OS
- Navigate to directory
- Enter: conda env create -f environment.yml



**REPO OUTLINE:**

This program allows you to create novel spare representations of images through K-SVD and OMP dictionary learning

Directories:

- **images**: folder for saving images that the program can easily access
- **encodings**: folder holds text representation (of numpy matrices) of sparse encodings (original image, dictionary, sparse encoding)

Files:

- **main.py**: access point of the entire project
- **dict_learner.py**: object responsible for running dictionary learning algorithms and holding data
- **image_utils.py**: utility functions for easily handling image data


