import argparse
import numpy as np
from PIL import Image
from dict_learner import DictionaryLearner as dLearner
from image_utils import show_images, normalize
"""
Main file where all program functionality can be accessed
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pat_size", type=int, default=5)  # patch size
    parser.add_argument("--pic_size", type=int, default=8)  # plot size of images
    parser.add_argument("--atoms", type=int, default=512)  # number of atoms in dictionary
    parser.add_argument("--sparse", type=int, default=12)  # sparsity of representation
    parser.add_argument("--iter", type=int, default=10)  # iterations of k-svd
    parser.add_argument("--image", type=str, default="shoe")  # image name from images folder
    parser.add_argument("--path", type=str, default="")  # file path to any image
    parser.add_argument("--load_state", type=str, default="piano")  # prefix of saved encoding file
    parser.add_argument("--save_state", type=str, default="test")  # prefix for saving state
    parser.add_argument("--func", type=str, default="train",
                        choices=["train", "train_save", "load_show"])  # functionality of program
    args = parser.parse_args()

    if args.func == 'train' or args.func == 'train_save':
        """
        Construct DictionaryLearner object from an image
        Run K-SVD from image input
        Plot original input and spare representation 
        """
        path = args.path if len(args.path) > 0 else f'images/{args.image}.png'
        image = Image.open(path).convert('L')
        img = np.array(image)

        # Create spare representation & dictionary learn from image
        learner = dLearner(img, (args.pat_size, args.atoms))
        learner.ksvd_train(args.sparse, args.iter)
        rec = learner.get_recreation()

        # Save state of dLearner object to text file in encodings folder
        if args.func == 'train_save':
            learner.save_state(args.save_state)
            print('State Saved \n')

        # Plot original image and representation
        show_images([normalize(img), rec], args.pic_size)

    elif args.func == 'load_show':
        """
        Load DictionaryLearner object from text files 
        Plot original input and sparse representation 
        """
        learner = dLearner(args.load_state, None)
        rec = learner.get_recreation()
        show_images([learner.get_image(), rec], args.pic_size)
