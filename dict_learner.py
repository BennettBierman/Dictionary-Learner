import numpy as np
from image_utils import patch_img, unpatch_img, normalize


class DictionaryLearner:
    """
    Class maintains all elements needed for Dictionary Learning, including
    an image, a learned dictionary, and the sparse representation of that image

    Utilizes Orthogonal Matching Pursuit and K-SVD to train dictionary and create
    spare representation from input image
    """
    def __init__(self, image, size_info):
        """
        DictionaryLearner constructor
        Normalizes input to black-and-white image, creates image representation
        through patches, initializes dictionary to random columns of representation,
        and sets constants
        :param image: original png image OR path to saved state
        :param patch_size: dimension for square patches
        :param atoms: number of columns in dictionary
        """
        # Normal Constructor
        if size_info is not None:
            patch_size, atoms = size_info
            self.true_image = normalize(image)
            self.patch_size = patch_size
            p, (r, c, x, y) = patch_img(self.true_image, self.patch_size)
            self.patch_dims = (r, c, x, y)
            self.image_rep = p
            self.features = x * y  # rows of image representation
            self.signals = r * c  # columns of image representation
            self.atoms = atoms  # columns of dictionary
            self.sparse_rep = None
            self.dictionary = None
            self.recreation = None
            self.create_dict()
        # Loading in saved state from text files
        else:
            self.true_image = np.loadtxt(f'encodings/{image}_image')
            self.dictionary = np.loadtxt(f'encodings/{image}_dictionary')
            self.sparse_rep = np.loadtxt(f'encodings/{image}_sparse')
            dict_row, dict_col = self.dictionary.shape
            self.patch_size = np.sqrt(dict_row)
            p, (r, c, x, y) = patch_img(self.true_image, self.patch_size)
            self.patch_dims = (r, c, x, y)
            self.image_rep = p
            self.features = x * y  # rows of image representation
            self.signals = r * c  # columns of image representation
            self.atoms = dict_col  # columns of dictionary

            r, c, vals = np.loadtxt(f'encodings/{image}_sparse')
            temp = np.zeros((self.atoms, self.signals))
            temp[r.astype('int'), c.astype('int')] = vals
            self.sparse_rep = temp

            self.recreation = None
            self.set_recreation()

    def create_dict(self):
        """
        Sets dictionary to random columns of image representation
        """
        space = np.arange(0, self.signals, 1, dtype='int')
        col_idx = np.random.choice(space, self.atoms, replace=False)
        self.set_dictionary(self.image_rep.copy()[:, col_idx])

    def omp(self, img_rep, L):
        """
        Orthogonal Matching Pursuit Algorithm
        Produces a spare encoding given a dictionary and a signal
        :param img_rep: signal to sparse encode
        :param L: sparsity of the encoding
        :return alphas: sparse encoding of image representation
        """
        alphas = np.zeros(self.atoms)
        res = img_rep.copy()
        idx_set = []

        for _ in range(L):
            # Greedily find best atom
            inner_prod = np.matmul(self.dictionary.T, res)
            max_idx = np.argmax(np.abs(inner_prod))
            idx_set.append(max_idx)

            # Update all coefficients with orthogonal projection
            lst_sq = np.linalg.lstsq(self.dictionary[:, idx_set], res, rcond=None)[0]
            alphas[idx_set] = lst_sq

            # Update residual
            res = img_rep - np.matmul(self.dictionary, alphas)

        return alphas

    def ksvd_train(self, L, iter):
        """
        K-SVD Dictionary Learning Algorithm
        Simultaneously generates a sparse encoding and a useful dictionary of atoms
        given a signal or database of signals
        Reports accuracy and sparsity of encoding
        :param L: sparsity of each encoding
        :param iter: iterations of the K-SVD algorithm
        """
        img_norm = np.linalg.norm(self.image_rep)

        for t in range(iter):
            print(f'Iteration {t + 1}')

            # Update the spare encoding of each signal
            self.sparse_rep = np.apply_along_axis(self.omp, 0, self.image_rep, L)
            print('OMP Step Done')

            # Update Dictionary
            for k in range(self.atoms):
                idx_k = np.nonzero(self.sparse_rep[k, :])[0]

                # Skip step if no signals use this atom
                if len(idx_k) == 0:
                    continue

                # Use SVD to update dictionary
                X_k = self.sparse_rep[:, idx_k]
                delta = np.outer(self.dictionary[:, k], X_k[k, :]) - np.dot(self.dictionary, X_k)
                E = self.image_rep[:, idx_k] + delta
                U, S, V = np.linalg.svd(E)
                self.dictionary[:, k] = U[:, 0]
                self.sparse_rep[k, idx_k] = S[0] * V[0, :]

            print('Dict Update Step Done')

            # Error measured as (residual norm) / (image representation norm)
            error = np.linalg.norm(self.image_rep - np.matmul(self.dictionary, self.sparse_rep))/img_norm
            acc_msg = "{:.4f}".format((1-error)*100)
            print(f'Accuracy: {acc_msg}%')
            print('---------')

        # Create re-creation of original image with dictionary & sparse encoding
        self.set_recreation()

        # Reports sparsity of sparse encoding as percent
        s = self.sparse_rep.size
        nz = np.count_nonzero(self.sparse_rep)
        sp_msg = "{:.4f}".format(100*(s-nz)/s)
        print(f'Sparsity: {sp_msg}% \n')

    def save_state(self, file_name):
        """
        Save true image, dictionary, and sparse representation to a text file
        :param file_name
        """
        if self.sparse_rep is not None:
            with open(f'encodings/{file_name}_image', 'w') as f:
                np.savetxt(f, self.true_image)
            with open(f'encodings/{file_name}_dictionary', 'w') as f:
                np.savetxt(f, self.dictionary)
            with open(f'encodings/{file_name}_sparse', 'w') as f:
                r, c = np.nonzero(self.sparse_rep)
                vals = self.sparse_rep[r, c]
                np.savetxt(f, (r, c, vals))

    def set_recreation(self):
        """
        Recreates original image with learned dictionary and sparse representation
        and saves recreated image
        """
        if self.dictionary is not None and self.sparse_rep is not None:
            prod = np.matmul(self.dictionary, self.sparse_rep)
            rec_img = unpatch_img(prod, self.patch_dims)
            self.recreation = rec_img

    def get_recreation(self):
        """
        Returns recreated image
        :return: recreated image
        """
        return self.recreation

    def get_image(self):
        """
        Returns true image
        :return: true image
        """
        return self.true_image

    def set_dictionary(self, d):
        """
        Sets dictionary to any value
        :param d: dictionary value
        """
        self.dictionary = d
