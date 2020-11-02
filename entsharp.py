# entsharp.py
#
# Embed words in image space.
# Input: image embeddings, image-word co-occurrence matrix
# Output: word embeddings in image space
# 
# Image embeddings are fixed.
# Assign images membership in clusters corresponding to co-occurring words.
# Iteratively sharpen distributions.

import numpy as np
import os
import argparse
import scipy.sparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_features',
        type=str)

    parser.add_argument(
        '--image_word_cooccurrences',
        type=str)

    parser.add_argument(
        '--n_iterations',
        type=int,
        default=100)

    parser.add_argument(
        '--name',
        type=str)

    return parser.parse_args()

def main():
    args = parse_args()

    save_dir = './results_{}'.format(args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load and normalize image embeddings
    print('Loading image embeddings.')
    image_embeddings = np.load(args.image_features)
    normalizer = 1.0 / np.sqrt(np.sum(image_embeddings ** 2, axis=1))
    image_embeddings *= normalizer[:, np.newaxis]
    
    print('Loading image-word co-occurrence matrix.')
    image_word_cooccurrences = scipy.sparse.load_npz(args.image_word_cooccurrences).toarray()
    print(image_word_cooccurrences.shape)

    num_clusters = image_word_cooccurrences.shape[1]
    num_images = image_embeddings.shape[0]
    num_dims = image_embeddings.shape[1]

    print('Number of words/clusters: {}'.format(num_clusters))
    print('Initializing clusters to average co-occurring image locations.')
    centroids = np.zeros((num_clusters, num_dims))
    for i in range(num_clusters):
        if i % 1000 == 0:
            print('Initializing word %d' % i)
        membership = image_word_cooccurrences[:, i]
        cooccurrences = np.squeeze(np.argwhere(membership > 0))
        associated_ims = image_embeddings[cooccurrences, :]
        centroids[i, :] = np.mean(associated_ims, axis=0)
    normalizer = 1.0 / np.sqrt(np.sum(centroids ** 2, axis=1))
    centroids *= normalizer[:, np.newaxis]

    print("Initializing each image's distribution over clusters.")
    cosines = np.dot(image_embeddings, centroids.transpose())
    exponentiated = np.exp(cosines)
    masked_cosines = np.multiply(exponentiated, image_word_cooccurrences)
    sums = np.sum(masked_cosines, axis=1)
    # handle the case where an image doesn't co-occur with any words
    # the sum doesn't matter because the row will remain zeros
    sums[sums == 0] = 1
    normalizer = 1.0 / sums
    norm_exp = masked_cosines * normalizer[:, np.newaxis]
    # now norm_exp is num_images x num_clusters
    # and each row is a membership distribution over clusters
    # with 0s for non-co-occurring image/word pairs

    print('Saving average baseline word embeddings!')
    np.save('{}/word-embeddings_average-zero-iter.npy'.format(save_dir), centroids)

    for iteration in range(1, args.n_iterations + 1):
        print('Iteration {}/{}'.format(iteration, args.n_iterations))

        print('\tUpdating centroids.')
        centroids = np.dot(norm_exp.transpose(), image_embeddings)
        normalizer = 1.0 / np.sqrt(np.sum(centroids ** 2, axis=1))
        centroids *= normalizer[:, np.newaxis]
        
        print('\tUpdating membership.')
        # sharpness coefficient is the iteration number
        sharpness = iteration
        cosines = sharpness * np.dot(image_embeddings, centroids.transpose())
        exponentiated = np.exp(cosines)
        masked_cosines = np.multiply(exponentiated, image_word_cooccurrences)
        sums = np.sum(masked_cosines, axis=1)
        # handle the case where an image doesn't co-occur with any words
        # the sum doesn't matter because the row will remain zeros
        sums[sums == 0] = 1
        normalizer = 1.0 / sums
        norm_exp = masked_cosines * normalizer[:, np.newaxis]

    print('Saving final word embeddings!')
    np.save('{}/word-embeddings.npy'.format(save_dir), centroids)

if __name__ == '__main__':
    main()
