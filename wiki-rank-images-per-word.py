# wiki-rank-images-per-word.py
# 
# Process EntSharp results for multimodal Wikipedia dataset
# by calculating pairwise distances between word and image embeddings.
# Input: image embeddings and learned word embeddings
# Output: list of top image filenames for each word

import numpy as np
import os
import json
import operator
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_features',
        type=str)

    parser.add_argument(
        '--word2col',
        type=str)

    parser.add_argument(
        '--name',
        type=str)

    parser.add_argument(
        '--word_embeddings',
        type=str)

    parser.add_argument(
        '--image_names_in_order',
        type=str)

    parser.add_argument(
        '--num_top_images',
        type=int,
        default=50)

    return parser.parse_args()

def main():
    args = parse_args()

    save_dir = './results_{}'.format(args.name)
    if not os.path.exists(save_dir):
        print('Results directory {} does not exist.'.format(save_dir))
        exit()
    print('Saving results in: {}'.format(save_dir))

    with open(args.word2col, 'r') as f:
        word_to_col = json.load(f)

    print('Loading list of image names.')
    image_names_in_order = []
    with open(args.image_names_in_order, 'r') as f:
        image_names_in_order = json.load(f)

    # load and normalize image embeddings
    print('Loading image embeddings.')
    image_embeddings = np.load(args.image_features)
    normalizer = 1.0 / np.sqrt(np.sum(image_embeddings ** 2, axis=1))
    image_embeddings *= normalizer[:, np.newaxis]

    num_clusters = len(word_to_col)
    num_images = image_embeddings.shape[0]
    num_dims = image_embeddings.shape[1]
    
    print('Loading word embeddings.')
    word_to_embedding = np.load(args.word_embeddings, mmap_mode='r')
    
    for i, (word, col) in enumerate(sorted(word_to_col.items())):
        if i % 100 == 0:
            print('Saving word {}: {}'.format(i, word))
        word_vector = word_to_embedding[col, :].reshape(1, word_to_embedding.shape[1])
        dists = np.linalg.norm(image_embeddings - word_vector, axis=1)
        sorted_images_and_dists = sorted(zip(dists, image_names_in_order), key=operator.itemgetter(0))
        with open('./{}/{}.json'.format(save_dir, word), 'w') as f:
            json.dump([im for _, im in sorted_images_and_dists[:args.num_top_images]], f, indent=4)





if __name__ == '__main__':
    main()


