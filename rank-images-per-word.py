# rank-images-per-word.py
# 
# Process EntSharp results by calculating pairwise distances between word and image embeddings.
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
        '--output_dir',
        type=str)

    parser.add_argument(
        '--word_embeddings',
        type=str)

    parser.add_argument(
        '--image2row',
        type=str)

    # for wikipedia results only
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

    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving results in: {}'.format(save_dir))

    with open(args.word2col, 'r') as f:
        word_to_col = json.load(f)

    print('Loading image names.')
    image_names_in_order = []
    if args.image_names_in_order == None and args.image2row == None:
        print('Error: Need to specify either --image_names_in_order or --image2row.')
        exit()
    elif args.image_names_in_order != None and args.image2row != None:
        print('Error: Need to specify only one of either --image_names_in_order or --image2row.')
        exit()
    elif args.image_names_in_order != None and args.image2row == None:
        # for reproducing wikipedia results
        with open(args.image_names_in_order, 'r') as f:
            image_names_in_order = json.load(f)
    elif args.image_names_in_order == None and args.image2row != None:
        with open(args.image2row, 'r') as f:
            image_to_index = json.load(f)
        index_to_image = {i: im for im, i in image_to_index.items()}
        image_names_in_order = [index_to_image[i] for i in range(len(index_to_image))]

    # load and normalize image embeddings
    print('Loading image embeddings.')
    image_embeddings = np.load(args.image_features, mmap_mode='r')

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


