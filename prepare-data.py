# prepare-data.py
#
# Generate consistent word order, word-image co-occurrence matrix,
# and randomly project image features.
# Input: data files in the format specified in the readme
#        extracted image features
# Output: sparse image-word co-occurrence matrix
#         JSON dict from word to column in the above matrix
#         randomly-projected image features

import argparse
import numpy as np
import scipy.sparse
import os
import json
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_dir',
        type=str)

    parser.add_argument(
        '--vocab',
        type=str)

    parser.add_argument(
        '--doc2images',
        type=str)

    parser.add_argument(
        '--doc2words',
        type=str)

    parser.add_argument(
        '--image2row',
        type=str)

    parser.add_argument(
        '--image_features',
        type=str)

    parser.add_argument(
        '--sparse_mask',
        default=False,
        action='store_true')

    return parser.parse_args()

# randomly project image embeddings
# from a 1664-dimensional-space down to 256 dimensions
def random_projection(image_features):

    reduced_dims = 256
    new_fn = '{}_random-projection-{}.npy'.format(image_features[:-4], reduced_dims)

    print('Loading image embeddings.')
    image_embeddings = np.load(image_features, mmap_mode='r')
    print('Image embedding shape:', image_embeddings.shape)
    num_images = image_embeddings.shape[0]
    num_dims = image_embeddings.shape[1]

    print('Getting random basis vectors.')
    # V random 256-dimensional directions
    # from gaussian with 0 mean and 1 variance
    random_basis = np.zeros((num_dims, reduced_dims))
    for i in range(num_dims):
        random_basis[i,:] = np.random.randn(reduced_dims)
    print('Random basis shape (old dims x new dims): ', random_basis.shape)

    # project the documents into the lower-dimensional space
    print('Projecting.')
    embeddings_projected = np.matmul(image_embeddings, random_basis)

    print('Normalizing rows.')
    normalizer = 1.0 / np.sqrt(np.sum(embeddings_projected ** 2, axis=1))
    embeddings_projected *= normalizer[:, np.newaxis]
    print('Saving low-rank projection!')
    np.save(new_fn, embeddings_projected)


def main():

    args = parse_args()

    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(args.vocab, 'r') as f:
        vocab = set(json.load(f))
    print('Size of vocabulary: %d' % len(vocab))

    with open(args.doc2words, 'r') as f:
        doc_to_words = json.load(f)
        # remove a bunch of non-alphanumeric characters and empties
        doc_to_words = {doc: [w.replace('‘', '')
                               .replace('’', '')
                               .replace('“', '')
                               .replace('”', '')
                               .replace('…', '')
                              for w in words if w.replace('‘', '')
                                                 .replace('’', '')
                                                 .replace('“', '')
                                                 .replace('”', '')
                                                 .replace('…', '') != '']
                              for doc, words in doc_to_words.items()}
        # filter out words not in the vocabulary
        doc_to_words = {doc: [w for w in words if w in vocab]
                             for doc, words in doc_to_words.items()}
    num_tokens = sum([len(ws) for ws in doc_to_words.values()])

    with open(args.image2row, 'r') as f:
        image_name_to_row = json.load(f)

    num_images = len(image_name_to_row)
    # we need to map rows to images as well
    image_row_to_name = {row: name for name, row in image_name_to_row.items()}
    image_names_in_order = [image_row_to_name[i] for i in range(num_images)]

    print('Number of documents: {}'.format(len(doc_to_words)))
    print('Number of tokens: {}'.format(num_tokens))
    print('Number of images: {}'.format(num_images))

    with open(args.doc2images, 'r') as f:
        doc_to_images = json.load(f)
        # filter out the few images that don't have extracted features
        doc_to_images = {doc: [im for im in ims if im in image_name_to_row] for doc, ims in doc_to_images.items()}

    # filter out docs with 0 images
    docs_to_remove = []
    for doc, images in doc_to_images.items():
        if len(images) == 0:
            docs_to_remove.append(doc)
    print("Docs with no images: %d" % len(docs_to_remove))
    assert len(docs_to_remove) == 0

    unique_words = sorted(list(vocab))
    num_clusters = len(unique_words)
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    print('Number of unique words / clusters = %d' % num_clusters)

    # get the words each image co-occurs with (these are fixed from documents)
    # run once! and save
    print('Getting word counts for each image as a dict.')
    image_name_to_word_counts = defaultdict(lambda: defaultdict(int))
    for doc, images in doc_to_images.items():
        this_doc_to_word_counts = defaultdict(int)
        # get counts for this doc
        for word in doc_to_words[doc]:
            this_doc_to_word_counts[word] = this_doc_to_word_counts[word] + 1
        # update all co-occurring images by merging this doc with existing counts
        for image in images:
            current = image_name_to_word_counts[image]
            for word, count in this_doc_to_word_counts.items():
                image_name_to_word_counts[image][word] = current[word] + count

    print('Constructed image membership mask:')
    image_membership_mask = np.zeros((num_images, num_clusters))
    for num, (image, word_counts) in enumerate(image_name_to_word_counts.items()):
        if num % 50000 == 0:
            print('    {}/{} images done so far.'.format(num, num_images))
        image_row = image_name_to_row[image]
        for word, count in word_counts.items():
            word_col = word_to_index[word]
            image_membership_mask[image_row][word_col] = 1

    # save everything!
    with open('{}/word-to-col.json'.format(save_dir), 'w') as f:
        json.dump(word_to_index, f, indent=4)

    # sparse matrix has a much smaller size
    # but saving the dense enables memory mapping later for much faster load times
    if args.sparse_mask:
        print('Saving sparse image membership mask.')
        sparse_mask = scipy.sparse.coo_matrix(image_membership_mask)
        scipy.sparse.save_npz('{}/image-membership-mask_sparse.npz'.format(save_dir), sparse_mask)
    else:
        print('Saving dense image membership mask.')
        np.save('{}/image-membership-mask.npy'.format(save_dir), image_membership_mask)

    # finally randomly project the image features
    random_projection(args.image_features)
    

if __name__ == '__main__':
    main()