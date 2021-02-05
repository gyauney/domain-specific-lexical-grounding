# domain-specific-lexical-grounding

Code for [Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents][] (EMNLP 2020).

[Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents]: https://arxiv.org/abs/2010.16363

Requirements for EntSharp: `numpy`, `scipy`.
Requirements for data preparation: `numpy`, `tensorflow`.

If you find the paper or code useful, please consider citing:

```
@inproceedings{yauney2020domainspecific,
  title={Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents},
  author={Yauney, Gregory and Hessel, Jack and Mimno, David},
  booktitle={EMNLP},
  year={2020}
}
```

What's in this repository:
- [How to run `EntSharp` on your dataset](#how-to-run-entsharp-on-your-dataset)
- [Reproducing the paper's results on multimodal Wikipedia](#reproducing-the-papers-results-on-multimodal-wikipedia)
- [Sample StreetEasy documents](#sample-streeteasy-documents)


---

## How to run `EntSharp` on your dataset

This section will show you how to convert data into the required format for `EntSharp` so you can run the algorithm as a baseline on your own data!

Data should be organized into documents, where each document has a unique ID and contains both words and images.

### 1. Files you will need

You will need your dataset in the following format.
Each file's name is arbitrary, but the later steps will use these names to refer to each file.
The example commands in the next steps assume these files are all in a directory named `data`,
e.g. the path to the vocabulary list would be `./data/vocab.json`.


- Vocabulary: `vocab.json`

JSON list of the words in your dataset. An embedding will be produced for each word.

e.g. Here's a list of four of the words used in the paper:
```
['a', 'ability', 'able', 'abode']
```

- List of image names: `image-names.txt`

A list of paths to each image in the dataset, each on its own line. They can be absolute or relative paths. If relative, then make sure they are relative to where you will run the feature extraction script in the next step.

e.g. Here's a list containing three images that are in a subdirectory named `images`:
```
data/images/323943925.jpg
data/images/323943924.jpg
data/images/323943923.jpg
```

- Images in each document: `doc-to-images.json`

JSON dictionary mapping from document ID to a list of image names (without file extension) in the document.

e.g. Here's a dictionary with a single document with ID `"1349631"`. This document contains three images:
```
{
    "1349631": [
        "323943925",
        "323943924",
        "323943923"
    ]
}
```

- Text in each document: `doc-to-words.json`

JSON dictionary mapping from document ID to a list of words (including duplicates) in the document.

e.g. Here's a dictionary that only contains a document with ID `"1349631"`, which contains six words:
```
{
    "1349631": [
        "situated",
        "in",
        "the",
        "heart",
        "of",
        "greenpoint"
    ]
}
```


### 2. Image feature extraction

We use the DenseNet169 image feature extraction script from [Unsupervised Discovery of Multimodal Links in Multi-image/Multi-sentence Documents][], a.k.a. `multi-retrieval`, by Hessel, Lee, & Mimno (EMNLP 2019).

First, clone the [`multi-retrieval` repository][]. Install `tensorflow` if necessary. You should only need `tensorflow` rather than all of its requirements.

Run the following three commands from this `domain-specific-lexical-grounding` directory after replacing both instances of <i>[path to multi-retrieval/]</i> with the correct path to your `multi-retrieval` directory:
<pre>
python3 <i>[path to multi-retrieval/]</i>/image_feature_extract/extract.py data/image-names.txt data/extracted_features
python3 <i>[path to multi-retrieval/]</i>image_feature_extract/make_python_image_info.py data/extracted_features data/image-names.txt
mv id2row.json data/image-to-row.json
</pre>

**Output:**
- `extracted_features.npy`: a `numpy` array where each row contains the 1,664-dimensional extracted features for one image.
Dimensions: (number of images) times (1,664 features). 
- `image-to-row.json`: a JSON dict mapping from image name to row in `extracted_features.npy`.

[Unsupervised Discovery of Multimodal Links in Multi-image/Multi-sentence Documents]: https://arxiv.org/abs/1904.07826
[`multi-retrieval` repository]: https://github.com/jmhessel/multi-retrieval

### 3. Data pre-processing: image-word co-occurrences and random projection

This step gets all the data into the final format for `EntSharp` by
a) constructing an image-word co-occurrence matrix and
b) [randomly projecting][] the image features down from the 1,664-dimensional DenseNet169 feature-space to 256 dimensions, greatly increasing the speed of `EntSharp`.

[randomly projecting]: https://en.wikipedia.org/wiki/Random_projection

Run:
```
python3 prepare-data.py \
     --output_dir data \
     --vocab data/vocab.json \
     --image_features data/extracted_features.npy \
     --doc2images data/doc-to-images.json \
     --doc2words data/doc-to-words.json \
     --image2row data/image-to-row.json \
     --sparse_mask
```

**Output:**
- `image-membership-mask_sparse.npz`: a (sparse) image-word co-occurrence `numpy` array. Dimensions: (number of images) times (number of words).
An entry `(i,j)` is 1 if image `i` and word `j` co-occur in any document and 0 otherwise.

Excluding the `--sparse_mask` flag in the above command will produce a dense `numpy` array instead.
A dense matrix takes up much more disk space but is faster to load because it can be memory-mapped.

- `word-to-col.json`: JSON dictionary mapping each word in the vocabulary to its column in the above co-occurrence matrix.

- `extracted_features_random-projection-256.npy`: Reduced dimensionality image feature `numpy` array.
Dimensions: (number of images) times (256 features).

### 4. Run `EntSharp`

You can now run `EntSharp`, the soft-clustering algorithm from the paper:

```
python3 entsharp.py \
    --output_dir results \
    --image_features data/extracted_features_random-projection-256.npy \
    --image_word_cooccurrences data/image-membership-mask_sparse.npz \
    --n_iterations 100
```

**Output:**
- `word-embeddings.npy`: a `numpy` array of learned word embeddings in image space. Each row corresponds to a word.
Dimensions: (number of words) times (number of dimensions in the image embedding).
- Optional: `word-embeddings_average-zero-iter.npy`:
Only saved if you include the flag `--average_baseline`. This file contains an array of word embeddings like `word-embeddings.npy`, but it is the zero-iteration baseline described in the paper.

### 5. Find the closest images to each learned word embedding

This is the step that produces lexical groundings!

```
python3 rank-images-per-word.py \
    --output_dir results \
    --image_features data/extracted_features_random-projection-256.npy \
    --word_embeddings results/word-embeddings.npy \
    --word2col data/word-to-col.json \
    --image2row data/image-to-row.json \
    --num_top_images 50
```

**Output:**
One JSON list of top image filenames in order for each word.
For example, the images closest to an "architect" word embedding would be in a file named `architect.json`.

---

## Reproducing the paper's results on [multimodal Wikipedia][]

1. Download the [pre-processed dataset][] (600 MB) and move the extracted directory into this directory.

[multimodal Wikipedia]: https://jmhessel.com/projects/concreteness/concreteness.html
[pre-processed dataset]: https://drive.google.com/file/d/1gdhRi7MRvPQJEkLcoGTWrOeg-0i_ouOE/view?usp=sharing

2. Run `EntSharp` on the Wikipedia dataset:

```
python3 entsharp.py \
    --output_dir results_wiki \
    --image_features multimodal-wikipedia-preprocessed/wiki_extracted-features_random-projection-100.npy \
    --image_word_cooccurrences multimodal-wikipedia-preprocessed/wiki_image-membership-mask_sparse.npz \
    --n_iterations 100
```

**Output:**
- `word-embeddings.npy`: Embeddings .

3. You can get the closest images for each learned word embedding by running:

```
python3 rank-images-per-word.py \
    --output_dir results_wiki \
    --image_features multimodal-wikipedia-preprocessed/wiki_extracted-features_random-projection-100.npy \
    --word_embeddings results_wiki/word-embeddings.npy \
    --word2col multimodal-wikipedia-preprocessed/wiki_word-to-col.json \
    --image_names_in_order multimodal-wikipedia-preprocessed/wiki_image-names-in-order.json \
    --num_top_images 50 
```

This will create one JSON list per word, where entries are filenames in Wikimedia.
To view an image, append its filename to this URL: `https://commons.wikimedia.org/wiki/File:`

For example, the URL for an image from the file `architect.json` is:
`https://commons.wikimedia.org/wiki/File:Chateau-de-versailles-cour.jpg`

---

## Sample StreetEasy documents

100 sample documents from the StreetEasy dataset used in the paper are available for download [here][] (400 MB).
These include documents with images used in the paper as well as documents selected uniformly at random.
Running `EntSharp` on just these documents will produce different results from those in the paper
because these are only a subset of the paper's dataset.

[here]: https://drive.google.com/file/d/1LCYCfU7LKKYfDWSFnQ8vHFYcxnRMu0uN/view?usp=sharing

Summary statistics:
- Documents: 100
- Unique word types: 2,665
- Total word tokens: 14,845
- Total images: 1,431

List of files included:
- `vocab.json`: JSON list of the 7,971 word vocabulary used in the paper. Not all of these words appear in this released dataset.
- `streeteasy_doc-to-words.json`: JSON dictionary mapping from document ID to a list of words in the document.
Documents have undergone preprocessing, including punctuation removal and lowercasing.
- `streeteasy_doc-to-images.json`: JSON dictionary mapping from document ID to a list of image names in the document.
- `images`: Directory containing the image files.
- `streeteasy_image-names.txt`: Newline-separated list of relative paths to the images, e.g. `images/330008062.jpg`.

---

## Baselines

Code is forthcoming for the baselines in the paper.

