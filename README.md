# domain-specific-lexical-grounding

Code for [Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents][] (EMNLP 2020).

[Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents]: https://arxiv.org/abs/2010.16363

Requirements: `numpy`, `scipy`.

If you find the paper or code useful, please consider citing:

```
@inproceedings{yauney2020domainspecific,
  title={Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents},
  author={Yauney, Gregory and Hessel, Jack and Mimno, David},
  booktitle={EMNLP},
  year={2020}
}
```

---

## Reproducing the paper's results on [multimodal Wikipedia][]

1. Download the [pre-processed dataset][] (600 MB) and place the extracted folder in this directory.

[multimodal Wikipedia]: https://jmhessel.com/projects/concreteness/concreteness.html
[pre-processed dataset]: https://drive.google.com/file/d/1gdhRi7MRvPQJEkLcoGTWrOeg-0i_ouOE/view?usp=sharing

2. Run `EntSharp` on the Wikipedia dataset:

```
python3 entsharp.py \
    --name wiki \
    --image_features multimodal-wikipedia-preprocessed/wiki_extracted-features_random-projection-100.npy \
    --image_word_cooccurrences multimodal-wikipedia-preprocessed/wiki_image-membership-mask_sparse.npz \
    --n_iterations 100
```

3. You can get the closest images for each learned word embedding by running:

```
python3 wiki-rank-images-per-word.py \
    --name wiki \
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

## Baselines and data processing

Code is forthcoming for a) the baselines in the paper and b) converting data into the required format for `EntSharp` so you can easily run the algorithm on your own data!
