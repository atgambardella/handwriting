# handwriting
Chinese Handwriting Recognition with CNNs

Evaluate a pre-trained model on an image by calling:
```
python convolutional.py --evaluate "my_image1.png my_image2.png ..."
```

# Nota Bene:

I've noticed an increased amount of interest in this repo in the past day, so I thought it might be useful to write a little bit about what this is and how to use it.

This was an assignment in the interview process for my former workplace [Cogent Labs](https://www.cogent.co.jp/). They provided a dataset of Chinese characters which is not publicly available, and my task was to create a classifier for them, and provide a writeup for what I did and why I did it.

This repo is now a year old and was using an early version of Tensorflow. It is almost certain that some parts of the code have broken since then.

If you are willing to fix those issues, what you must next do is acquire a dataset, and replace the function `extract_data_and_labels()` to deal with reading your data and putting them into NumPy vectors.

Other than that, almost everything should still work with possible minor changes. I won't be able to help much if you run into issues, however, as this repo was never really meant to be used by the general public.
