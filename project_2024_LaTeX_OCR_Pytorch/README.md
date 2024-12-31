# LaTeX OCR

A Pytorch implementation of [LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO), forked from [qs956/Latex_OCR_Pytorch](https://github.com/qs956/Latex_OCR_Pytorch)

It uses CNN as the encoder, and RNN as the decoder

## Training

- Prepare the dataset files in `.npy` format, each element of the ndarray is formatted as:

    ```python
    {
        'ID': 1,
        'label': "x ^ { 2 } - 1 3 x + 3 6 < 0",
        'image': np.ndarray of shape (width, height, RGB)
    }
    ```
- Change the `dataset_dir` to the folder holding the `.npy` files
- install extra dependencies

    ```shell
    pip install -r requirements.txt
    ```
- start training

    ```shell
    python train.py
    ```

## Warning

The code is meant for research purpose and is far from production ready.

There're several major drawbacks:
- slow training & inference

    1h/epoch on laptop RTX 2060
- bucketing not implementated, which is presented in the original versions
