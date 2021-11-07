# Transformers From Scratch

Transformers have disrupted the field of Deep Learning in both NLP and Vision. It has become a general computation neural network which can be applied almost anywhere. While applying them to any use case has become easier than ever with libraries like [Hugging Face Transformers](https://huggingface.co/transformers/), I wanted to dig deeper and understand all the nitty gritties of how they work. In this repository I learn to implement Transformers from Scratch using PyTorch.

## Usage

To test the transformers implementation on a toy example of reversing a sequence checkout the [toy_example.py](toy_example.py) script which contains example code for everything that you would need to train transformers for a sequence-to-sequence task.

- [Sample training loop snippet](toy_example.py#L26)
- [Auto-Regressive Inference snippet](toy_example.py#L61)

Apart from the toy example, a practical application of transformers is demonstrated in [Transliteration.ipynb](Transliteration.ipynb) notebook which trains a transliteration model on a small sample of 32 examples. This notebook can be used as a reference implementation for your own seq2seq projects.

## Notes

- The usage examples can be used with the [official PyTorch implementation of Transformers](https://pytorch.org/docs/stable/nn.html#transformer-layers) as well with a few changes like providing the source and target masks externally instead of it getting generated automatically as in this implementation.
- If you have any questions or want to discuss something about the implementation, feel free to open an issue in this repository.

## References

This implementation is based on the wonderful [Tutorial by Aladdin Persson](https://www.youtube.com/watch?v=U0s0f995w14). Do give it an upvote if you decide to watch it as well.
