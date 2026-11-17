# Whisper

Speech recognition with Whisper in MLX. Whisper is a set of open source speech
recognition models from OpenAI, ranging from 39 million to 1.5 billion
parameters.[^1]

### Serving fork

The `serving` branch is based on the source used for `mlx-whisper` 0.4.3
(commit `c1af8c46`). It imports word-alignment code only when a transcription
actually requests word timestamps. Ordinary transcription can therefore start
without importing the alignment module and its Numba/SciPy dependency chain.
The declared dependencies, inference computations, and alignment algorithm are
unchanged. A function-local import keeps the change compatible with the
upstream Python versions; it does not require global lazy-import mode.

Attention uses MLX's fused kernel (`mx.fast.scaled_dot_product_attention`)
instead of separate `q @ k^T`, float32 softmax and `w @ v`; the attention
logits stay a lazy value that only word timestamps read, so decoding no longer
rescales and transposes the cached cross-attention keys at every step.
large-v3 FP16 on an M5 Ultra, temperature 0: 1.26-1.31x faster (LibriSpeech
100 files 28.9 -> 22.8 s, AISHELL-1 100 files 26.6 -> 21.1 s, 16-minute
recording 20.0 -> 15.4 s); 261 of 262 transcripts identical, the other has the
same word error rate.

The first word-timestamp request still pays the alignment initialization cost. Import regression
checks run without model downloads:

```sh
cd whisper
python -m unittest test_lazy_alignment -v
```

Full model comparisons and measurements are maintained by the serving deployment
repository. A change being present on this branch does not imply an end-to-end
performance improvement has been measured for every environment.

### Setup

Install [`ffmpeg`](https://ffmpeg.org/):

```
# on macOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

Install the `mlx-whisper` package with:

```
pip install mlx-whisper
```

### Run

#### CLI

At its simplest:

```sh
mlx_whisper audio_file.mp3
```

This will make a text file `audio_file.txt` with the results.

Use `-f` to specify the output format and `--model` to specify the model. There
are many other supported command line options. To see them all, run
`mlx_whisper -h`.

You can also pipe the audio content of other programs via stdin:

```sh
some-process | mlx_whisper -
```

The default output file name will be `content.*`. You can specify the name with
the `--output-name` flag.

#### API

Transcribe audio with:

```python
import mlx_whisper

text = mlx_whisper.transcribe(speech_file)["text"]
```

The default model is "mlx-community/whisper-tiny". Choose the model by
setting `path_or_hf_repo`. For example:

```python
result = mlx_whisper.transcribe(speech_file, path_or_hf_repo="models/large")
```

This will load the model contained in `models/large`. The `path_or_hf_repo` can
also point to an MLX-style Whisper model on the Hugging Face Hub. In this case,
the model will be automatically downloaded. A [collection of pre-converted
Whisper
models](https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc)
are in the Hugging Face MLX Community.

The `transcribe` function also supports word-level timestamps. You can generate
these with:

```python
output = mlx_whisper.transcribe(speech_file, word_timestamps=True)
print(output["segments"][0]["words"])
```

To see more transcription options use:

```
>>> help(mlx_whisper.transcribe)
```

### Converting models

> [!TIP]
> Skip the conversion step by using pre-converted checkpoints from the Hugging
> Face Hub. There are a few available in the [MLX
> Community](https://huggingface.co/mlx-community) organization.

To convert a model, first clone the MLX Examples repo:

```
git clone https://github.com/ml-explore/mlx-examples.git
```

Then run `convert.py` from `mlx-examples/whisper`. For example, to convert the
`tiny` model use:

```
python convert.py --torch-name-or-path tiny --mlx-path mlx_models/tiny
```

Note you can also convert a local PyTorch checkpoint which is in the original
OpenAI format.

To generate a 4-bit quantized model, use `-q`. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_models`
and save the converted `weights.npz` and `config.json` there.

Each time it is run, `convert.py` will overwrite any model in the provided
path. To save different models, make sure to set `--mlx-path` to a unique
directory for each converted model. For example:

```bash
model="tiny"
python convert.py --torch-name-or-path ${model} --mlx-path mlx_models/${model}_fp16
python convert.py --torch-name-or-path ${model} --dtype float32 --mlx-path mlx_models/${model}_fp32
python convert.py --torch-name-or-path ${model} -q --q_bits 4 --mlx-path mlx_models/${model}_quantized_4bits
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2212.04356), [blog post](https://openai.com/research/whisper), and [code](https://github.com/openai/whisper) for more details.
