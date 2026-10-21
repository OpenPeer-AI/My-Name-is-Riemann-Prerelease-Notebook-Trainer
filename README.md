# My Name is Riemann Prerelease: test it, then train it

This notebook loads the 13.1 billion parameter [My Name is Riemann Prerelease](https://www.kaggle.com/models/openpeer-ai/my-name-is-riemann-prerelease) checkpoint straight from Kaggle Models, runs a handful of tests that show what the model can and cannot do yet, and then fits an NTK gate controller on data you pick. They are loaded in half precision (the model was trained in bf16; the T4s run fp16 natively) and sharded across the two GPUs, with gradient checkpointing so training fits in what is left over.

Before you run it:

1. Settings > Accelerator > **GPU T4 x2**. One T4 is not enough for 26 GB of weights.
2. Settings > Internet > **On** (needed for `pip install` and, optionally, to stream training data from Hugging Face).
3. The notebook looks for the model under `/kaggle/input` first (add it from the Models tab if it is not attached) and otherwise downloads it from Hugging Face, which Kaggle does at several hundred MB/s, so either way it takes a few minutes.

Loading takes about 8 to 12 minutes because the fp32 shards are read from the model mount and cast on the way in. After that each test cell runs in seconds to a minute. The whole notebook end to end is roughly 25 to 35 minutes.

Sections:

* **Load**: tokenizer, model, memory check.
* **Test**: free continuation, a token by token surprisal map, a fluency contrast table, and the stage 6 NTK controller that ships with the checkpoint.
* **Train**: fit your own NTK controller with [ntkmirror](https://github.com/leochlon/ntkmirror) (the OpenPeer AI NTK Trainer). Weights stay frozen; only a few thousand gate values are learned.
* **Optional**: a short LoRA run on the same frozen base, for comparison.

Related: the [Hugging Face model card](https://huggingface.co/OpenPeerAI/My-Name-is-Riemann-Prerelease) with the full training writeup, and three live demos ([basic](https://huggingface.co/spaces/OpenPeerAI/My-Name-is-Riemann-Prerelease-basic-demo), [OpenPeerLLM comparison](https://huggingface.co/spaces/OpenPeerAI/OpenPeerLLM-Demo), [NTK Trainer](https://huggingface.co/spaces/OpenPeerAI/NTK-Trainer-Demo)).

A word on expectations. This is a prerelease that has seen 2.3 billion tokens of pretraining data. It writes fluent text in an encyclopedic register and knows the shape of Python, but it has not been instruction tuned, so it will not chat with you. The tests below are built around what it does do.
