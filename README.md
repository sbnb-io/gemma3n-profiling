# Profiling Google Gemma 3n Model Using PyTorch Profiler

## Prerequisites

- Ubuntu 24.04  
- NVIDIA GPU

## Quick Start

```bash
apt-get update && apt-get install -y libgl1 python3-venv
```

```bash
git clone https://github.com/sbnb-io/gemma3n-profiling
cd gemma3n-profiling
python3 -m venv .
. bin/activate
pip3 install -r requirements.txt
```

To update CUDA to version 12.9 (which supports the latest NVIDIA Blackwell 50 series GPUs), run:

```bash
pip3 install --pre torch torchvision torchaudio nvidia-cuda-cupti-cu12==12.9.79 --index-url https://download.pytorch.org/whl/nightly/cu129
```

Set your Hugging Face token:

```bash
export HF_TOKEN=hf_REPLACE_ME
```

To start profiling, run:

```bash
python3 gemma3n-profiling.py
```

## Viewing the Results

A `gemma3n-profiling.json` file will be generated, approximately 80MB in size.  

To visualize the trace, go to [https://ui.perfetto.dev/](https://ui.perfetto.dev/) and select **Open trace file**, pointing to your `gemma3n-profiling.json`.

- Expand the `python3 PID` row to explore the code running on the CPU.  
- Expand the `python3 0` (stream 7 7) row to examine code running on the GPU.

![](media/gemma3n-gpu-utilization.gif)

## Profiling Notes

- The Gemma 3n model is asked to describe this image: [bee.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg).
- We limit generation to 10 tokens to keep the resulting trace file smaller and easier to analyze.
- The script performs two runs and skips the first as a warm-up. The first run takes around 60 seconds, but subsequent runs finish in about 0.4 seconds. If you wish to profile the warm-up run, you can adjust the `warmup` and `active` arguments of `torch.profiler.schedule`.
- For convenience, we have included the resulting `gemma3n-profiling.json` file in this repository, in case you prefer to explore the results without running the setup yourself.
- The exact package versions in our environment:
    ```
    transformers             4.53.1
    timm                     1.0.16
    nvidia-cuda-cupti-cu12   12.9.79
    nvidia-cuda-nvrtc-cu12   12.9.86
    nvidia-cuda-runtime-cu12 12.9.79
    torch                    2.9.0.dev20250706+cu129
    ```
- Profiling was done inside an Ubuntu 24.04.2 LTS virtual machine running on [AI Linux (Sbnb Linux)](https://github.com/sbnb-io/sbnb), with an NVIDIA RTX 5060 Ti 16GB Blackwell GPU.

## Diving Deeper Into the Results

Total runtime measured was 483 milliseconds.  

Initially, the trace shows the `get_image_features` function of Gemma3n ([source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py#L2253)), which then calls `forward_features` in MobileNetV5 ([source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv5.py#L535)), taking about 74 milliseconds.  

Next, a series of `Gemma3nTextDecoderLayer` ([source](https://github.com/huggingface/transformers/blob/ca7e1a3756c022bf31429c452b2f313f043f32de/src/transformers/models/gemma3n/modular_gemma3n.py#L1829)) calls took 142 milliseconds.  

Finally, generating the 10 tokens took approximately 244 milliseconds total, which averages around 24 milliseconds per token.  

Each token generation involves a [`cudaGraphLaunch`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73) (which launches an executable graph in a stream), followed by a [`cudaStreamSynchronize`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e) (which waits for the stream’s tasks to complete).  

The MobileNetV5 and `Gemma3nTextDecoderLayer` phases accounted for around 50% of the total runtime. However, their share would decrease significantly if more tokens are generated. For instance, generating 100 tokens would reduce their share to roughly 10%.

## Open Questions

1. **Potential for Speedup**:  
   The GPU sits idle for around 12 milliseconds after each token is generated. This delay occurs because the CPU is busy with the next call to `prepare_inputs_for_generation`. Could this step be optimized to load the next tasks into the GPU more quickly, improving GPU utilization?

---

**Questions or Suggestions?**  
We recognize this work might be missing critical pieces. We welcome any feedback! Feel free to open an issue or discussion in this repository.
