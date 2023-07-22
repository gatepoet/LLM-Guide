# Guide: Converting Llama 2 Models for CPU, CUDA GPU, and Intel GPU

This guide will walk you through the process of converting the new Llama 2 models for use with CPU, CUDA GPU, and Intel GPU. It is based on the previous guide found [here](https://raw.githubusercontent.com/gatepoet/LLM-Guide/main/README.md) and additional resources from [Hugging Face](https://huggingface.co/meta-llama) and [MLC LLM](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html).

## Prerequisites

Before you start, ensure that you have TVM Unity and the Wasm Build Environment installed on your machine. 

- [TVM Unity](https://tvm.apache.org/docs/install/index.html) is a compiler stack that enables efficient deployment of deep learning models on a variety of hardware platforms. 
- The [Wasm Build Environment](https://github.com/emscripten-core/emsdk) is necessary if you plan to build for webgpu. 

Additionally, you will need the CLI app that can be used to chat with the compiled model. You can obtain it by following the instructions in the CLI and C++ API section.

## Step 1: Model Download

Hugging Face is a platform that hosts a wide variety of pre-trained models, including the Llama 2 models. If you haven't downloaded the model yet, you can do so directly from Hugging Face using the `--hf-path` option. For example, `--hf-path meta-llama/Llama-2-7b-chat-hf` will download the model from the corresponding Hugging Face repository to the `dist/models/` directory. You can explore more models on the [Hugging Face website](https://huggingface.co/models).

If you have already downloaded the model, ensure that it is located in the `dist/models/` directory before proceeding to the next step.

## Step 2: Model Compilation

Model compilation is a crucial step in preparing the model for use. It involves translating the model into a form that can be efficiently executed on a specific hardware platform.

In this step, you will use the `mlc_llm.build` module to compile the model. The command should specify the model, target platform, and quantization mode.

- The `--model` option specifies the name of the model you want to compile. For example, `--model Llama-2-7b-chat-hf` specifies that you want to compile the Llama-2-7b-chat-hf model.
- The `--target` option specifies the target platform (e.g., `cuda` for CUDA).
- The `--quantization` option specifies the quantization mode (e.g., `q4f16_1` for 4-bit quantization).

For example, to compile the Llama-2-7b-chat-hf model for CUDA, use the following command: 

```bash
python3 -m mlc_llm.build --model Llama-2-7b-chat-hf --target cuda --quantization q4f16_1
```

The `--target` option specifies the target platform (e.g., `cuda` for CUDA). The `--quantization` option specifies the quantization mode (e.g., `q4f16_1` for 4-bit quantization).

## Step 3: Model Validation

After the compilation, you can validate the result by chatting with the model using the CLI app. For example, `mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_1` will start a chat session with the compiled model.

## Step 4: Model Distribution

If you want to distribute the model, refer to the 'Distribute Compiled Models' section.

## Note

Each compilation target produces a specific model library for the given platform. The model weight is shared across different targets. Also, using 3-bit quantization can be overly aggressive and only works for limited settings. If you encounter issues where the compiled model does not perform as expected, consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

## Conclusion

This guide has walked you through the process of converting the new Llama 2 models for use with CPU, CUDA GPU, and Intel GPU. By following these steps, you should be able to compile and use these models effectively.
