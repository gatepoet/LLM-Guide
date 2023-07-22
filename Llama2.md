# Guide: Converting Llama 2 Models for CPU, CUDA GPU, and Intel GPU

This guide will walk you through the process of converting the new Llama 2 models for use with CPU, CUDA GPU, and Intel GPU. It is based on the previous guide found [here](https://raw.githubusercontent.com/gatepoet/LLM-Guide/main/README.md) and additional resources from [Hugging Face](https://huggingface.co/meta-llama) and [MLC LLM](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html).

## Prerequisites

Before you start, ensure that you have TVM Unity and the Wasm Build Environment installed on your machine. 

- [TVM Unity](https://tvm.apache.org/docs/install/index.html) is a compiler stack that enables efficient deployment of deep learning models on a variety of hardware platforms. 
- The [Wasm Build Environment](https://github.com/emscripten-core/emsdk) is necessary if you plan to build for webgpu. 

Additionally, you will need the CLI app that can be used to chat with the compiled model. You can obtain it by following the instructions in the CLI and C++ API section.

## Model Download

Hugging Face is a platform that hosts a wide variety of pre-trained models, including the Llama 2 models. If you haven't downloaded the model yet, you can do so directly from Hugging Face using the `--hf-path` option. For example, `--hf-path meta-llama/Llama-2-7b-chat-hf` will download the model from the corresponding Hugging Face repository to the `dist/models/` directory. You can explore more models on the [Hugging Face website](https://huggingface.co/models).

If you have already downloaded the model, ensure that it is located in the `dist/models/` directory before proceeding to the next step.

## Model Compilation

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

## Model Validation

Model validation is an essential step that ensures the compiled model is functioning as expected. It involves interacting with the model and evaluating its performance.

After the compilation, you can validate the result by chatting with the model using the CLI app. This will allow you to assess the model's performance and ensure it's ready for use. For example, `mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_1` will start a chat session with the compiled model.

## Model Distribution

Model distribution involves preparing the compiled model for use in other systems or applications. This includes packaging the model and any necessary dependencies in a format that can be easily imported and used elsewhere.

If you want to distribute the model, you will need to package it appropriately and ensure that it can be easily integrated into the target system or application. For detailed instructions on how to do this, please refer to the 'Distribute Compiled Models' section.

## Model Deployment

Model deployment involves integrating the distributed model into a system or application where it can be used to make predictions. This includes loading the model, setting up any necessary interfaces for input and output, and ensuring that the model performs as expected in the target environment.

If you want to deploy the model, you will need to integrate it into your target system or application and ensure that it functions correctly. For detailed instructions on how to do this, please refer to the 'Deploy Compiled Models' section.

## Model Use

Using the model involves interacting with the deployed model to make predictions. This includes providing input to the model, processing the model's output, and using the predictions in a meaningful way.

If you want to use the model, you will need to interact with it in your target system or application and ensure that the predictions are useful and accurate. For detailed instructions on how to do this, please refer to the 'Use Compiled Models' section.

## Distribute Compiled Models

Distributing the compiled Llama 2 models involves packaging the models and any necessary dependencies in a format that can be easily imported and used in other systems or applications. Here's a step-by-step guide on how to do this:

1. **Prepare the Llama 2 Model**: Ensure that the Llama 2 model has been compiled successfully and is ready for distribution. The compiled model should include the model weights, model library, and a chat config. For example, if you've compiled the Llama 2 model for the CUDA target with 4-bit quantization, you should have a directory like `dist/Llama-2-7b-chat-hf-q4f16_0` containing the model library (`Llama-2-7b-chat-hf-q4f16_0-cuda.so`), model weights (`params_shard_*.bin`), and chat config (`mlc-chat-config.json`).

2. **Package the Llama 2 Model**: The Llama 2 model and its dependencies should be packaged in a way that can be easily imported and used in other systems or applications. This includes the model weights, model library, and chat config. You can zip the `dist/Llama-2-7b-chat-hf-q4f16_0` directory to create a package that can be easily distributed.

3. **Test the Llama 2 Model**: Before distributing the Llama 2 model, it's important to test it in the target environment to ensure that it functions correctly. This can be done by loading the model in the target system or application and making some predictions. For example, you can use the `mlc_chat_cli` command-line tool to chat with the compiled model.

4. **Distribute the Llama 2 Model**: Once the Llama 2 model has been packaged and tested, it can be distributed. The exact method of distribution will depend on the specific requirements of the target system or application. For example, you can upload the zip package to a file sharing service or a package registry.

For more detailed instructions on how to distribute compiled models, please refer to the [MLC LLM documentation](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html).

## Deploy Compiled Models

Deploying the compiled Llama 2 models involves integrating the models into a system or application where they can be used to make predictions. Here's a step-by-step guide on how to do this:

1. **Load the Llama 2 Model**: After distributing the model, the first step in deployment is to load the model into the target system or application. This can be done using the `mlc_chat_cli` command-line tool. For example, you can use the following command to load the model: `mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_0`. The CLI will use the config file `dist/Llama-2-7b-chat-hf-q4f16_0/params/mlc-chat-config.json` and model library `dist/Llama-2-7b-chat-hf-q4f16_0/Llama-2-7b-chat-hf-q4f16_0-cuda.so`.

2. **Set Up Interfaces**: Depending on the target system or application, you may need to set up interfaces for input and output. This could involve setting up a user interface for inputting prompts and displaying responses, or setting up an API for programmatically sending prompts and receiving responses.

3. **Test the Llama 2 Model**: After loading the model and setting up interfaces, it's important to test the model to ensure that it performs as expected. This can be done by sending some prompts to the model and evaluating the responses.

For more detailed instructions on how to deploy compiled models, please refer to the [MLC LLM documentation](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html).

## Troubleshooting

Troubleshooting involves identifying and resolving issues that may arise when compiling, distributing, and deploying the Llama 2 models. Here's a list of common issues and their solutions:

1. **Issue: Compilation fails**
   Solution: Ensure that you have correctly installed TVM Unity, which is a necessary foundation for compiling models with MLC LLM. If you want to build for WebGPU, please also complete the installation of the Wasm Build Environment.

2. **Issue: Model performance is not as expected**
   Solution: The quantization mode used during compilation can significantly impact the model's performance. Using a 3-bit quantization can be overly aggressive and only works for limited settings. If you encounter issues where the compiled model does not perform as expected, consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

3. **Issue: Errors when loading the model**
   Solution: Ensure that the model weights, model library, and chat config are correctly packaged and distributed. The model library should match the target platform, and the model weights should be compatible with the quantization mode used during compilation.

For more detailed troubleshooting guidance, please refer to the [MLC LLM documentation](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html).

## Additional Resources

Additional resources can provide further information and help you better understand and work with the Llama 2 models. Here's a list of additional resources:

1. [Meta Llama 2 Page](https://ai.meta.com/llama/): This is the official page for the Llama 2 models from Meta. It provides a comprehensive overview of the models, including technical details, partnerships, responsibility, and resources.

2. [Llama 2 Technical Overview](https://ai.meta.com/llama/#technical-details): This section of the Meta Llama 2 page provides all the technical information you need to use the Llama 2 models.

3. [Research Paper](https://ai.meta.com/llama/#research-paper): This research paper provides the details on the research behind the Llama 2 models.

4. [Responsible Use Guide](https://ai.meta.com/llama/#responsible-use-guide): This guide provides best practices and considerations for building products powered by large language models (LLMs) in a responsible manner.

5. [Open Innovation AI Research Community](https://ai.meta.com/llama/#open-innovation-ai-research-community): This community was established to promote transparency and collaboration between academic partners doing LLM research.

Please note that the exact resources may vary depending on the specific topics you are interested in.

## Conclusion

In conclusion, this guide provides a comprehensive overview of how to compile, distribute, and deploy the Llama 2 models. It also provides troubleshooting tips and links to additional resources. We encourage you to experiment with the Llama 2 models and contribute to the community. Happy modeling!
