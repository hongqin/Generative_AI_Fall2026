# Week 6, Topic 4: Practical Use of Large-Model Workflows in Applied Settings

## Topic Overview

**Plain-English summary first.** Most people using transformers today are not training them from scratch. They are using someone else’s pre-trained model and adapting it to a specific task or domain. This Topic surveys the modern practical toolkit: where to find pre-trained models (Hugging Face), what your options are for adapting them (prompting, retrieval-augmented generation, fine-tuning, parameter-efficient fine-tuning like LoRA), when to call a hosted API vs self-host, and how agent frameworks layer task-execution on top of base LLMs. This is the workflow most of your course projects will use; the previous three Topics built the conceptual foundation, and this one is the applied vocabulary.

### The pre-trained model ecosystem

The single most important resource in modern ML is the [Hugging Face Hub](https://huggingface.co/), an open registry of pre-trained models. As of 2026 it hosts over a million models across text, vision, audio, multimodal, and protein domains. Each model has a *model card* documenting: the architecture, the training data, the license, the evaluation benchmarks, and known limitations. The `transformers` library (also from Hugging Face) gives you a uniform PyTorch/TensorFlow API for loading and running almost any of them.

A minimal text-generation example:

```
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("The quick brown fox", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Three lines to download a 124M-parameter GPT-2 and use it. The same pattern works for billion-parameter models with a bigger machine.

### Four ways to adapt a pre-trained model

#### 1. Prompting (no training)

The cheapest option. Write a prompt that elicits the behavior you want. For instruction-tuned models (GPT-4, Claude, Llama-Instruct), this is often enough. Variations:

- **Zero-shot prompting**: just ask. “Translate this to French: ...”
- **Few-shot prompting**: provide examples in the prompt. “Translate: dog =\> chien. cat =\> chat. bird =\> ?”
- **Chain-of-thought prompting**: ask the model to reason step by step before answering. Improves performance on math and reasoning tasks.

Cost: API tokens (or local inference time). No training. Iteration is fast (change the prompt, re-run).

#### 2. Retrieval-Augmented Generation (RAG)

When the model lacks domain knowledge or is hallucinating facts, give it the relevant context in the prompt. Standard RAG architecture: maintain a vector database of your documents (embedded with a sentence-embedding model); at query time, retrieve the top-k relevant documents and put them in the LLM’s context window along with the user query. The LLM’s answer is grounded in retrieved facts rather than its training data.

Cost: vector-database infrastructure + embedding inference + per-query LLM call. No model training. Standard libraries: LangChain, LlamaIndex, Haystack.

#### 3. Parameter-efficient fine-tuning (PEFT)

When prompting and RAG are not enough but you cannot afford full fine-tuning, train only a small number of additional parameters. The dominant technique is **Low-Rank Adaptation (LoRA)** (Hu et al., 2021): freeze the pre-trained weights, add small trainable low-rank update matrices to selected layers. If the original layer is $`W \in \mathbb{R}^{d \times d}`$, the LoRA update factorizes the change as $`\Delta W = B A`$ where $`B \in \mathbb{R}^{d \times r}`$ and $`A \in \mathbb{R}^{r \times d}`$, with rank $`r`$ much smaller than $`d`$ (typical $`r = 8`$ to $`64`$):

``` math
W_{\text{adapted}} \;=\; W \,+\, \Delta W \;=\; W \,+\, B A, \qquad r \,\ll\, d.
```

Training only $`B`$ and $`A`$ means ~1% as many trainable parameters as full fine-tuning, fits on a single GPU even for large models, and the resulting LoRA weights are tiny (a few MB) so you can have many task-specific adapters for one base model. The standard library is Hugging Face `peft`. **QLoRA** (Dettmers et al., 2023) adds 4-bit quantization to the base model, making it possible to fine-tune even 70B-parameter models on a single 24GB GPU.

#### 4. Full fine-tuning

Update all model weights on your task-specific data. Most expensive, most flexible. Necessary if your domain is far from the pre-training distribution or if you need many task-specific behaviors integrated into one model. For all but the most demanding applications, LoRA is the better starting point.

### Hosted APIs vs self-hosting

Two ways to actually run a pre-trained model in production:

- **Hosted API**: call OpenAI, Anthropic, Google, AWS Bedrock, Together, Replicate. Pay per token. No infrastructure. State-of-the-art models accessible immediately. Tradeoffs: data privacy (your prompts go to a third party), API rate limits, vendor lock-in, no fine-tuning on closed models (you can do prompt engineering and sometimes light fine-tuning via the vendor’s API).
- **Self-hosting**: download an open-weights model (Llama, Mistral, Qwen, Gemma, etc.), run it on your own GPUs. You control the data and the deployment; you pay for hardware and operational complexity. Tools: vLLM, TGI (Text Generation Inference), Ollama, llama.cpp. ODU’s Wahab cluster is suitable for self-hosting smaller open models (7B to 30B parameters); larger models need multi-GPU setups.

For course projects: hosted APIs are often the right choice for iteration speed. Self-hosting is the right choice if your project explicitly involves the model weights themselves (fine-tuning experiments, mechanistic interpretability, etc.).

### Agent frameworks (forward reference)

An *agent* is an LLM that interacts with tools (APIs, web search, code execution, databases, other LLMs) in a multi-step loop to accomplish a task that requires more than one LLM call. Standard frameworks: LangChain, LlamaIndex, OpenAI Assistants API, Anthropic’s Claude Agent SDK. The agent typically:

1.  Receives a user request.
2.  Plans the steps needed.
3.  Calls a tool (search, code, retrieval).
4.  Observes the result.
5.  Decides the next step. Iterates 3-5 until done.
6.  Returns a final answer.

This is the “Agentic AI” territory that the second half of the syllabus title points to. We will not cover agents in depth in this course, but for your project: an agent built on top of a strong pre-trained model is often the most natural way to deliver a useful end-to-end system that does something specific in your domain.

### Deployment considerations

If your project involves an actual user-facing system (even just for a demo), at minimum consider:

- **Latency budget**: what is the acceptable response time? Determines model size, batching, hardware.
- **Cost per request**: token costs for APIs; amortized hardware costs for self-hosting.
- **Hallucination evaluation**: how do you know when the model is making things up? Critical for any factual application.
- **Safety and content filtering**: what behavior do you NOT want, and how do you prevent it?
- **Monitoring and logging**: capture queries and outputs (with privacy considerations) so you can diagnose failures and improve the system.

### Where this Topic fits

The Module 6 coding exercise asks you to pick between training Foster’s small GPT from scratch (close to Topic 3) and a Hugging Face pre-trained workflow (close to this Topic). Module 7 covers World Models, the first model-based reinforcement-learning generative system, and brings the Recorded Paper Presentation milestone you set up in Module 1. Module 8 concludes with multimodal models and the Final Project Report.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 9 closing sections on practical use and fine-tuning.
- Hugging Face Transformers documentation, *Quick Tour*: [huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour). Spend 30 minutes; this is the API you will likely use in your project.

Recommended (the LoRA paper):

- E. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

### Consider

- For your course project, which of the four adaptation strategies (prompting, RAG, LoRA fine-tuning, full fine-tuning) is the natural starting point, and what would make you escalate to a more expensive option?
- Hosted APIs are fast to iterate with but expose your prompts to a third party. For your project domain, is that an acceptable tradeoff? If not, what is the smallest open-weights model that could plausibly handle the task?
- Agents add a loop of tool calls on top of a base LLM. Where does the line between “a clever prompt” and “an actual agent” really live, in your view?

### Resources

- Hugging Face Hub: [huggingface.co](https://huggingface.co/). The pre-trained model registry.
- Hugging Face `peft` library: [huggingface.co/docs/peft](https://huggingface.co/docs/peft). LoRA, prefix-tuning, adapter methods.
- T. Dettmers et al., “QLoRA: Efficient Finetuning of Quantized LLMs,” NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). 4-bit quantized fine-tuning.
- LangChain documentation: [python.langchain.com](https://python.langchain.com/). Standard agent / RAG framework.
