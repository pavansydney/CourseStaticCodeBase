// ============================================================
// AI Engineering: Zero to Hero — extended course content.
// Loaded only on the Courses page (after script.js). It extends the
// existing global `courseData` object with new tracks, then the shared
// loaders in script.js render them into their grids.
// Module shape matches script.js:
// { number, title, description, duration, lessons, isNew, isLocked,
//   topics: [...], detailedDescription, detailedContent: [{title, content, code}] }
// ============================================================

/* global courseData */

// ---------- Stage 1: Deep Learning ----------
courseData.deepLearning = [
    {
        number: "Module 1",
        title: "Neural Networks",
        description: "How artificial neurons, layers, activations, and backpropagation turn data into predictions.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Neurons & layers", "Activation functions", "Forward pass", "Backpropagation", "Gradient descent"],
        detailedDescription: "Neural networks are the foundation of modern AI. This module builds your intuition from a single neuron up to a trainable multi-layer network, then shows the same network in PyTorch.",
        detailedContent: [
            {
                title: "From a neuron to a network",
                content: `A neuron computes a weighted sum of its inputs plus a bias, then passes the result through a non-linear activation function.

<strong>output = activation( w1*x1 + w2*x2 + ... + b )</strong>

Stacking neurons into <strong>layers</strong>, and layers into a <strong>network</strong>, lets the model learn increasingly abstract features: early layers detect simple patterns, later layers combine them into complex concepts. The "deep" in deep learning simply means many layers.`,
                code: ""
            },
            {
                title: "Activation functions & why non-linearity matters",
                content: `Without a non-linear activation, stacking layers collapses into a single linear function — no matter how deep. Non-linearity is what gives networks their expressive power.

<strong>Common choices:</strong>
• <strong>ReLU</strong> max(0, x) — the default for hidden layers; fast and avoids vanishing gradients.
• <strong>Sigmoid</strong> — squashes to (0,1); used for binary outputs.
• <strong>Softmax</strong> — turns logits into a probability distribution for multi-class output.`,
                code: `import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),   # input layer -> hidden
    nn.ReLU(),             # non-linearity
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)      # 10-class output (logits)
)`
            },
            {
                title: "Learning = forward pass + backpropagation",
                content: `Training repeats a simple loop:

1. <strong>Forward pass:</strong> run inputs through the network to get predictions.
2. <strong>Loss:</strong> measure how wrong the predictions are.
3. <strong>Backpropagation:</strong> use the chain rule to compute how each weight contributed to the error (the gradient).
4. <strong>Gradient descent:</strong> nudge each weight slightly in the direction that reduces the loss.

Repeat over many batches and the network gradually improves.`,
                code: `import torch

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in train_loader:
    optimizer.zero_grad()
    preds = model(x)          # forward pass
    loss = loss_fn(preds, y)  # how wrong?
    loss.backward()           # backprop -> gradients
    optimizer.step()          # gradient descent update`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Training Neural Networks",
        description: "Loss functions, optimizers, overfitting, and the regularization techniques that make models generalize.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Loss functions", "Optimizers", "Overfitting vs underfitting", "Regularization & dropout", "Train/val/test splits"],
        detailedDescription: "A model that memorizes the training set is useless. This module covers how to train networks that generalize to unseen data.",
        detailedContent: [
            {
                title: "Loss functions & optimizers",
                content: `The <strong>loss function</strong> defines what "good" means. Pick it to match the task:
• <strong>Cross-entropy</strong> for classification.
• <strong>MSE / MAE</strong> for regression.

The <strong>optimizer</strong> decides how to update weights from gradients. <strong>Adam</strong> is a strong default (adaptive per-parameter learning rates). The <strong>learning rate</strong> is the single most important hyperparameter — too high diverges, too low crawls.`,
                code: ""
            },
            {
                title: "Overfitting and how to fight it",
                content: `<strong>Overfitting</strong>: the model does great on training data but poorly on new data — it memorized noise. <strong>Underfitting</strong>: the model is too simple to capture the pattern.

<strong>Tools to generalize:</strong>
• <strong>More/augmented data</strong> — the most reliable fix.
• <strong>Dropout</strong> — randomly zero activations during training so the network can't rely on any single path.
• <strong>Weight decay (L2)</strong> — penalize large weights.
• <strong>Early stopping</strong> — stop when validation loss stops improving.`,
                code: `model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Dropout(0.3),               # regularization
    nn.Linear(128, 10)
)
# weight decay adds L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`
            },
            {
                title: "Splitting data honestly",
                content: `Always split data into three sets:
• <strong>Train</strong> — the model learns from this.
• <strong>Validation</strong> — you tune hyperparameters against this.
• <strong>Test</strong> — touched only once, to report final performance.

If you tune on the test set, your reported accuracy is a lie. Keep the test set locked away until the very end.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 3",
        title: "Transformers & Attention",
        description: "The architecture behind every modern LLM: self-attention, tokens, and why transformers replaced RNNs.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Tokens & embeddings", "Self-attention", "Multi-head attention", "Positional encoding", "Encoder vs decoder"],
        detailedDescription: "Transformers power GPT, Gemini, Claude, and Llama. Understand attention and you understand what LLMs actually do under the hood.",
        detailedContent: [
            {
                title: "Why attention beat recurrence",
                content: `Older sequence models (RNNs, LSTMs) processed text one token at a time, struggling with long-range dependencies and being slow to train.

The <strong>Transformer</strong> (2017, "Attention Is All You Need") processes all tokens in parallel and lets every token directly "look at" every other token via <strong>self-attention</strong>. This scales beautifully on GPUs and captures long-range context — the key unlock behind large language models.`,
                code: ""
            },
            {
                title: "Self-attention intuition",
                content: `For each token, attention produces three vectors: <strong>Query</strong>, <strong>Key</strong>, and <strong>Value</strong>. A token's new representation is a weighted blend of all tokens' Values, where the weights come from how well its Query matches each Key.

In plain terms: each word decides <em>which other words are relevant to it</em> and pulls in their meaning. "It" learns to attend to the noun it refers to. <strong>Multi-head</strong> attention runs several of these in parallel to capture different relationships.`,
                code: `import torch, torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    weights = F.softmax(scores, dim=-1)   # who attends to whom
    return weights @ V                    # blended values`
            },
            {
                title: "Tokens, positions, and model families",
                content: `Text is first split into <strong>tokens</strong> (sub-words) and mapped to <strong>embeddings</strong>. Since attention has no built-in order, <strong>positional encodings</strong> inject where each token sits.

<strong>Three families:</strong>
• <strong>Encoder-only</strong> (BERT) — understanding tasks like classification.
• <strong>Decoder-only</strong> (GPT, Llama) — text generation; predicts the next token. This is what most LLMs are.
• <strong>Encoder-decoder</strong> (T5) — translation and summarization.`,
                code: ""
            }
        ]
    }
];

// ---------- Stage 2: Generative AI ----------
courseData.generativeAI = [
    {
        number: "Module 1",
        title: "Generative AI",
        description: "What generative AI is, how it differs from traditional ML, and the landscape of models and use cases.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Discriminative vs generative", "Foundation models", "Modalities", "Capabilities & limits"],
        detailedDescription: "Generative AI creates new content — text, images, audio, and code. This module frames the field before you dive into LLMs and agents.",
        detailedContent: [
            {
                title: "Generative vs discriminative",
                content: `<strong>Discriminative</strong> models draw boundaries — "is this spam or not?" <strong>Generative</strong> models learn the underlying distribution well enough to <em>produce new samples</em> — a paragraph, an image, a function.

Modern generative AI is built on <strong>foundation models</strong>: very large models pre-trained on massive datasets, then adapted to many tasks via prompting, RAG, or fine-tuning.`,
                code: ""
            },
            {
                title: "Modalities and the model landscape",
                content: `Generative AI spans modalities:
• <strong>Text</strong> — GPT, Gemini, Claude, Llama.
• <strong>Images</strong> — diffusion models (Stable Diffusion, Imagen).
• <strong>Audio/Speech</strong> — TTS and speech-to-text.
• <strong>Code</strong> — Copilot-style assistants.
• <strong>Multimodal</strong> — models that mix text, images, and audio.

As an AI engineer you'll mostly compose these via APIs rather than train them from scratch.`,
                code: ""
            },
            {
                title: "Capabilities and honest limitations",
                content: `LLMs are excellent at summarizing, drafting, extracting, translating, and reasoning over provided context. But they:
• <strong>Hallucinate</strong> — produce confident, wrong answers.
• Have a <strong>knowledge cutoff</strong> and no live data unless you give it.
• Are <strong>stateless</strong> between calls unless you manage memory.

The rest of this track is largely about engineering around these limits with grounding, tools, and memory.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 2",
        title: "Large Language Models",
        description: "How LLMs generate text: tokens, context windows, temperature, and calling a model via an API.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Next-token prediction", "Context window", "Temperature & sampling", "System/user/assistant roles", "Structured output"],
        detailedDescription: "Before building agents, you must be fluent with the raw material: the LLM API. This module covers how generation works and how to control it.",
        detailedContent: [
            {
                title: "Next-token prediction & the context window",
                content: `An LLM generates text one token at a time, each time predicting the most likely next token given everything so far. Loop that and you get sentences.

The <strong>context window</strong> is the maximum number of tokens the model can consider at once (input + output). Everything the model "knows" for a request must fit inside it — this is why long documents get chunked and retrieved (RAG).`,
                code: ""
            },
            {
                title: "Controlling generation",
                content: `Key knobs:
• <strong>temperature</strong> — 0 is deterministic/focused; higher is more creative/random.
• <strong>top_p</strong> — nucleus sampling; another diversity control.
• <strong>max_tokens</strong> — caps output length.

Messages use <strong>roles</strong>: a <strong>system</strong> prompt sets behavior, <strong>user</strong> messages are requests, and <strong>assistant</strong> messages are the model's replies (and prior turns).`,
                code: `from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain embeddings in one sentence."}
    ],
)
print(resp.choices[0].message.content)`
            },
            {
                title: "Structured output",
                content: `For real applications you rarely want free-form prose — you want JSON your code can use. Ask the model for a schema and validate it. Most providers support a JSON / structured-output mode that guarantees parseable results.

Treat model output as untrusted input: validate it, and never directly execute it without checks.`,
                code: `resp = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Return JSON: {sentiment, confidence}."},
        {"role": "user", "content": "I love this product!"}
    ],
)
import json
data = json.loads(resp.choices[0].message.content)`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Prompt Engineering",
        description: "Practical techniques — zero/few-shot, chain-of-thought, and role prompting — to get reliable results.",
        duration: "40 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Clear instructions", "Few-shot examples", "Chain-of-thought", "Role & format control", "Guardrails"],
        detailedDescription: "Prompt engineering is the cheapest, fastest way to improve LLM output. Learn the patterns that consistently work.",
        detailedContent: [
            {
                title: "Be specific, show the format",
                content: `The biggest wins come from clarity:
• State the <strong>task, audience, and constraints</strong> explicitly.
• Show the <strong>exact output format</strong> you want (a template or JSON schema).
• Give the model a <strong>role</strong>: "You are a senior data engineer reviewing SQL."

Vague prompts get vague answers. Specific prompts get usable ones.`,
                code: ""
            },
            {
                title: "Few-shot and chain-of-thought",
                content: `<strong>Zero-shot</strong>: just ask. <strong>Few-shot</strong>: include 2-5 worked examples so the model mimics the pattern — great for consistent formatting or edge cases.

<strong>Chain-of-thought</strong>: for reasoning tasks, ask the model to work step by step before answering. It improves accuracy on math and logic. For clean output, have it reason internally and then return only the final structured answer.`,
                code: `prompt = """Classify the ticket as: billing, bug, or feature.

Example 1: "I was charged twice" -> billing
Example 2: "The app crashes on login" -> bug

Ticket: "Can you add dark mode?" ->"""`
            },
            {
                title: "Guardrails in the prompt",
                content: `Prompts also encode safety and reliability:
• Tell the model what to do when it <strong>doesn't know</strong> ("say 'I don't know' rather than guess").
• Constrain scope ("only answer from the provided context").
• Beware <strong>prompt injection</strong>: untrusted text (web pages, user data) can contain instructions. Never blindly trust retrieved content, and keep system instructions separate from user data.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 4",
        title: "Embeddings & Vector Databases",
        description: "Turn text into vectors to power semantic search — the retrieval half of RAG and agent memory.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["What embeddings are", "Cosine similarity", "Chunking", "Vector databases", "Semantic search"],
        detailedDescription: "Embeddings map meaning to geometry. This module is the backbone of retrieval, RAG, and long-term memory.",
        detailedContent: [
            {
                title: "Meaning as vectors",
                content: `An <strong>embedding model</strong> converts text into a fixed-length vector (e.g., 1536 numbers). Texts with similar meaning land close together in that space, even if they share no words.

"How do I reset my password?" and "I forgot my login" will have nearby vectors. This is what makes <strong>semantic search</strong> possible — matching by meaning, not keywords.`,
                code: `from openai import OpenAI
client = OpenAI()

def embed(text):
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

v1 = embed("How do I reset my password?")
v2 = embed("I forgot my login")`
            },
            {
                title: "Measuring similarity & chunking",
                content: `Closeness is usually measured with <strong>cosine similarity</strong> (1 = identical direction, 0 = unrelated).

Long documents are split into <strong>chunks</strong> (e.g., a few hundred tokens with slight overlap) before embedding, so retrieval returns focused, relevant passages rather than whole documents. Chunking strategy strongly affects quality.`,
                code: `import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine(v1, v2))  # high -> semantically similar`
            },
            {
                title: "Vector databases",
                content: `A <strong>vector database</strong> (Pinecone, Weaviate, pgvector, Chroma, FAISS) stores millions of embeddings and answers "find the k most similar vectors" in milliseconds using approximate nearest-neighbor search.

The pattern: embed your knowledge base once, store the vectors, then at query time embed the question and retrieve the closest chunks. That retrieved context feeds the LLM.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 5",
        title: "Retrieval-Augmented Generation (RAG)",
        description: "Ground LLMs in your own data to reduce hallucination and answer from up-to-date, private sources.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Why RAG", "Ingest & index", "Retrieve & augment", "Citations", "RAG vs fine-tuning"],
        detailedDescription: "RAG is the most common production pattern for LLM apps. Learn to build a pipeline that answers from your documents.",
        detailedContent: [
            {
                title: "The RAG pattern",
                content: `RAG has two phases:

<strong>Indexing (offline):</strong> load documents → chunk → embed → store in a vector DB.

<strong>Query (online):</strong> embed the user's question → retrieve the top-k relevant chunks → stuff them into the prompt as context → the LLM answers <em>from that context</em>.

This grounds answers in your data, keeps them current, and lets you show sources — without retraining the model.`,
                code: ""
            },
            {
                title: "A minimal RAG query",
                content: `The core of query-time RAG is: retrieve, then augment the prompt. Always instruct the model to answer <em>only</em> from the retrieved context and to admit when the answer isn't there — this is what cuts hallucination.`,
                code: `def rag_answer(question, vector_store):
    chunks = vector_store.similarity_search(question, k=4)
    context = "\\n\\n".join(c.text for c in chunks)
    prompt = f"""Answer using ONLY the context. If it's not there, say you don't know.

Context:
{context}

Question: {question}"""
    return llm(prompt), chunks   # return sources for citations`
            },
            {
                title: "RAG vs fine-tuning",
                content: `Choose deliberately:
• <strong>RAG</strong> — best for <em>knowledge</em> that changes or is private (docs, tickets, policies). Cheap to update: just re-index.
• <strong>Fine-tuning</strong> — best for <em>behavior/style/format</em> the model should always follow.

They're complementary: fine-tune <em>how</em> the model responds, use RAG for <em>what</em> facts it responds with.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 6",
        title: "Fine-Tuning LLMs",
        description: "When and how to adapt a model's behavior with techniques like LoRA and instruction tuning.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["When to fine-tune", "Data preparation", "LoRA / PEFT", "Evaluation"],
        detailedDescription: "Fine-tuning specializes a model to your task or style. Learn when it's worth it and how modern parameter-efficient methods make it practical.",
        detailedContent: [
            {
                title: "When fine-tuning is (and isn't) the answer",
                content: `Reach for fine-tuning when you need a consistent <strong>style, format, or behavior</strong>, want to <strong>compress a long system prompt</strong> into the weights, or need a smaller/cheaper model to match a bigger one on a narrow task.

Do <em>not</em> fine-tune to add changing facts — use RAG. Always try strong prompting and RAG first; they're faster and cheaper.`,
                code: ""
            },
            {
                title: "Data is everything",
                content: `Fine-tuning quality is dominated by data quality. You need a clean dataset of input → ideal output examples that demonstrate exactly the behavior you want. A few hundred excellent examples beat thousands of noisy ones.

Format is usually chat-style JSONL: each row a conversation with the desired assistant response.`,
                code: `# Example JSONL row for instruction tuning
{"messages": [
  {"role": "system", "content": "You extract invoice fields as JSON."},
  {"role": "user", "content": "Invoice #A-102, total $59.90, due 2026-08-01"},
  {"role": "assistant", "content": "{\\"id\\":\\"A-102\\",\\"total\\":59.90,\\"due\\":\\"2026-08-01\\"}"}
]}`
            },
            {
                title: "LoRA & parameter-efficient tuning",
                content: `Full fine-tuning updates all weights — expensive. <strong>LoRA</strong> (Low-Rank Adaptation) freezes the base model and trains tiny adapter matrices, cutting compute and memory dramatically while keeping quality. This is the standard for open models (via Hugging Face PEFT).

Always evaluate on a held-out set and compare against your prompt/RAG baseline before shipping.`,
                code: ""
            }
        ]
    }
];

// ---------- Stage 3: AI Agents ----------
courseData.aiAgents = [
    {
        number: "Module 1",
        title: "Agentic AI",
        description: "What makes AI 'agentic': the perceive-reason-act loop, autonomy, and the ReAct pattern.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Agents vs pipelines", "The agent loop", "ReAct (reason + act)", "Autonomy levels", "When to use agents"],
        detailedDescription: "Agentic AI moves from single prompts to systems that plan, act, observe, and iterate toward a goal. This module sets the mental model for the whole stage.",
        detailedContent: [
            {
                title: "From pipelines to agents",
                content: `A <strong>pipeline</strong> follows a fixed sequence of steps you wrote. An <strong>agent</strong> uses an LLM to decide the steps at runtime: it chooses which tool to call, reads the result, and decides what to do next — looping until the goal is met.

Agentic systems trade predictability for flexibility. Use them when the path can't be hard-coded; use plain pipelines when it can.`,
                code: ""
            },
            {
                title: "The agent loop & ReAct",
                content: `Most agents run a loop:

<strong>Thought → Action → Observation → (repeat) → Answer</strong>

This is the <strong>ReAct</strong> pattern (Reason + Act): the model reasons about what to do, takes an action (calls a tool), observes the result, and reasons again. The loop continues until it decides it has enough to answer.`,
                code: `# Conceptual agent loop
goal = "What's the weather in Tokyo in Fahrenheit?"
while not done:
    thought, action = llm_decide(goal, history)   # reason + pick a tool
    observation = run_tool(action)                # act
    history.append((thought, action, observation))
    done = is_goal_met(history)
answer = llm_finalize(goal, history)`
            },
            {
                title: "Levels of autonomy",
                content: `Agency is a spectrum:
• <strong>Assisted</strong> — model suggests, human approves each step.
• <strong>Supervised</strong> — agent acts, human reviews checkpoints.
• <strong>Autonomous</strong> — agent runs end-to-end.

Start low. For anything with side effects (sending email, spending money, writing to prod), keep a human in the loop and add hard limits.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 2",
        title: "AI Agents",
        description: "The anatomy of an agent: the LLM brain, tools, memory, and a planner working together.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["LLM as controller", "Tools", "Memory", "Planning", "Failure handling"],
        detailedDescription: "An agent is more than an LLM. This module breaks down the components you'll assemble to build reliable agents.",
        detailedContent: [
            {
                title: "The four components",
                content: `A capable agent combines:
• <strong>Brain (LLM)</strong> — decides and reasons.
• <strong>Tools</strong> — functions/APIs that let it act (search, DB, code, HTTP).
• <strong>Memory</strong> — short-term (the current conversation) and long-term (facts recalled across sessions, often via a vector store).
• <strong>Planner</strong> — breaks a goal into steps, sometimes revising as it learns.`,
                code: ""
            },
            {
                title: "Tools are the agent's hands",
                content: `An LLM alone can only produce text. <strong>Tools</strong> give it the ability to affect and read the world. You describe each tool (name, purpose, parameters) and the model chooses when to call it. Good tool design — clear names, tight schemas, helpful errors — matters more than prompt wording for reliability.`,
                code: `tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}]`
            },
            {
                title: "Designing for failure",
                content: `Agents fail in ways pipelines don't: infinite loops, wrong tool choices, hallucinated arguments. Build in guardrails:
• <strong>Max steps</strong> / iteration budget.
• <strong>Validation</strong> of tool arguments before executing.
• <strong>Timeouts and retries</strong>.
• A <strong>fallback</strong> ("if stuck, ask the user").

Reliability engineering is most of what separates a demo agent from a production one.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 3",
        title: "Tool & Function Calling",
        description: "The mechanism that lets LLMs invoke your code — the foundation of every agent.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Tool schemas", "The call/execute loop", "Returning results", "Security"],
        detailedDescription: "Function calling is how an LLM turns intent into action. Master this and agents become straightforward.",
        detailedContent: [
            {
                title: "How function calling works",
                content: `You send the model a list of available tools with JSON schemas. Instead of answering directly, the model can respond with a <strong>tool call</strong>: the function name plus arguments it invented from the conversation. <em>Your code</em> executes the function and sends the result back. The model then continues, using the result.

The model never runs code itself — it only requests calls; you stay in control of execution.`,
                code: ""
            },
            {
                title: "The execute-and-return loop",
                content: `The full loop: send messages + tools → if the model returns tool calls, run them → append the results as tool messages → call the model again → repeat until it returns a normal answer.`,
                code: `def run_weather(city):
    return {"temp_c": 21, "city": city}

messages = [{"role": "user", "content": "Weather in Tokyo?"}]
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
call = resp.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)
result = run_weather(**args)                       # you execute
messages.append(resp.choices[0].message)
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(result)})   # return result
final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)`
            },
            {
                title: "Treat tool calls as untrusted",
                content: `The model can hallucinate arguments or be manipulated via prompt injection. Before executing:
• <strong>Validate</strong> arguments against the schema and business rules.
• <strong>Scope</strong> tools with least privilege (read-only where possible).
• Require <strong>human approval</strong> for destructive or costly actions.
• <strong>Log</strong> every call for audit.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 4",
        title: "Agent Memory",
        description: "Give agents short-term and long-term memory so they stay coherent and personalized.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Short-term vs long-term", "Conversation buffers", "Vector memory", "Summarization"],
        detailedDescription: "LLMs are stateless. This module covers the memory patterns that make agents feel continuous and context-aware.",
        detailedContent: [
            {
                title: "Two kinds of memory",
                content: `<strong>Short-term memory</strong> is the running conversation held in the context window — it vanishes when the window fills or the session ends.

<strong>Long-term memory</strong> persists across sessions: user preferences, past facts, prior decisions. It's typically stored outside the model (a database or vector store) and retrieved when relevant, then injected into the prompt.`,
                code: ""
            },
            {
                title: "Vector memory in practice",
                content: `Long-term memory usually reuses the embeddings + vector DB pattern: write important facts as embedded notes; at each turn, retrieve the most relevant memories and add them to the context. This is essentially RAG applied to the agent's own history.`,
                code: `def remember(text, store):        # write
    store.add(embed(text), text)

def recall(query, store, k=3):    # read relevant memories
    return store.search(embed(query), k)

context_memories = recall(user_message, store)`
            },
            {
                title: "Managing a growing context",
                content: `As conversations grow, you can't keep everything. Strategies:
• <strong>Buffer window</strong> — keep only the last N turns.
• <strong>Summarization</strong> — periodically compress old turns into a short summary.
• <strong>Selective recall</strong> — retrieve only memories relevant to the current query.

Good memory management keeps agents coherent without blowing the token budget.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 5",
        title: "How to Build an Agent",
        description: "Put it together: build a working tool-using agent loop from scratch, then know when to reach for a framework.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Define the goal & tools", "The control loop", "Stopping conditions", "From scratch vs framework", "Testing agents"],
        detailedDescription: "A capstone for this stage: assemble the brain, tools, and loop into an agent you fully understand before adopting LangChain or LangGraph.",
        detailedContent: [
            {
                title: "Design first",
                content: `Before code, answer:
• <strong>Goal</strong> — what does 'done' look like?
• <strong>Tools</strong> — the minimal set of actions needed.
• <strong>Constraints</strong> — max steps, cost, what needs human approval.
• <strong>Output</strong> — the shape of the final answer.

Fewer, well-designed tools beat many overlapping ones.`,
                code: ""
            },
            {
                title: "A minimal from-scratch agent",
                content: `The whole agent is a loop around function calling with a step budget and a clear stopping condition. Understanding this ~20 lines makes every framework feel familiar.`,
                code: `def agent(goal, tools, tool_impls, max_steps=6):
    messages = [{"role": "user", "content": goal}]
    for _ in range(max_steps):
        r = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content                 # done
        for call in msg.tool_calls:            # act
            args = json.loads(call.function.arguments)
            out = tool_impls[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(out)})
    return "Stopped: step budget exceeded."`
            },
            {
                title: "From scratch vs framework",
                content: `Building from scratch teaches you the mechanics and gives full control. Reach for a <strong>framework</strong> (next stage) when you need prebuilt tool integrations, streaming, persistence, retries, tracing, and multi-agent orchestration without reinventing them.

Test agents like software: unit-test tools, and use scripted scenarios plus an LLM-as-judge to catch regressions.`,
                code: ""
            }
        ]
    }
];

// ---------- Stage 4: Agent Frameworks & Systems ----------
courseData.agentFrameworks = [
    {
        number: "Module 1",
        title: "LangChain",
        description: "The most popular framework for LLM apps: models, prompts, tools, chains, and agents.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Core abstractions", "LCEL & chains", "Tools & agents", "Integrations", "When to use it"],
        detailedDescription: "LangChain gives you batteries-included building blocks for LLM applications. Learn its model of the world and build a tool-using agent.",
        detailedContent: [
            {
                title: "What LangChain gives you",
                content: `LangChain standardizes the pieces of an LLM app behind common interfaces: <strong>chat models</strong>, <strong>prompt templates</strong>, <strong>output parsers</strong>, <strong>tools</strong>, <strong>retrievers</strong>, and <strong>memory</strong> — plus hundreds of integrations (OpenAI, Anthropic, vector DBs, APIs).

Its value is composability and ecosystem: swap a model or vector store without rewriting your app.`,
                code: ""
            },
            {
                title: "Composing with LCEL",
                content: `The <strong>LangChain Expression Language</strong> (LCEL) pipes components together with the | operator: prompt | model | parser. This makes chains readable, streamable, and easy to modify.`,
                code: `from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Summarize in one line: {text}")
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
print(chain.invoke({"text": "LangChain composes LLM building blocks."}))`
            },
            {
                title: "Tools and agents in LangChain",
                content: `You define tools with a decorator, bind them to a model, and let LangChain run the call/execute loop. For anything stateful or multi-step, LangChain now points you to <strong>LangGraph</strong> (next module) for more control over the agent's flow.`,
                code: `from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    "Get current weather for a city."
    return f"21C and sunny in {city}"

llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])`
            }
        ]
    },
    {
        number: "Module 2",
        title: "LangGraph",
        description: "Build reliable, stateful agents as graphs with branches, cycles, persistence, and human-in-the-loop.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Graphs vs chains", "State", "Nodes & edges", "Cycles & control", "Checkpointing"],
        detailedDescription: "LangGraph models agents as state machines/graphs — the production-grade way to get control, durability, and human oversight.",
        detailedContent: [
            {
                title: "Why a graph",
                content: `Linear chains struggle with loops, branching, and recovery. <strong>LangGraph</strong> represents an agent as a <strong>graph</strong>: <strong>nodes</strong> do work (call the LLM, run a tool), <strong>edges</strong> decide what runs next, and a shared <strong>state</strong> object flows through.

Because edges can loop back, you get the agent's think-act-observe cycle explicitly and controllably — plus durability and human-in-the-loop.`,
                code: ""
            },
            {
                title: "State, nodes, and edges",
                content: `You define a typed <strong>state</strong>, add <strong>nodes</strong> (functions that read/update state), and connect them with edges. <strong>Conditional edges</strong> route based on state — e.g., "if the model asked for a tool, go to the tool node; else finish."`,
                code: `from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class State(TypedDict):
    messages: List[dict]

def call_model(state): ...      # returns {"messages": [...]}
def call_tools(state): ...

g = StateGraph(State)
g.add_node("model", call_model)
g.add_node("tools", call_tools)
g.set_entry_point("model")
g.add_conditional_edges("model", needs_tool, {"yes": "tools", "no": END})
g.add_edge("tools", "model")    # loop back
app = g.compile()`
            },
            {
                title: "Persistence & human-in-the-loop",
                content: `LangGraph can <strong>checkpoint</strong> state after every step. That enables:
• <strong>Durability</strong> — resume after a crash.
• <strong>Human-in-the-loop</strong> — pause before a risky action, let a human approve, then continue.
• <strong>Time travel</strong> — rewind and branch.

These features are exactly what production agents need and are painful to build by hand.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 3",
        title: "MCP Servers",
        description: "The Model Context Protocol: a standard way to expose tools, data, and prompts to any AI app.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["What MCP is", "Client-server model", "Tools, resources, prompts", "Building a server", "Why it matters"],
        detailedDescription: "MCP is an open standard (from Anthropic) that lets AI applications connect to tools and data through a uniform interface — 'USB-C for AI tools'.",
        detailedContent: [
            {
                title: "The problem MCP solves",
                content: `Every AI app used to integrate tools and data its own way — an N×M mess. <strong>MCP (Model Context Protocol)</strong> standardizes it: build a tool/integration <em>once</em> as an MCP <strong>server</strong>, and any MCP-compatible <strong>client</strong> (Claude Desktop, IDEs, your own agent) can use it.

Think of it as a universal adapter between AI models and the outside world.`,
                code: ""
            },
            {
                title: "Servers, clients, and primitives",
                content: `An MCP <strong>server</strong> exposes three things:
• <strong>Tools</strong> — actions the model can invoke (like function calling).
• <strong>Resources</strong> — data the model can read (files, DB rows, API responses).
• <strong>Prompts</strong> — reusable prompt templates.

The <strong>client</strong> (inside the AI app) discovers and calls these over a simple protocol (stdio or HTTP). The model's host wires it all together.`,
                code: ""
            },
            {
                title: "A minimal MCP server",
                content: `With the Python SDK, you declare tools with a decorator and run the server. Any MCP client can then discover and call your tool — no custom integration per app.`,
                code: `from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    "Return current weather for a city."
    return f"21C and sunny in {city}"

if __name__ == "__main__":
    mcp.run()   # exposes the tool over MCP`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Multi-Agent Orchestration",
        description: "Coordinate multiple specialized agents — supervisors, handoffs, and crews — to solve complex tasks.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Why multiple agents", "Supervisor pattern", "Handoffs", "Shared state", "Frameworks (CrewAI, AutoGen)"],
        detailedDescription: "Some problems are better split across specialists. This module covers the patterns and pitfalls of multi-agent systems.",
        detailedContent: [
            {
                title: "When one agent isn't enough",
                content: `A single agent with too many tools and responsibilities gets confused. <strong>Multi-agent</strong> systems assign narrow roles — e.g., a Researcher, a Coder, and a Reviewer — each with focused tools and instructions. Specialization improves reliability and makes the system easier to reason about.

But coordination adds cost and latency: only go multi-agent when a single agent genuinely struggles.`,
                code: ""
            },
            {
                title: "Orchestration patterns",
                content: `Common topologies:
• <strong>Supervisor / router</strong> — a lead agent decides which specialist handles each subtask and combines results.
• <strong>Handoff</strong> — agents pass control to one another (like a support escalation).
• <strong>Pipeline</strong> — output of one agent feeds the next.
• <strong>Debate/review</strong> — agents critique each other to improve quality.

A shared <strong>state</strong> or message bus lets them collaborate.`,
                code: `# Supervisor pattern (pseudocode)
def supervisor(task):
    plan = llm_plan(task)                 # break into subtasks
    results = []
    for step in plan:
        agent = pick_agent(step.role)     # researcher / coder / reviewer
        results.append(agent.run(step))
    return llm_combine(task, results)`
            },
            {
                title: "Frameworks & pitfalls",
                content: `<strong>LangGraph</strong>, <strong>CrewAI</strong>, and <strong>AutoGen</strong> provide multi-agent primitives (roles, handoffs, shared memory). Watch for common failure modes: runaway loops between agents, ballooning token cost, and agents that confidently agree on a wrong answer. Add step budgets, clear termination, and evaluation just as with single agents.`,
                code: ""
            }
        ]
    },
    {
        number: "Module 5",
        title: "Evaluating & Deploying Agents",
        description: "Take agents to production: evaluation, tracing, cost/latency control, safety, and monitoring.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Evaluation & LLM-as-judge", "Tracing & observability", "Cost & latency", "Guardrails & safety", "Monitoring"],
        detailedDescription: "The last mile. This capstone covers what it takes to run agents reliably and responsibly in production.",
        detailedContent: [
            {
                title: "Evaluating non-deterministic systems",
                content: `Agents don't have a single right output, so testing differs from normal software:
• Build a <strong>dataset</strong> of representative tasks with expected outcomes or rubrics.
• Use <strong>LLM-as-judge</strong> to score responses against criteria (correctness, grounding, tone).
• Track <strong>task success rate</strong>, not just token-level metrics.
• Re-run this suite on every prompt/model change to catch regressions.`,
                code: ""
            },
            {
                title: "Observability, cost, and latency",
                content: `You can't fix what you can't see. Use <strong>tracing</strong> (LangSmith, OpenTelemetry) to record every step, tool call, and token. In production, watch:
• <strong>Cost</strong> — tokens per request; cache and use smaller models where possible.
• <strong>Latency</strong> — stream responses, parallelize tool calls, cap steps.
• <strong>Failure rate</strong> — retries, timeouts, and graceful fallbacks.`,
                code: ""
            },
            {
                title: "Safety & monitoring",
                content: `Ship responsibly:
• <strong>Guardrails</strong> — input/output filtering, PII redaction, allow-listed tools.
• <strong>Least privilege</strong> and human approval for high-impact actions.
• <strong>Prompt-injection defenses</strong> — treat all external content as untrusted.
• <strong>Monitoring & feedback</strong> — log outcomes, collect user feedback, and continuously improve your eval set.

Congratulations — that's the full path from a single neuron to production AI agents.`,
                code: ""
            }
        ]
    }
];
