// ============================================================
// AI Engineering: Zero to Hero — premium deep rewrite
// Loaded on Courses page after script.js
// ============================================================

/* global courseData */

// ---------- Stage 1: Deep Learning ----------
courseData.deepLearning = [
    {
        number: "AIE - Module 1",
        title: "Neural Network Foundations and Training Intuition",
        description: "Build strong first-principles understanding of neurons, activations, loss surfaces, and gradient-based learning.",
        duration: "120 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Perceptrons", "Activation functions", "Forward and backward pass", "Optimization", "Initialization", "Generalization"],
        detailedDescription: "This module turns neural networks from black boxes into inspectable systems with clear mathematical and engineering behavior.",
        detailedContent: [
            {
                title: "Lesson 1: From Linear Models to Deep Networks",
                content: `Learning Objective: Explain why stacked non-linear layers can represent complex functions that linear models cannot.
Core Theory: A linear model learns one global affine mapping. Deep networks compose multiple affine transformations with non-linear activations, enabling hierarchical feature learning. Early layers learn low-level patterns while later layers combine them into semantically meaningful representations.
Diagram (Mermaid):
flowchart LR
A[Input features] --> B[Linear transform]
B --> C[Non-linearity]
C --> D[Deeper transform]
D --> E[Prediction]
Worked Example: A sentiment classifier can learn local token-level patterns in earlier layers and sentence-level sentiment cues in deeper layers.
Common Mistakes: Assuming more layers always improve results regardless of data volume and optimization stability.
Recap:
- Depth enables compositional representation
- Non-linearity is the critical expressivity unlock
- Architecture should match task complexity
Practice:
- Compare one task where linear models are sufficient and one where deep models are clearly better`
            },
            {
                title: "Lesson 2: Activation Functions and Gradient Flow",
                content: `Learning Objective: Choose activation functions based on optimization behavior and task context.
Core Theory: Activation functions determine signal propagation and gradient stability. ReLU variants are common in hidden layers because they mitigate saturation seen in sigmoid and tanh for deep stacks. Output-layer activation must match task type, such as sigmoid for binary probabilities and softmax for categorical class distributions.
Diagram (Mermaid):
flowchart TD
A[Pre-activation z] --> B{Activation}
B --> C[ReLU family]
B --> D[Sigmoid or tanh]
C --> E[Stable deep training]
D --> F[Potential saturation]
Worked Example: Replacing tanh with ReLU in a 10-layer image model reduces vanishing-gradient issues and accelerates convergence.
Common Mistakes: Applying softmax inside hidden layers and destabilizing optimization.
Recap:
- Activation choice shapes gradient behavior
- Hidden and output activations serve different goals
- Saturation and dead-neuron risks should be monitored
Practice:
- Propose output activation and loss for binary classification, multiclass classification, and regression`
            },
            {
                title: "Lesson 3: Forward Pass, Loss, and Backpropagation",
                content: `Learning Objective: Trace how prediction error propagates backward to update each parameter.
Core Theory: Training consists of forward computation, loss evaluation, backward gradient computation via chain rule, and optimizer update. Backpropagation is efficient dynamic programming over computation graphs, not symbolic re-derivation at every step.
Diagram (Mermaid):
flowchart LR
A[Input batch] --> B[Forward pass]
B --> C[Loss]
C --> D[Backward gradients]
D --> E[Optimizer update]
E --> F[Next iteration]
Worked Example: In a two-layer classifier, gradients from cross-entropy flow through logits, hidden activations, and first-layer weights to adjust all parameters coherently.
Common Mistakes: Forgetting to reset optimizer gradients each iteration and accumulating stale gradients.
Recap:
- Backprop computes parameter sensitivity efficiently
- Loss function defines optimization target
- Correct training loop order is essential
Practice:
- Write pseudocode for one minibatch training iteration with explicit zero_grad, forward, backward, and step`
            },
            {
                title: "Lesson 4: Optimization Algorithms and Learning Rate Strategy",
                content: `Learning Objective: Select optimizers and learning-rate schedules based on convergence dynamics.
Core Theory: SGD with momentum provides strong generalization but may require careful schedule tuning. Adam family optimizers adapt per-parameter step sizes and typically converge faster early. Learning-rate warmup and decay schedules often matter more than optimizer brand for final quality.
Diagram (Mermaid):
flowchart TD
A[Initial learning rate] --> B[Warmup]
B --> C[Stable training phase]
C --> D[Decay]
D --> E[Fine convergence]
Worked Example: A transformer model diverges without warmup; adding linear warmup plus cosine decay stabilizes training and improves final validation loss.
Common Mistakes: Keeping one fixed learning rate through entire training run.
Recap:
- Optimizer and schedule must be tuned together
- Warmup helps large models and large batches
- Late-stage decay improves final minima quality
Practice:
- Compare expected behavior of SGD+momentum vs AdamW on noisy minibatch training`
            },
            {
                title: "Lesson 5: Initialization, Normalization, and Stable Depth",
                content: `Learning Objective: Explain how initialization and normalization influence deep-network trainability.
Core Theory: Poor initialization can explode or vanish activations. He/Xavier-style schemes align variance with layer fan-in/out assumptions. Batch normalization and layer normalization stabilize internal feature distributions and improve optimization robustness.
Diagram (Mermaid):
flowchart LR
A[Weight initialization] --> B[Activation scale]
B --> C[Gradient scale]
C --> D[Training stability]
D --> E[Normalization support]
Worked Example: Deep MLP fails to train with naive random initialization but converges after He initialization and normalization layers.
Common Mistakes: Reusing initialization defaults across fundamentally different architectures.
Recap:
- Initialization determines starting signal quality
- Normalization improves optimization conditioning
- Stable depth requires both architectural and numeric care
Practice:
- Describe why residual connections pair well with normalization in deep stacks`
            },
            {
                title: "Lesson 6: Overfitting, Regularization, and Validation Discipline",
                content: `Learning Objective: Build training workflows that maximize generalization rather than memorization.
Core Theory: Overfitting appears when training error decreases while validation error plateaus or worsens. Regularization methods include dropout, weight decay, data augmentation, early stopping, and label smoothing. Honest model selection requires strict train/validation/test separation.
Diagram (Mermaid):
flowchart TD
A[Train loss down] --> B{Validation trend}
B -->|Improves| C[Continue]
B -->|Worsens| D[Regularize and tune]
D --> E[Re-evaluate]
Worked Example: Image classifier with strong augmentation and weight decay outperforms larger unregularized model on unseen test data.
Common Mistakes: Tuning repeatedly on the test set and inflating reported performance.
Recap:
- Generalization is the production objective
- Regularization is multi-dimensional, not one technique
- Evaluation protocol quality matters as much as model design
Practice:
- Create a checklist for detecting and mitigating overfitting in one training project`
            }
        ]
    },
    {
        number: "AIE - Module 2",
        title: "Transformer Architecture and Attention Engineering",
        description: "Understand attention computation, positional encoding, scaling behavior, and decoder generation mechanics.",
        duration: "125 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Tokenization", "Self-attention", "Multi-head attention", "Positional encoding", "Decoder inference"],
        detailedDescription: "This module explains the core architecture behind modern LLM systems and practical implications for latency and quality.",
        detailedContent: [
            {
                title: "Lesson 1: Tokenization and Embedding Space",
                content: `Learning Objective: Explain how text becomes model-ready token IDs and dense vectors.
Core Theory: Tokenization maps raw text into subword units that trade vocabulary size for compositional coverage. Embedding layers convert token IDs into continuous vectors where semantic and syntactic relationships can be represented geometrically.
Diagram (Mermaid):
flowchart LR
A[Raw text] --> B[Tokenizer]
B --> C[Token IDs]
C --> D[Embedding lookup]
D --> E[Dense vectors]
Worked Example: Rare domain terms split into multiple subwords, affecting context length and prompting strategy.
Common Mistakes: Assuming one token equals one word for budgeting context windows.
Recap:
- Tokenization is a lossy but practical compression of language
- Embeddings are learned semantic representations
- Token budgeting starts at tokenizer behavior
Practice:
- Estimate token-count impact of long numeric tables vs prose paragraphs`
            },
            {
                title: "Lesson 2: Self-Attention Computation",
                content: `Learning Objective: Derive the intuition for query-key-value attention and relevance weighting.
Core Theory: Each token produces query, key, and value vectors. Attention weights are similarity scores between query and keys, normalized with softmax. The resulting weighted sum of values gives context-aware token representations.
Diagram (Mermaid):
flowchart TD
A[Token embeddings] --> B[Q K V projections]
B --> C[Score Q·K]
C --> D[Softmax weights]
D --> E[Weighted sum of V]
Worked Example: Pronoun resolution emerges when a token attends strongly to earlier noun tokens providing referential context.
Common Mistakes: Interpreting attention weights as a complete causal explanation of model behavior.
Recap:
- Attention dynamically routes contextual information
- Softmax normalizes relevance distribution
- Representations become context-dependent at each layer
Practice:
- Explain why scaling by sqrt(d_k) is used in attention scores`
            },
            {
                title: "Lesson 3: Multi-Head Attention and Representation Diversity",
                content: `Learning Objective: Understand why multiple attention heads improve modeling capacity.
Core Theory: Multi-head attention learns parallel relation subspaces. Different heads can specialize in syntactic dependencies, positional cues, or semantic associations. Concatenating head outputs increases representational richness.
Diagram (Mermaid):
flowchart LR
A[Input states] --> B[Head 1 attention]
A --> C[Head 2 attention]
A --> D[Head N attention]
B --> E[Concat and projection]
C --> E
D --> E
Worked Example: One head tracks nearby grammatical structure while another captures long-range topic consistency.
Common Mistakes: Assuming more heads always help regardless of model width and training data.
Recap:
- Heads create parallel attention perspectives
- Diversity across heads can improve contextual encoding
- Architectural scaling must remain balanced
Practice:
- Describe one sign that head count might be over-provisioned`
            },
            {
                title: "Lesson 4: Positional Encoding and Sequence Order",
                content: `Learning Objective: Explain how transformer models recover token order despite parallel processing.
Core Theory: Self-attention alone is permutation-invariant, so positional information must be injected. Absolute or relative positional encodings provide order-aware signals that attention can use for sequence reasoning.
Diagram (Mermaid):
flowchart LR
A[Token embeddings] --> B[Add positional signals]
B --> C[Attention layers]
C --> D[Order-aware contextual states]
Worked Example: Relative positional bias helps models retain meaningful behavior on longer contexts than seen in training.
Common Mistakes: Ignoring positional strategy when extending context length in deployment.
Recap:
- Positional signals are mandatory for ordered language tasks
- Absolute and relative methods have different extrapolation behavior
- Context extension choices affect quality and stability
Practice:
- Compare absolute and relative position encoding trade-offs`
            },
            {
                title: "Lesson 5: Decoder Inference and KV Caching",
                content: `Learning Objective: Understand autoregressive decoding cost and optimization with KV cache.
Core Theory: Decoder-only LLMs generate one token at a time, repeatedly attending to prior context. KV caching stores previous key/value tensors so each new token avoids recomputing full history, significantly reducing inference latency.
Diagram (Mermaid):
flowchart TD
A[Prompt tokens] --> B[Initial forward pass]
B --> C[Store KV cache]
C --> D[Generate next token]
D --> E[Append token and reuse cache]
Worked Example: Chat response latency drops substantially after enabling KV caching in long multi-turn conversations.
Common Mistakes: Measuring throughput without separating prefill and decode phases.
Recap:
- Decoding is inherently sequential
- KV cache is a core production optimization
- Prefill and decode have different performance profiles
Practice:
- Define metrics to evaluate decode-time optimization effectiveness`
            }
        ]
    },
    {
        number: "AIE - Module 3",
        title: "Deep Learning Systems and MLOps for LLM Workloads",
        description: "Move from notebooks to reproducible training and inference systems with monitoring, deployment, and rollback safety.",
        duration: "130 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Data pipelines", "Experiment tracking", "Model registry", "Serving patterns", "Monitoring"],
        detailedDescription: "This module focuses on the engineering discipline required to run deep learning models reliably in production.",
        detailedContent: [
            {
                title: "Lesson 1: Data Pipelines and Feature Integrity",
                content: `Learning Objective: Design data pipelines that preserve training-serving consistency.
Core Theory: Model behavior is highly sensitive to data preprocessing. Versioned datasets, deterministic transforms, and schema validation reduce silent drift. Feature parity between training and serving is mandatory to avoid online/offline skew.
Diagram (Mermaid):
flowchart LR
A[Raw data] --> B[Validation]
B --> C[Transform pipeline]
C --> D[Versioned dataset]
D --> E[Training and serving parity]
Worked Example: Token normalization mismatch between offline training and online API causes significant quality drop despite unchanged model weights.
Common Mistakes: Updating preprocessing logic in serving path without retraining or backtesting.
Recap:
- Data contracts are production dependencies
- Reproducible transforms reduce debugging cost
- Serving parity is a non-negotiable quality condition
Practice:
- Define a minimal dataset versioning and schema-check policy`
            },
            {
                title: "Lesson 2: Experiment Tracking and Reproducibility",
                content: `Learning Objective: Make model experiments auditable and repeatable.
Core Theory: Every run should log code revision, data snapshot, hyperparameters, metrics, and artifacts. Reproducibility enables trustworthy model comparisons and incident forensics.
Diagram (Mermaid):
flowchart TD
A[Experiment run] --> B[Log params and data hash]
B --> C[Capture metrics]
C --> D[Store artifacts]
D --> E[Compare runs]
Worked Example: Regression bug is traced quickly because run metadata links degraded model to a specific tokenizer change.
Common Mistakes: Keeping only final metrics without run context and environment information.
Recap:
- Reproducibility is an engineering requirement
- Metadata quality drives decision quality
- Comparative evaluation needs consistent logging standards
Practice:
- List the mandatory metadata fields for one training run`
            },
            {
                title: "Lesson 3: Model Registry, Promotion, and Rollback",
                content: `Learning Objective: Implement controlled model lifecycle management across environments.
Core Theory: Model registry systems track model versions, lineage, approval status, and deployment stage. Promotion gates should include offline metrics, safety checks, and canary performance before full rollout.
Diagram (Mermaid):
flowchart LR
A[Candidate model] --> B[Registry stage: Staging]
B --> C[Canary deployment]
C --> D{Healthy metrics}
D -->|Yes| E[Promote to production]
D -->|No| F[Rollback]
Worked Example: Canary detects token-cost spike and hallucination increase, preventing faulty model from full release.
Common Mistakes: Promoting models directly from local experiments to production endpoints.
Recap:
- Registry enforces lifecycle discipline
- Promotion gates reduce blast radius
- Rollback readiness must be automatic
Practice:
- Design promotion criteria for a customer-support LLM model`
            },
            {
                title: "Lesson 4: Inference Serving Patterns and Cost Control",
                content: `Learning Objective: Choose serving architecture for latency, throughput, and cost constraints.
Core Theory: Common patterns include synchronous API serving, asynchronous batch generation, and hybrid retrieval-plus-generation flows. Cost control uses request caching, prompt compression, model routing, and token budget limits.
Diagram (Mermaid):
flowchart TD
A[Incoming request] --> B[Route policy]
B --> C[Small model path]
B --> D[Large model path]
C --> E[Low-cost response]
D --> F[High-quality fallback]
Worked Example: Routing easy classification queries to a smaller model cuts total spend while preserving quality on complex cases via larger-model fallback.
Common Mistakes: Optimizing only latency while ignoring exploding token costs.
Recap:
- Serving architecture should reflect workload classes
- Model routing can improve cost-quality balance
- Budget guards are essential for sustainable operations
Practice:
- Propose a two-tier model-routing rule for helpdesk queries`
            },
            {
                title: "Lesson 5: Monitoring Drift, Quality, and Incidents",
                content: `Learning Objective: Build monitoring that catches model regressions before user impact escalates.
Core Theory: Monitor input drift, output quality signals, safety violations, latency, and cost anomalies. Couple alerts to runbooks and ownership. Use periodic human review and golden-set re-evaluation for semantic quality tracking.
Diagram (Mermaid):
flowchart LR
A[Production telemetry] --> B[Quality dashboards]
B --> C[Alert rules]
C --> D[Mitigation and rollback]
D --> E[Postmortem and fixes]
Worked Example: Input domain shift after product launch causes rising abstain rates and incorrect answers, triggering controlled rollback and retraining plan.
Common Mistakes: Monitoring only infrastructure metrics without output-quality measurement.
Recap:
- Model observability must include semantics, not just uptime
- Alerting requires response playbooks
- Continuous evaluation prevents silent degradation
Practice:
- Define a monitoring matrix with at least five quality and reliability signals`
            }
        ]
    }
];

// ---------- Stage 2: Generative AI ----------
courseData.generativeAI = [
    {
        number: "AIE - Module 4",
        title: "Generative AI Foundations and Product Framing",
        description: "Map model capabilities to product requirements with honest constraints and risk-aware architecture choices.",
        duration: "110 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Generative vs discriminative", "Model families", "Use-case framing", "Risk surfaces", "Evaluation goals"],
        detailedDescription: "This module sets the product and engineering frame for building serious generative AI systems.",
        detailedContent: [
            {
                title: "Lesson 1: What Generative Models Actually Learn",
                content: `Learning Objective: Distinguish generative objectives from discriminative prediction tasks.
Core Theory: Discriminative models estimate decision boundaries, while generative models learn data distributions sufficient to synthesize plausible samples. In language, autoregressive models estimate next-token distributions conditioned on prior context.
Diagram (Mermaid):
flowchart LR
A[Training corpus] --> B[Distribution learning]
B --> C[Conditional generation]
C --> D[Text or code output]
Worked Example: A generative support assistant can draft responses in varied styles, while a classifier only predicts predefined labels.
Common Mistakes: Expecting deterministic outputs from probabilistic generation systems.
Recap:
- Generative systems optimize distributional modeling
- Output variability is intrinsic to sampling
- Product design must account for non-determinism
Practice:
- Identify one product feature best solved by classification and one by generation`
            },
            {
                title: "Lesson 2: Capability Mapping and Task Fit",
                content: `Learning Objective: Match LLM strengths and weaknesses to realistic product tasks.
Core Theory: LLMs excel at transformation tasks (summarization, extraction, rewriting) and context-grounded reasoning. They are weak at guaranteed factual recall without grounding and can produce plausible but incorrect output.
Diagram (Mermaid):
flowchart TD
A[Task request] --> B{Needs grounded facts}
B -->|Yes| C[Add retrieval or tools]
B -->|No| D[Direct generation]
Worked Example: Policy Q&A requires retrieval from current policy corpus instead of open-ended model recall.
Common Mistakes: Shipping factual assistants without grounding and source attribution.
Recap:
- Task fit determines architecture complexity
- Grounding is mandatory for factual reliability
- Capability mapping should precede implementation
Practice:
- Classify five candidate features into direct-generation vs grounded-generation buckets`
            },
            {
                title: "Lesson 3: Quality Dimensions Beyond Accuracy",
                content: `Learning Objective: Define multidimensional quality targets for generative products.
Core Theory: Quality spans factuality, relevance, helpfulness, style consistency, latency, cost, and safety compliance. Single scalar metrics can hide severe failure modes; use rubric-based and scenario-based evaluation.
Diagram (Mermaid):
flowchart LR
A[Generated output] --> B[Factuality check]
A --> C[Usefulness check]
A --> D[Safety check]
A --> E[Latency and cost check]
Worked Example: Response quality improves after adding style constraints, but latency regression exceeds SLA; release is blocked until optimization.
Common Mistakes: Optimizing one metric while degrading user-trust dimensions.
Recap:
- Generative quality is multi-objective
- Evaluation design must reflect product goals
- Trade-offs should be explicit and measurable
Practice:
- Build a scoring rubric with at least six dimensions for a writing assistant`
            },
            {
                title: "Lesson 4: Safety, Abuse, and Policy-Driven Design",
                content: `Learning Objective: Integrate safety controls into architecture from day one.
Core Theory: Safety layers include input moderation, output policy checks, sensitive-topic handling, and audit logging. High-risk actions require additional verification and potentially human approval.
Diagram (Mermaid):
flowchart TD
A[User prompt] --> B[Input policy gate]
B --> C[Model response]
C --> D[Output policy gate]
D --> E[Safe response or block]
Worked Example: Financial-advice assistant routes high-risk investment requests to approved guidance workflow.
Common Mistakes: Treating safety as a post-launch patch rather than a core design dimension.
Recap:
- Safety controls are architectural components
- Policy enforcement must be testable
- Auditability is crucial for incident review
Practice:
- Propose a two-stage moderation flow for a public chatbot`
            },
            {
                title: "Lesson 5: Product Rollout Strategy for GenAI Features",
                content: `Learning Objective: Plan staged rollout with measurable confidence gates.
Core Theory: Start with internal pilots, then limited cohorts, then broad release. Each stage needs success criteria for quality, safety, and support load. Feedback loops should rapidly convert observed failures into evaluation tests.
Diagram (Mermaid):
flowchart LR
A[Internal pilot] --> B[Limited beta]
B --> C[Guarded GA]
C --> D[Continuous improvement]
Worked Example: Beta logs reveal ambiguity failures; prompt and retrieval updates are verified on regression set before wider release.
Common Mistakes: Launching globally before collecting representative failure patterns.
Recap:
- Rollout strategy reduces operational risk
- Stage gates require objective metrics
- Feedback-to-test conversion hardens product quality
Practice:
- Define go or no-go criteria for moving from beta to general availability`
            }
        ]
    },
    {
        number: "AIE - Module 5",
        title: "Prompt Engineering for Reliability",
        description: "Design prompts that are robust, testable, and production-ready under adversarial and messy user input.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Instruction design", "Few-shot patterns", "Output contracts", "Injection resistance", "Prompt testing"],
        detailedDescription: "This module turns prompting from art into engineering discipline.",
        detailedContent: [
            {
                title: "Lesson 1: Prompt Structure and Role Separation",
                content: `Learning Objective: Compose prompts with clear instruction hierarchy and stable role boundaries.
Core Theory: System messages define durable behavior constraints. User messages provide task-specific intent. Retrieved context should be clearly delimited as data, not instructions. Explicitly separate policy, task, and context to reduce ambiguity.
Diagram (Mermaid):
flowchart TD
A[System instruction] --> D[Prompt assembly]
B[User request] --> D
C[Retrieved context] --> D
D --> E[Model output]
Worked Example: Classification assistant improves consistency after separating rubric rules from user message body.
Common Mistakes: Blending task policy and retrieved text into one undelimited blob.
Recap:
- Hierarchy clarity improves determinism
- Delimiters reduce instruction confusion
- Role separation supports maintainability
Practice:
- Refactor one unstructured prompt into system, user, and context sections`
            },
            {
                title: "Lesson 2: Few-Shot Patterning and Edge Cases",
                content: `Learning Objective: Use examples to enforce formatting and decision boundaries.
Core Theory: Few-shot prompting gives the model demonstrations of desired reasoning style and output format. Include representative edge cases to reduce brittle behavior on unusual inputs.
Diagram (Mermaid):
flowchart LR
A[Instruction] --> B[Representative examples]
B --> C[Target input]
C --> D[Patterned output]
Worked Example: Ticket-routing prompt handles sarcasm better after adding examples with implicit intent.
Common Mistakes: Providing examples that conflict with instructions or include inconsistent formatting.
Recap:
- Examples are behavioral constraints
- Edge-case coverage improves robustness
- Example quality outweighs quantity
Practice:
- Design three few-shot examples for sentiment labels with ambiguous phrasing`
            },
            {
                title: "Lesson 3: Output Contracts and Structured Generation",
                content: `Learning Objective: Enforce machine-usable outputs with schema constraints and validation.
Core Theory: Production systems should avoid free-form outputs where downstream parsing is required. Use strict JSON schemas or provider-level structured output modes, then validate before business logic execution.
Diagram (Mermaid):
flowchart TD
A[Prompt with schema] --> B[Model generation]
B --> C[Schema validation]
C --> D[Business logic]
C --> E[Repair or retry path]
Worked Example: Extraction pipeline shifts from brittle regex parsing to schema-validated JSON, reducing failure rate.
Common Mistakes: Executing model-produced arguments without type and range validation.
Recap:
- Structured outputs improve reliability
- Validation must be mandatory, not optional
- Retry or repair flows handle malformed responses
Practice:
- Create a JSON schema for extracting invoice id, amount, and due date`
            },
            {
                title: "Lesson 4: Prompt Injection and Context Isolation",
                content: `Learning Objective: Defend prompt pipelines against untrusted instruction content.
Core Theory: Retrieved pages, user uploads, and tool outputs may contain malicious instructions. Treat these as untrusted data. Keep immutable high-priority system constraints and tool permission checks outside model-controllable text.
Diagram (Mermaid):
flowchart LR
A[Untrusted context] --> B[Isolation and sanitization]
B --> C[Prompt assembly]
C --> D[Model]
D --> E[Guarded tool policy]
Worked Example: Web-browsing assistant ignores embedded "reveal system prompt" attack after strict context isolation and response policy checks.
Common Mistakes: Letting tool-call authority depend solely on generated text intent.
Recap:
- External context must be treated as adversarial
- Policy and permissions belong outside untrusted content
- Defense requires layered controls
Practice:
- Propose two controls that prevent unauthorized tool usage via prompt injection`
            },
            {
                title: "Lesson 5: Prompt Evaluation and Regression Harness",
                content: `Learning Objective: Build repeatable test suites for prompt revisions.
Core Theory: Prompt quality should be measured on fixed benchmark sets with rubric scoring. Every change runs regression evaluation to detect quality, safety, and formatting regressions before deployment.
Diagram (Mermaid):
flowchart TD
A[Prompt change] --> B[Run eval suite]
B --> C[Compare baseline]
C --> D{Pass thresholds}
D -->|Yes| E[Deploy]
D -->|No| F[Revise]
Worked Example: New concise prompt improves latency but fails citation completeness checks; revision adds mandatory evidence section.
Common Mistakes: Evaluating prompt changes only on ad hoc manual examples.
Recap:
- Prompting requires CI-like validation
- Benchmark sets should represent real workload diversity
- Threshold-based gates improve release quality
Practice:
- Define three regression metrics for a support-answer prompt`
            }
        ]
    },
    {
        number: "AIE - Module 6",
        title: "Embeddings, Retrieval, and RAG Architecture",
        description: "Engineer retrieval pipelines that improve factual reliability and keep responses current and source-grounded.",
        duration: "130 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Embeddings", "Chunking", "Indexing", "Retrieval", "RAG orchestration", "Evaluation"],
        detailedDescription: "This module covers the most important production pattern for trustworthy LLM applications.",
        detailedContent: [
            {
                title: "Lesson 1: Semantic Embeddings and Similarity Search",
                content: `Learning Objective: Explain how embeddings enable semantic matching beyond keywords.
Core Theory: Embeddings map text to high-dimensional vectors where semantic similarity corresponds to geometric proximity. Similarity metrics such as cosine or dot product drive nearest-neighbor retrieval.
Diagram (Mermaid):
flowchart LR
A[Document text] --> B[Embedding model]
B --> C[Vector space]
D[User query] --> E[Query embedding]
E --> C
C --> F[Nearest chunks]
Worked Example: Query "refund timeline" retrieves policy section titled "reimbursement processing window" despite no literal keyword overlap.
Common Mistakes: Assuming embedding quality is invariant across domains without validation.
Recap:
- Embeddings represent semantic meaning compactly
- Similarity search enables semantic retrieval
- Domain testing is essential for reliability
Practice:
- Compare cosine similarity and dot product considerations for retrieval`
            },
            {
                title: "Lesson 2: Chunking Strategy and Context Quality",
                content: `Learning Objective: Design chunking policies that maximize retrieval relevance.
Core Theory: Chunk size and overlap influence recall and precision. Too small loses context coherence; too large dilutes topical relevance. Structural chunking based on headings often outperforms naive fixed windows.
Diagram (Mermaid):
flowchart TD
A[Source document] --> B[Chunk policy]
B --> C[Chunk set]
C --> D[Embed and index]
D --> E[Retrieve for query]
Worked Example: Legal policy corpus improves answer grounding after heading-aware chunking replaces fixed 1k-token windows.
Common Mistakes: One-size-fits-all chunking across different document formats.
Recap:
- Chunk design is a major quality lever
- Structural boundaries often improve retrieval precision
- Chunk tuning should be empirical
Practice:
- Propose chunking rules for API docs, FAQs, and long contracts`
            },
            {
                title: "Lesson 3: Indexing Pipelines and Metadata Filtering",
                content: `Learning Objective: Build retrieval indices that support relevance and governance constraints.
Core Theory: Index pipelines should capture embeddings plus metadata such as source, timestamp, document type, and access control tags. Metadata filters narrow candidate space before semantic ranking.
Diagram (Mermaid):
flowchart LR
A[Parsed chunks] --> B[Attach metadata]
B --> C[Vector index]
C --> D[Filtered retrieval]
Worked Example: Enterprise assistant filters by tenant and policy version before similarity search, preventing cross-tenant leakage.
Common Mistakes: Indexing raw content without provenance and access tags.
Recap:
- Metadata is critical for safe retrieval
- Filtering improves relevance and compliance
- Provenance enables traceable citations
Practice:
- Define mandatory metadata fields for internal knowledge retrieval`
            },
            {
                title: "Lesson 4: Retrieval Fusion and Re-ranking",
                content: `Learning Objective: Improve recall and precision using hybrid retrieval and re-ranking.
Core Theory: Pure vector search may miss exact keyword constraints. Hybrid retrieval combines lexical and semantic search. Re-rankers reorder candidates using cross-encoder relevance scoring.
Diagram (Mermaid):
flowchart TD
A[User query] --> B[Vector retrieval]
A --> C[Keyword retrieval]
B --> D[Candidate merge]
C --> D
D --> E[Re-ranker]
E --> F[Top context]
Worked Example: Compliance assistant improves citation accuracy after adding BM25 plus semantic fusion and re-ranker stage.
Common Mistakes: Over-relying on top-k vector results without relevance verification.
Recap:
- Hybrid retrieval mitigates single-method blind spots
- Re-ranking improves final context quality
- Retrieval stack complexity should match error profile
Practice:
- Design a hybrid retrieval flow for technical troubleshooting queries`
            },
            {
                title: "Lesson 5: RAG Prompting, Citations, and Answer Control",
                content: `Learning Objective: Construct grounded prompts that enforce source-bounded answering.
Core Theory: RAG prompts should instruct model to answer from provided evidence and abstain when evidence is insufficient. Citation formatting must map output claims to source chunks for auditability.
Diagram (Mermaid):
flowchart LR
A[Retrieved evidence] --> B[Grounded prompt]
B --> C[LLM answer]
C --> D[Citation mapping]
D --> E[User response]
Worked Example: Support bot returns "I do not know" for uncovered edge case instead of hallucinating policy details.
Common Mistakes: Injecting retrieved context without explicit grounding rules and abstention policy.
Recap:
- Grounding instructions control factual behavior
- Citations improve trust and debugging
- Abstention is better than confident fabrication
Practice:
- Write a grounded-answer prompt template with explicit abstain behavior`
            },
            {
                title: "Lesson 6: RAG Evaluation and Continuous Improvement",
                content: `Learning Objective: Evaluate retrieval and generation stages separately and jointly.
Core Theory: End-to-end answer metrics are insufficient. Measure retrieval recall at k, context relevance, citation correctness, groundedness, and task success. Feed observed production failures back into evaluation sets.
Diagram (Mermaid):
flowchart TD
A[Eval query set] --> B[Retrieval metrics]
A --> C[Generation metrics]
B --> D[Error analysis]
C --> D
D --> E[Pipeline tuning]
Worked Example: Low groundedness traced to retrieval misses, fixed by chunking update and metadata filters.
Common Mistakes: Tuning generation prompt when failure root cause is retrieval quality.
Recap:
- Separate-stage metrics accelerate diagnosis
- Evaluation should mirror production scenarios
- Continuous feedback loops compound quality gains
Practice:
- Define five metrics for a quarterly RAG quality dashboard`
            }
        ]
    },
    {
        number: "AIE - Module 7",
        title: "Fine-Tuning and Adaptation Strategies",
        description: "Choose between prompting, RAG, adapters, and full fine-tuning based on cost, risk, and behavior goals.",
        duration: "110 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["When to tune", "Dataset curation", "PEFT and LoRA", "Evaluation", "Deployment"],
        detailedDescription: "This module clarifies when model adaptation is worth the operational complexity and how to execute it safely.",
        detailedContent: [
            {
                title: "Lesson 1: Decision Framework for Fine-Tuning",
                content: `Learning Objective: Decide when to fine-tune versus improving prompts and retrieval.
Core Theory: Fine-tuning is best for persistent behavior shifts such as style, format adherence, domain-specific instruction following, or compression of long system prompts. Dynamic factual knowledge should remain in retrieval systems.
Diagram (Mermaid):
flowchart TD
A[Performance gap] --> B{Knowledge or behavior}
B -->|Knowledge| C[Improve retrieval]
B -->|Behavior| D[Consider fine-tuning]
Worked Example: Assistant requiring strict JSON compliance improves with fine-tuning on validated examples after prompt-only attempts plateau.
Common Mistakes: Fine-tuning to inject rapidly changing operational facts.
Recap:
- Adaptation choice depends on failure category
- Retrieval and tuning are complementary, not substitutes
- Start with lower-cost interventions first
Practice:
- Classify three failure scenarios into retrieval fix vs tuning fix`
            },
            {
                title: "Lesson 2: Curating High-Quality Tuning Data",
                content: `Learning Objective: Build datasets that encode desired behavior without hidden noise.
Core Theory: Tuning quality depends more on label quality and distribution coverage than sheer dataset size. Include difficult and adversarial examples, enforce formatting consistency, and exclude contradictory targets.
Diagram (Mermaid):
flowchart LR
A[Raw interaction logs] --> B[Filtering and dedup]
B --> C[Human curation]
C --> D[Train and validation splits]
D --> E[Tuning dataset]
Worked Example: Domain support responses are curated with rejection of ambiguous low-confidence historical answers.
Common Mistakes: Training on unreviewed chat logs that include incorrect assistant behavior.
Recap:
- Data curation is the primary quality driver
- Coverage and consistency matter more than volume
- Validation split integrity is required for trustworthy evaluation
Practice:
- Define acceptance criteria for adding one sample to tuning dataset`
            },
            {
                title: "Lesson 3: PEFT, LoRA, and Resource-Efficient Adaptation",
                content: `Learning Objective: Explain why parameter-efficient methods dominate practical tuning workflows.
Core Theory: Parameter-efficient fine-tuning methods update a small subset of parameters or low-rank adapters while freezing base weights. This reduces memory and compute requirements while preserving strong baseline capabilities.
Diagram (Mermaid):
flowchart LR
A[Base model weights frozen] --> B[Train adapter layers]
B --> C[Merge or attach adapters]
C --> D[Adapted behavior]
Worked Example: LoRA adapters specialize an open model for customer-support formatting with far lower training cost than full fine-tune.
Common Mistakes: Treating adapter rank and target modules as fixed defaults for every task.
Recap:
- PEFT enables practical adaptation cycles
- Adapter configuration influences quality-cost trade-off
- Base-model choice still strongly affects final performance
Practice:
- List hyperparameters you would sweep for LoRA tuning`
            },
            {
                title: "Lesson 4: Post-Tuning Evaluation and Safety Regression",
                content: `Learning Objective: Validate tuned models against baseline quality, safety, and cost criteria.
Core Theory: Evaluation must compare tuned and baseline models on held-out domain tasks, safety probes, and structural output checks. Guard against regressions where improved domain fit harms general instruction adherence.
Diagram (Mermaid):
flowchart TD
A[Tuned candidate] --> B[Domain eval set]
A --> C[Safety eval set]
A --> D[Format compliance eval]
B --> E[Promotion decision]
C --> E
D --> E
Worked Example: Tuned model improves extraction accuracy but degrades refusal policy compliance; release is blocked pending safety rebalancing.
Common Mistakes: Evaluating only in-domain quality and ignoring broad safety behavior.
Recap:
- Tuning can introduce unintended regressions
- Multi-axis evaluation is mandatory
- Baseline comparison prevents false confidence
Practice:
- Design a pass or fail scorecard for tuned-model release`
            },
            {
                title: "Lesson 5: Deployment and Adapter Lifecycle Management",
                content: `Learning Objective: Deploy adapted models with rollback safety and clear version governance.
Core Theory: Treat adapters and prompts as versioned artifacts. Maintain rollout stages, compatibility matrices, and rapid rollback paths. Monitor drift between tuned behavior and evolving product requirements.
Diagram (Mermaid):
flowchart LR
A[Adapter version] --> B[Staging tests]
B --> C[Canary release]
C --> D[Full deployment]
D --> E[Continuous monitoring]
Worked Example: Region-specific adapter rollout includes automatic fallback to base model when latency or safety thresholds fail.
Common Mistakes: Deploying adapters without explicit compatibility checks against tokenizer and base model versions.
Recap:
- Adapter lifecycle needs same rigor as model lifecycle
- Rollback must be immediate and rehearsed
- Ongoing monitoring preserves tuned-value over time
Practice:
- Define a versioning scheme for base model, adapter, and prompt package`
            }
        ]
    }
];

// ---------- Stage 3: AI Agents ----------
courseData.aiAgents = [
    {
        number: "AIE - Module 8",
        title: "Agentic Systems: Reason-Act-Observe Loops",
        description: "Understand when agentic architectures are appropriate and how to constrain them for reliability.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Agent vs workflow", "Planning loops", "Tool orchestration", "Termination criteria", "Risk control"],
        detailedDescription: "This module establishes the architectural mental model for robust agent behavior.",
        detailedContent: [
            {
                title: "Lesson 1: Agentic Workflows vs Deterministic Pipelines",
                content: `Learning Objective: Decide whether a task needs dynamic planning or fixed workflow execution.
Core Theory: Deterministic workflows are predictable and testable for known paths. Agentic systems are useful when task decomposition is uncertain or context-dependent. Added flexibility increases observability and safety requirements.
Diagram (Mermaid):
flowchart LR
A[Task request] --> B{Path known upfront}
B -->|Yes| C[Deterministic pipeline]
B -->|No| D[Agentic planning loop]
Worked Example: Invoice extraction with fixed schema fits deterministic pipeline, while multi-source research synthesis benefits from agentic exploration.
Common Mistakes: Using agents for tasks that can be solved by simple deterministic orchestration.
Recap:
- Agentic design should be need-driven
- Flexibility introduces control complexity
- Simpler architecture wins when sufficient
Practice:
- Classify four product tasks into deterministic vs agentic approaches`
            },
            {
                title: "Lesson 2: ReAct Loop and State Transitions",
                content: `Learning Objective: Model agent behavior as explicit state transitions.
Core Theory: ReAct cycles through reasoning, tool action, observation, and revision. Explicit state tracking enables budget enforcement, reproducibility, and debugging.
Diagram (Mermaid):
flowchart TD
A[Reason about next step] --> B[Call tool]
B --> C[Observe result]
C --> D{Goal met}
D -->|No| A
D -->|Yes| E[Return answer]
Worked Example: Research agent performs search, reads documents, and iterates until evidence threshold is satisfied.
Common Mistakes: Implicit loops without step counters or stop conditions.
Recap:
- Explicit loop state improves reliability
- Observations must feed next-step decisions
- Stop conditions are core safety controls
Practice:
- Define three termination rules for a research agent`
            },
            {
                title: "Lesson 3: Tool Selection Policy and Permission Boundaries",
                content: `Learning Objective: Prevent unsafe or irrelevant tool usage through policy design.
Core Theory: Tool catalogs should include capability descriptions, input schemas, and permission scopes. Policy layers can disallow categories of actions based on user role, task context, or risk score.
Diagram (Mermaid):
flowchart LR
A[Agent intent] --> B[Tool policy gate]
B --> C[Allowed tool call]
B --> D[Denied or escalation]
Worked Example: Agent can query read-only CRM data for support requests but cannot mutate billing records without approval.
Common Mistakes: Exposing broad tool privileges to all agent runs.
Recap:
- Tool governance is security-critical
- Least privilege reduces blast radius
- Policy checks should be external to model text decisions
Practice:
- Draft permission tiers for read, write, and high-impact tools`
            },
            {
                title: "Lesson 4: Recovery from Failures and Loop Degeneration",
                content: `Learning Objective: Detect and recover from stuck loops and low-progress behavior.
Core Theory: Failure handling includes max-step budgets, repeated-action detection, fallback prompts, and human handoff triggers. Observability should capture loop trajectories to identify systematic planner issues.
Diagram (Mermaid):
flowchart TD
A[Loop execution] --> B{Progress signal}
B -->|Healthy| C[Continue]
B -->|Stalled| D[Fallback strategy]
D --> E[Escalate or terminate]
Worked Example: Agent repeatedly calls same search query; repeated-action detector forces reformulation strategy before another tool call.
Common Mistakes: Allowing unlimited loops in production workflows.
Recap:
- Degeneration detection is mandatory
- Recovery plans should be deterministic
- Escalation paths preserve user trust
Practice:
- Define a stalled-loop detector using two measurable signals`
            },
            {
                title: "Lesson 5: Agent Evaluation for Task Completion",
                content: `Learning Objective: Evaluate agents on end-task success, not only response quality.
Core Theory: Agent evaluation should include completion rate, tool-efficiency, failure mode taxonomy, and human-review burden. Scenario-based benchmarks expose planning defects invisible to single-turn prompts.
Diagram (Mermaid):
flowchart LR
A[Agent run logs] --> B[Task success metrics]
A --> C[Tool-call efficiency]
A --> D[Failure taxonomy]
B --> E[Iteration improvements]
C --> E
D --> E
Worked Example: Agent success rises after introducing plan-verification step before executing expensive tools.
Common Mistakes: Measuring only final answer quality while ignoring excessive or unsafe tool usage.
Recap:
- Completion metrics capture real utility
- Efficiency and safety are first-class outcomes
- Failure taxonomy guides targeted fixes
Practice:
- Build an evaluation table for agent task success and operational overhead`
            }
        ]
    },
    {
        number: "AIE - Module 9",
        title: "Tool Calling, Function Execution, and Safe Integrations",
        description: "Implement production-grade function-calling loops with validation, retries, and auditability.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Function schemas", "Execution loop", "Argument validation", "Retries and timeouts", "Auditing"],
        detailedDescription: "This module focuses on the most practical foundation of real-world agents: reliable tool execution.",
        detailedContent: [
            {
                title: "Lesson 1: Function Schema Design",
                content: `Learning Objective: Design tool schemas that reduce hallucinated arguments and ambiguous calls.
Core Theory: Good schemas use explicit required fields, constrained enums, and descriptive parameter docs. Narrow schemas reduce model ambiguity and improve execution reliability.
Diagram (Mermaid):
flowchart TD
A[Tool intent] --> B[Schema definition]
B --> C[Model tool call]
C --> D[Validator]
D --> E[Execution]
Worked Example: Currency conversion tool with enum-constrained currency codes eliminates malformed requests.
Common Mistakes: Creating overly generic string parameters for structured business logic.
Recap:
- Schema clarity drives call correctness
- Constrained parameters improve reliability
- Tool ergonomics influence model behavior
Practice:
- Design a schema for flight search with strict date and cabin-class constraints`
            },
            {
                title: "Lesson 2: Robust Call-Execute-Return Loop",
                content: `Learning Objective: Implement safe loop orchestration between model and tool runtime.
Core Theory: The runtime should parse tool requests, validate arguments, execute side effects under policy, append tool results, and resume model reasoning. Each step must be logged with correlation IDs.
Diagram (Mermaid):
flowchart LR
A[Model proposes call] --> B[Validate]
B --> C[Execute tool]
C --> D[Capture result]
D --> E[Return observation to model]
Worked Example: Support agent calls account-status API, returns normalized payload, then composes customer response with clear status and next actions.
Common Mistakes: Feeding raw tool exceptions directly back to users.
Recap:
- Loop orchestration must be deterministic and inspectable
- Validation precedes all side effects
- Normalized observations improve downstream reasoning
Practice:
- Write pseudocode for tool execution middleware with structured error handling`
            },
            {
                title: "Lesson 3: Validation, Policy Checks, and Side-Effect Safety",
                content: `Learning Objective: Prevent harmful actions through layered pre-execution controls.
Core Theory: Use schema validation, business-rule checks, authorization checks, and dry-run simulations for high-impact operations. Include human approval gates for irreversible actions.
Diagram (Mermaid):
flowchart TD
A[Tool call request] --> B[Schema validation]
B --> C[Authorization check]
C --> D[Business rule check]
D --> E{High impact}
E -->|Yes| F[Human approval]
E -->|No| G[Execute]
Worked Example: Refund issuance tool requires transaction ownership validation and manager approval over threshold amounts.
Common Mistakes: Assuming model intent is equivalent to user authorization.
Recap:
- Safety requires policy checks beyond schema validity
- High-impact actions need extra controls
- Authorization logic must stay outside model text generation
Practice:
- Define approval policy for financial and account-deletion tool calls`
            },
            {
                title: "Lesson 4: Retry Semantics, Idempotency, and Timeouts",
                content: `Learning Objective: Make tool execution resilient without duplicating side effects.
Core Theory: Retries should be bounded and aware of idempotency. Non-idempotent operations require idempotency keys or compensating transactions. Timeouts prevent hung workflows and preserve system health.
Diagram (Mermaid):
flowchart LR
A[Tool execution] --> B{Success}
B -->|No| C[Retry policy]
C --> D{Idempotent}
D -->|Yes| E[Retry safely]
D -->|No| F[Escalate]
Worked Example: Payment-capture tool includes idempotency token to prevent double charges on retry.
Common Mistakes: Blind retries on non-idempotent operations.
Recap:
- Reliability and correctness must be balanced
- Retry policy depends on operation semantics
- Timeout and circuit-breaker patterns protect infrastructure
Practice:
- Propose retry classes for read-only, write-idempotent, and write-non-idempotent tools`
            },
            {
                title: "Lesson 5: Audit Logs and Compliance Readiness",
                content: `Learning Objective: Build traceability for every agent action and decision path.
Core Theory: Audit logs should capture user request, prompt version, tool calls, arguments, policy decisions, outputs, and timestamps. Immutable logging supports incident response and compliance obligations.
Diagram (Mermaid):
flowchart TD
A[Agent run] --> B[Event stream]
B --> C[Immutable audit store]
C --> D[Compliance reporting]
C --> E[Incident investigation]
Worked Example: Disputed account action is resolved quickly because full tool-call chain and approvals are recorded.
Common Mistakes: Logging only final responses and losing decision provenance.
Recap:
- End-to-end traceability is operationally critical
- Audit data supports security and regulatory workflows
- Provenance improves debugging and trust
Practice:
- Define required audit fields for a regulated customer-support agent`
            }
        ]
    },
    {
        number: "AIE - Module 10",
        title: "Agent Memory, Planning, and Multi-Step Reasoning",
        description: "Implement memory systems and planners that keep agents coherent across long and complex tasks.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Working memory", "Long-term memory", "Planning", "Reflection", "Cost-aware reasoning"],
        detailedDescription: "This module addresses the hardest practical challenge in agents: maintaining context and plan quality over many steps.",
        detailedContent: [
            {
                title: "Lesson 1: Working Memory and Context Budgeting",
                content: `Learning Objective: Manage short-term context so reasoning remains coherent and efficient.
Core Theory: Working memory includes recent dialogue, current plan state, and critical observations. Context windows are finite, so systems must prioritize salient state while pruning redundant history.
Diagram (Mermaid):
flowchart LR
A[Conversation history] --> B[Relevance filter]
B --> C[Working context]
C --> D[Model reasoning]
Worked Example: Agent summary memory compresses previous 30 turns into compact state and preserves decision continuity.
Common Mistakes: Appending full history blindly until context truncation causes hidden information loss.
Recap:
- Working memory should be intentionally curated
- Compression and relevance filters are essential
- Context budget is a first-class resource
Practice:
- Design a policy for retaining, summarizing, or dropping prior turns`
            },
            {
                title: "Lesson 2: Long-Term Memory with Retrieval",
                content: `Learning Objective: Store and recall durable user and task facts safely.
Core Theory: Long-term memory stores facts, preferences, and prior outcomes in external storage. Retrieval should be relevance-scored and policy-filtered before injection into prompts.
Diagram (Mermaid):
flowchart TD
A[Important fact detected] --> B[Memory write]
B --> C[Indexed store]
D[New request] --> E[Relevant memory retrieval]
E --> F[Prompt context]
Worked Example: Agent recalls user timezone preference from prior sessions and schedules reminders correctly.
Common Mistakes: Storing sensitive data without retention policy or consent boundaries.
Recap:
- Long-term memory enables personalization and continuity
- Retrieval should be selective and policy-aware
- Data governance must apply to memory systems
Practice:
- Define which memory types should expire vs persist indefinitely`
            },
            {
                title: "Lesson 3: Planning Strategies and Task Decomposition",
                content: `Learning Objective: Improve completion quality via explicit subtask planning.
Core Theory: Plans decompose complex goals into verifiable steps with dependencies. Dynamic replanning updates steps when observations invalidate assumptions. Planner quality strongly affects tool efficiency and success rate.
Diagram (Mermaid):
flowchart LR
A[Goal] --> B[Initial plan]
B --> C[Execute step]
C --> D[Observe result]
D --> E{Plan still valid}
E -->|No| F[Replan]
E -->|Yes| C
Worked Example: Incident-analysis agent revises plan after missing logs force alternate data source path.
Common Mistakes: Skipping explicit plans for long tasks and relying on ad hoc reasoning.
Recap:
- Planning adds structure to agent behavior
- Replanning handles uncertainty and non-determinism
- Step-level verification improves reliability
Practice:
- Draft a five-step plan template with validation checkpoint fields`
            },
            {
                title: "Lesson 4: Reflection and Self-Critique Loops",
                content: `Learning Objective: Use bounded self-critique to improve answer quality without runaway latency.
Core Theory: Reflection prompts ask model to evaluate draft output against rubric before finalizing. This can improve quality on complex tasks but increases token and latency cost, so should be triggered selectively.
Diagram (Mermaid):
flowchart TD
A[Draft output] --> B[Reflection check]
B --> C{Quality pass}
C -->|No| D[Revise]
D --> B
C -->|Yes| E[Finalize]
Worked Example: Legal-summary agent catches missing citation in reflection phase and corrects before user delivery.
Common Mistakes: Applying reflection to every trivial request and doubling costs unnecessarily.
Recap:
- Reflection is a targeted quality tool
- Trigger conditions should be policy-driven
- Bounded loops prevent latency explosion
Practice:
- Define trigger rules for when reflection is required`
            },
            {
                title: "Lesson 5: Cost-Aware Reasoning and Budget Enforcement",
                content: `Learning Objective: Keep multi-step reasoning economically sustainable.
Core Theory: Agent systems should track token, tool, and time budgets per request. Planner can adjust strategy based on remaining budget, choosing concise reasoning or fallback summary when limits near.
Diagram (Mermaid):
flowchart LR
A[Request budget] --> B[Plan selection]
B --> C[Execution tracking]
C --> D{Budget remaining}
D -->|Low| E[Fallback strategy]
D -->|Healthy| F[Continue]
Worked Example: Research agent shifts from exhaustive search to summary mode after budget threshold is reached.
Common Mistakes: Measuring quality without visibility into per-task cost and marginal utility.
Recap:
- Budget-aware planning protects operational viability
- Cost and quality should be co-optimized
- Transparent budgeting improves product predictability
Practice:
- Create a budget policy with token, tool-call, and time limits for one agent workflow`
            }
        ]
    }
];

// ---------- Stage 4: Frameworks, Protocols, and Production ----------
courseData.agentFrameworks = [
    {
        number: "AIE - Module 11",
        title: "LangChain and LangGraph for Production Agent Workflows",
        description: "Use framework primitives to build stateful, testable, and observable agent systems at scale.",
        duration: "125 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["LangChain primitives", "LCEL composition", "LangGraph state machines", "Human-in-the-loop", "Tracing"],
        detailedDescription: "This module shows how to apply popular frameworks without losing architectural clarity and control.",
        detailedContent: [
            {
                title: "Lesson 1: LangChain Core Abstractions",
                content: `Learning Objective: Understand model, prompt, tool, retriever, and parser abstractions.
Core Theory: LangChain standardizes interfaces for composing LLM apps and swapping providers. Clear abstraction boundaries reduce vendor lock-in and accelerate experimentation.
Diagram (Mermaid):
flowchart TD
A[Prompt template] --> B[Model]
B --> C[Output parser]
A --> D[Retriever]
D --> B
Worked Example: One retrieval pipeline switches from one model provider to another with minimal orchestration changes.
Common Mistakes: Treating framework abstraction as substitute for understanding underlying model behavior.
Recap:
- Abstractions improve composability
- Provider interchangeability reduces migration risk
- Architecture understanding remains essential
Practice:
- Map a RAG pipeline into LangChain abstraction components`
            },
            {
                title: "Lesson 2: LCEL Composition and Reusable Chains",
                content: `Learning Objective: Build reusable chain components with clear interfaces.
Core Theory: LCEL composition encourages declarative pipelines where each component has defined input/output contracts. Reusable chain modules improve maintainability and testing.
Diagram (Mermaid):
flowchart LR
A[Input payload] --> B[Prompt node]
B --> C[Model node]
C --> D[Parser node]
D --> E[Structured output]
Worked Example: Shared summarization chain is reused across support tickets, release notes, and executive reports with task-specific prompt inputs.
Common Mistakes: Embedding business logic directly in opaque prompt text instead of explicit chain steps.
Recap:
- Declarative chains support readability and reuse
- Component contracts enable testing
- Separation of concerns improves iteration speed
Practice:
- Design one reusable chain interface for text extraction`
            },
            {
                title: "Lesson 3: LangGraph State Machines and Durable Loops",
                content: `Learning Objective: Use graph-based orchestration for multi-step, branching, and looping agent workflows.
Core Theory: LangGraph models agent execution as a state graph with typed state, deterministic edges, and conditional routing. Checkpointing supports crash recovery and human intervention.
Diagram (Mermaid):
flowchart TD
A[Model node] --> B{Tool needed}
B -->|Yes| C[Tool node]
C --> A
B -->|No| D[Finalize]
Worked Example: Review agent loops through retrieval and citation validation until confidence threshold is reached.
Common Mistakes: Building long-running loops in linear chains without explicit state transitions.
Recap:
- Graph orchestration improves control and resilience
- Typed state reduces hidden coupling
- Conditional edges support explicit decision logic
Practice:
- Draw a graph for an approve-or-escalate support workflow`
            },
            {
                title: "Lesson 4: Human-in-the-Loop and Approval Gates",
                content: `Learning Objective: Integrate human checkpoints into automated workflows for high-impact actions.
Core Theory: Framework-level pause/resume and checkpoint APIs allow approval before irreversible operations. Human decisions should be logged as first-class events in run traces.
Diagram (Mermaid):
flowchart LR
A[Agent proposes action] --> B[Approval gate]
B --> C{Approved}
C -->|Yes| D[Execute action]
C -->|No| E[Revise or stop]
Worked Example: Finance agent pauses before refund issuance above policy threshold for supervisor approval.
Common Mistakes: Relying on natural-language model caution without formal approval gates.
Recap:
- HITL controls reduce risk in critical workflows
- Approval metadata improves accountability
- Pause-resume infrastructure should be tested regularly
Practice:
- Specify approval thresholds and escalation paths for three tool categories`
            },
            {
                title: "Lesson 5: Tracing and Debugging Framework Pipelines",
                content: `Learning Objective: Diagnose failures using structured traces and state snapshots.
Core Theory: Tracing captures prompt versions, model responses, tool calls, latency breakdown, and branch decisions. Root-cause analysis becomes practical when each node transition is observable.
Diagram (Mermaid):
flowchart TD
A[Pipeline run] --> B[Trace capture]
B --> C[Step-level metrics]
C --> D[Failure diagnosis]
D --> E[Patch and regression test]
Worked Example: Unexpected hallucination traced to missing retrieval node branch after a graph refactor.
Common Mistakes: Debugging by final output only and ignoring intermediate state transitions.
Recap:
- Observability is required for safe iteration
- Node-level traces accelerate debugging
- Trace-driven regression tests prevent repeat failures
Practice:
- Define a minimum tracing schema for framework-based agent workflows`
            }
        ]
    },
    {
        number: "AIE - Module 12",
        title: "MCP Protocol, Tool Ecosystem, and Enterprise Integration",
        description: "Use MCP-style tool interoperability patterns to scale agent capabilities across products and teams.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Protocol model", "Tools and resources", "Auth and trust", "Server design", "Enterprise governance"],
        detailedDescription: "This module focuses on protocol-level interoperability and governance for large AI engineering organizations.",
        detailedContent: [
            {
                title: "Lesson 1: Interoperability Problem and Protocol Value",
                content: `Learning Objective: Explain why protocol standards matter for AI tool ecosystems.
Core Theory: Without shared protocols, each AI client needs custom integrations for each tool backend, creating high maintenance and inconsistent security posture. Protocol-oriented design standardizes discovery, invocation, and metadata.
Diagram (Mermaid):
flowchart LR
A[Client A] --> D[Protocol server]
B[Client B] --> D
C[Client C] --> D
D --> E[Shared tools and resources]
Worked Example: Multiple internal assistants consume the same approved knowledge and action tools through one protocol surface.
Common Mistakes: Building one-off proprietary connectors for each team.
Recap:
- Protocols reduce integration duplication
- Standard surfaces improve governance
- Shared tooling accelerates platform velocity
Practice:
- List three integration costs reduced by protocol standardization`
            },
            {
                title: "Lesson 2: Tool, Resource, and Prompt Contracts",
                content: `Learning Objective: Design interoperable contracts for actions, data access, and reusable prompts.
Core Theory: Contracts should include capability metadata, input/output schema, version, and policy tags. Discovery endpoints enable clients to reason about available capabilities before invocation.
Diagram (Mermaid):
flowchart TD
A[Capability catalog] --> B[Tool contract]
A --> C[Resource contract]
A --> D[Prompt contract]
B --> E[Client invocation]
Worked Example: A shared financial-data tool advertises rate limits, required scopes, and schema so different clients can integrate safely.
Common Mistakes: Publishing tools without versioning or compatibility guarantees.
Recap:
- Contracts are the basis of reliable interoperability
- Capability metadata improves client orchestration
- Version governance prevents breaking integrations
Practice:
- Draft a versioning policy for protocol-exposed tools`
            },
            {
                title: "Lesson 3: Authentication, Authorization, and Trust Boundaries",
                content: `Learning Objective: Secure protocol-based tool access in multi-tenant environments.
Core Theory: Protocol clients should authenticate identities and pass scoped authorization context. Servers enforce least privilege, tenant isolation, and audit logging on each invocation.
Diagram (Mermaid):
flowchart LR
A[Client identity] --> B[Auth token]
B --> C[Protocol server]
C --> D[Policy engine]
D --> E[Tool execution]
Worked Example: Tenant-scoped retrieval server blocks cross-tenant document access despite similar query vectors.
Common Mistakes: Trusting client-declared tenant IDs without server-side policy validation.
Recap:
- Security boundaries must be explicit and enforced server-side
- Scoped tokens reduce unauthorized access risk
- Audit logs support compliance and forensics
Practice:
- Define required claims for a tool invocation authorization token`
            },
            {
                title: "Lesson 4: Protocol Server Reliability and Scalability",
                content: `Learning Objective: Build servers that remain reliable under concurrent tool traffic.
Core Theory: Server design requires timeout controls, concurrency limits, queueing policy, and idempotency handling for retried requests. Backpressure mechanisms protect downstream dependencies.
Diagram (Mermaid):
flowchart TD
A[Incoming calls] --> B[Rate and concurrency controls]
B --> C[Execution workers]
C --> D[Result and audit]
C --> E[Retry and error policy]
Worked Example: Tool server remains stable during traffic spike by applying concurrency caps and graceful degradation responses.
Common Mistakes: Unbounded parallel execution causing cascading downstream failures.
Recap:
- Reliability requires capacity-aware execution design
- Backpressure and rate control prevent collapse
- Error semantics should be explicit and consistent
Practice:
- Specify timeout and retry classes for three tool categories`
            },
            {
                title: "Lesson 5: Governance for Shared Tool Platforms",
                content: `Learning Objective: Operate shared protocol tool ecosystems with clear ownership and lifecycle controls.
Core Theory: Governance includes ownership metadata, SLA definitions, deprecation policy, security reviews, and change management. Platform teams should provide certification gates before tools become broadly discoverable.
Diagram (Mermaid):
flowchart LR
A[New tool proposal] --> B[Security and quality review]
B --> C[Certification]
C --> D[Catalog publish]
D --> E[Lifecycle monitoring]
Worked Example: Deprecated tool version remains available with warning window before enforced migration to secure replacement.
Common Mistakes: Publishing tools without support ownership or deprecation timelines.
Recap:
- Shared ecosystems need platform governance
- Certification improves trust and reliability
- Lifecycle policy prevents integration chaos
Practice:
- Create a governance checklist for onboarding new protocol tools`
            }
        ]
    },
    {
        number: "AIE - Module 13",
        title: "Multi-Agent Architectures and Coordination Patterns",
        description: "Design specialist-agent systems with supervisor control, shared memory, and bounded collaboration loops.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Role decomposition", "Supervisor routing", "Shared context", "Conflict resolution", "Performance trade-offs"],
        detailedDescription: "This module helps you design multi-agent systems that are useful and controllable rather than expensive and chaotic.",
        detailedContent: [
            {
                title: "Lesson 1: Role Decomposition and Specialization",
                content: `Learning Objective: Split complex workflows into specialist responsibilities.
Core Theory: Specialized agents reduce tool overload and instruction ambiguity by narrowing cognitive scope. Effective decomposition aligns each agent with distinct objectives, tools, and evaluation criteria.
Diagram (Mermaid):
flowchart LR
A[Complex task] --> B[Research agent]
A --> C[Execution agent]
A --> D[Review agent]
B --> E[Supervisor synthesis]
C --> E
D --> E
Worked Example: Incident-response pipeline uses separate diagnosis, remediation, and reviewer agents to improve correctness.
Common Mistakes: Creating many overlapping agents with unclear boundaries.
Recap:
- Specialization improves reliability when boundaries are clear
- Decomposition should follow task structure
- Ownership clarity reduces coordination errors
Practice:
- Define role boundaries for a three-agent technical-support system`
            },
            {
                title: "Lesson 2: Supervisor and Routing Policies",
                content: `Learning Objective: Build supervisor logic that routes subtasks effectively.
Core Theory: Supervisors can use rule-based routing, model-based routing, or hybrid strategies. Routing confidence thresholds and fallback paths reduce misassignment costs.
Diagram (Mermaid):
flowchart TD
A[Incoming subtask] --> B[Supervisor router]
B --> C[Agent A]
B --> D[Agent B]
B --> E[Agent C]
C --> F[Supervisor merge]
D --> F
E --> F
Worked Example: Routing errors drop after adding task taxonomy and confidence-based escalation to human reviewer.
Common Mistakes: Hard-coding one routing strategy without performance telemetry.
Recap:
- Routing is a measurable control function
- Confidence-aware fallback improves robustness
- Supervisor should remain lightweight and transparent
Practice:
- Propose routing features for assigning legal, billing, and technical subtasks`
            },
            {
                title: "Lesson 3: Shared Memory and Coordination State",
                content: `Learning Objective: Design shared state that supports collaboration without conflict.
Core Theory: Multi-agent systems need shared context stores for plans, evidence, and decisions. State updates should be versioned and conflict-aware to avoid stale-overwrite behavior.
Diagram (Mermaid):
flowchart LR
A[Agent outputs] --> B[Shared state store]
B --> C[Supervisor view]
B --> D[Agent retrieval]
Worked Example: Agents append evidence with source IDs and confidence scores, enabling transparent synthesis.
Common Mistakes: Free-form shared notes that become contradictory and unverifiable.
Recap:
- Shared state must be structured
- Versioning prevents coordination races
- Evidence provenance supports quality review
Practice:
- Define a shared-state schema with fields for claim, source, confidence, and owner`
            },
            {
                title: "Lesson 4: Conflict Resolution and Consensus Checks",
                content: `Learning Objective: Resolve contradictory agent outputs with structured policies.
Core Theory: Conflict policies can prioritize trusted agents, require evidence-weighted ranking, or trigger tie-breaker review agent. Consensus should be based on verifiable evidence, not majority opinion alone.
Diagram (Mermaid):
flowchart TD
A[Conflicting outputs] --> B[Conflict detector]
B --> C[Evidence comparison]
C --> D{Resolved}
D -->|No| E[Reviewer or human escalation]
D -->|Yes| F[Final decision]
Worked Example: Reviewer agent resolves discrepancy between retrieval and calculation agents by recomputing with explicit evidence constraints.
Common Mistakes: Merging outputs without checking contradiction or evidence quality.
Recap:
- Conflict handling is required in multi-agent systems
- Evidence quality should drive resolution
- Escalation paths reduce silent error propagation
Practice:
- Create a resolution policy for contradictory risk assessments`
            },
            {
                title: "Lesson 5: Cost, Latency, and Utility in Multi-Agent Systems",
                content: `Learning Objective: Evaluate when multi-agent orchestration creates net value.
Core Theory: Multi-agent designs increase coordination overhead and token usage. Utility should be measured as quality gain per added cost and latency. For many tasks, a single well-instrumented agent remains superior.
Diagram (Mermaid):
flowchart LR
A[Architecture choice] --> B[Quality gain]
A --> C[Cost increase]
A --> D[Latency increase]
B --> E[Utility decision]
C --> E
D --> E
Worked Example: Two-agent reviewer architecture improves factuality modestly but doubles latency; team keeps it only for high-risk tasks.
Common Mistakes: Assuming more agents always means better outcomes.
Recap:
- Multi-agent is a strategic choice, not default
- Utility must account for cost and latency
- Task-tiering can selectively apply complex orchestration
Practice:
- Propose criteria for when to enable multi-agent mode`
            }
        ]
    },
    {
        number: "AIE - Module 14",
        title: "Production Readiness: Evaluation, Safety, and Operations",
        description: "Ship agentic AI systems with robust quality gates, incident response, governance, and continuous improvement loops.",
        duration: "130 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Offline evals", "Online metrics", "Safety operations", "Incident response", "Cost governance", "Roadmap loops"],
        detailedDescription: "This capstone module turns prototypes into launch-ready AI engineering systems fit for high-stakes production use.",
        detailedContent: [
            {
                title: "Lesson 1: Offline Evaluation Harness Design",
                content: `Learning Objective: Build benchmark suites that represent real-world workload diversity.
Core Theory: Offline evaluation should include scenario coverage across routine, edge, and adversarial inputs. Use rubric-based grading with deterministic checks where possible and calibrated LLM judges where needed.
Diagram (Mermaid):
flowchart TD
A[Curated eval set] --> B[Run candidate system]
B --> C[Score by rubric]
C --> D[Gate decision]
Worked Example: Launch candidate fails edge-case escalation criterion despite high average quality score.
Common Mistakes: Overfitting to tiny benchmark sets that do not reflect production traffic.
Recap:
- Evaluation quality depends on scenario representativeness
- Rubrics should map to product objectives
- Gate criteria should be explicit and versioned
Practice:
- Define an eval dataset composition policy by traffic segment`
            },
            {
                title: "Lesson 2: Online Metrics and User-Centric Observability",
                content: `Learning Objective: Track live quality and reliability with actionable telemetry.
Core Theory: Online monitoring should include task completion rate, user correction rate, escalation rate, latency percentiles, safety intervention frequency, and per-request cost. Segment metrics by user cohort and task type.
Diagram (Mermaid):
flowchart LR
A[Live requests] --> B[Telemetry pipeline]
B --> C[Quality dashboards]
B --> D[Alert rules]
C --> E[Optimization backlog]
Worked Example: Rising correction rate in one region reveals localization prompt gap rather than model-level regression.
Common Mistakes: Relying on aggregate metrics that hide cohort-specific failures.
Recap:
- Live metrics should reflect user outcomes
- Segmentation improves root-cause clarity
- Dashboards must connect to ownership and actions
Practice:
- Design a dashboard with five core KPIs and owner mapping`
            },
            {
                title: "Lesson 3: Safety Operations and Policy Lifecycle",
                content: `Learning Objective: Operate safety policies as evolving systems, not static checklists.
Core Theory: Safety operations require policy versioning, abuse-pattern tracking, red-team exercises, and regular rule updates. False positives and false negatives should be measured and tuned.
Diagram (Mermaid):
flowchart TD
A[Policy version] --> B[Runtime enforcement]
B --> C[Incident feedback]
C --> D[Policy update]
D --> A
Worked Example: Prompt-injection attacks in uploaded documents trigger policy hardening and stricter context isolation.
Common Mistakes: Freezing safety policy after launch while threat patterns evolve.
Recap:
- Safety policy must evolve with threat landscape
- Feedback loops are essential for policy quality
- Trade-off tuning should be data-driven
Practice:
- Propose a monthly safety-operations review agenda`
            },
            {
                title: "Lesson 4: Incident Response for AI Systems",
                content: `Learning Objective: Respond to quality and safety incidents with structured operational discipline.
Core Theory: Incident workflow should include severity classification, containment, user-impact analysis, mitigation rollout, and retrospective remediation tasks. Preserve prompt, model, and tool-call evidence for forensic review.
Diagram (Mermaid):
flowchart LR
A[Incident detected] --> B[Triage severity]
B --> C[Containment]
C --> D[Mitigation release]
D --> E[Postmortem]
Worked Example: Hallucinated legal guidance incident is contained by enabling strict citation mode and temporarily routing high-risk queries to human review.
Common Mistakes: Patching symptoms without collecting reproducible evidence.
Recap:
- AI incidents need classic SRE rigor plus model-specific evidence
- Containment speed protects users
- Retrospectives must produce measurable follow-up actions
Practice:
- Write an incident template specific to agent hallucination events`
            },
            {
                title: "Lesson 5: Cost Governance and Capacity Planning",
                content: `Learning Objective: Keep AI operations financially sustainable under growth.
Core Theory: Governance includes per-feature budgets, model-tier routing, caching strategy, and quota controls. Capacity planning should model token demand, concurrency peaks, and tool backend bottlenecks.
Diagram (Mermaid):
flowchart TD
A[Demand forecast] --> B[Capacity plan]
B --> C[Budget controls]
C --> D[Runtime enforcement]
D --> E[Monthly optimization]
Worked Example: Team introduces request classification and cache layer, reducing monthly token spend while preserving quality targets.
Common Mistakes: Launching premium models broadly without usage controls.
Recap:
- Cost control is a product and platform concern
- Capacity planning prevents reliability surprises
- Governance should be proactive, not reactive
Practice:
- Define monthly cost review metrics for one AI product`
            },
            {
                title: "Lesson 6: Continuous Improvement and Launch Readiness Checklist",
                content: `Learning Objective: Establish a repeatable launch and iteration cadence for high-quality AI products.
Core Theory: Mature teams operate closed loops between user feedback, evaluation suites, prompt/model changes, and release governance. Launch readiness requires passing quality, safety, latency, cost, and supportability gates.
Diagram (Mermaid):
flowchart LR
A[User feedback] --> B[Eval backlog updates]
B --> C[Improvements]
C --> D[Gate checks]
D --> E[Release]
E --> A
Worked Example: Quarterly release cadence includes mandatory red-team pass, regression suite pass, and canary health validation.
Common Mistakes: Treating launch as endpoint instead of start of operational learning.
Recap:
- Continuous loops sustain long-term quality
- Multi-gate readiness reduces launch risk
- Operational excellence differentiates premium AI products
Practice:
- Create a launch checklist with at least ten go or no-go items`
            }
        ]
    }
];
