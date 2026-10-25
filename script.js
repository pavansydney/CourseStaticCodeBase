const courseData = {};

// Initialize the page
// ============================================================
// Practice quizzes (per certification track)
// Data-driven: add a panel id key with an array of questions to
// surface a quiz under that certification's modules. Each question:
// { q, options: [..], answer: <index of correct option>, explain }
// ============================================================
const quizData = {
    'panel-ai-901': [
        { q: "Which Azure service would you use to extract sentiment and key phrases from customer reviews?", options: ["Azure AI Vision", "Azure AI Language", "Azure AI Search", "Azure Machine Learning"], answer: 1, explain: "Azure AI Language provides natural language processing features such as sentiment analysis, key phrase extraction, and entity recognition." },
        { q: "A model that produces new text from a prompt is an example of which AI workload?", options: ["Computer vision", "Anomaly detection", "Generative AI", "Regression"], answer: 2, explain: "Generative AI creates new content (text, images, code) from a prompt, unlike classification or regression which predict labels or values." },
        { q: "Which Responsible AI principle focuses on ensuring a system does not discriminate against groups of people?", options: ["Reliability & safety", "Fairness", "Privacy & security", "Transparency"], answer: 1, explain: "Fairness is about treating all groups equitably and avoiding bias in AI outcomes." },
        { q: "Detecting and labelling objects within an image is part of which workload?", options: ["Natural language processing", "Computer vision", "Knowledge mining", "Conversational AI"], answer: 1, explain: "Computer vision covers image classification, object detection, and OCR." },
        { q: "What is Azure AI Foundry (Microsoft Foundry) primarily used for?", options: ["Managing virtual networks", "Building, testing, and deploying AI solutions and agents", "Storing relational data", "Monitoring virtual machines"], answer: 1, explain: "Azure AI Foundry is the unified platform for building, evaluating, and deploying generative AI apps and agents." },
        { q: "You need to extract totals and line items from scanned invoices. Which service fits best?", options: ["Azure AI Vision OCR only", "Azure AI Document Intelligence", "Azure AI Translator", "Azure AI Speech"], answer: 1, explain: "Document Intelligence extracts structured fields, tables, and key-value pairs from documents like invoices - beyond plain OCR." },
        { q: "Which service makes large sets of company documents searchable and grounds generative AI?", options: ["Azure AI Search", "Azure Key Vault", "Azure Monitor", "Azure DNS"], answer: 0, explain: "Azure AI Search provides knowledge mining and vector/hybrid search, the retrieval layer behind RAG grounding." },
        { q: "To detect hate, violence, sexual, and self-harm content in text and images, use:", options: ["Azure AI Content Safety", "Azure AI Vision", "Azure AI Language", "Azure Machine Learning"], answer: 0, explain: "Azure AI Content Safety screens both input and output across the four harm categories with severity levels." },
        { q: "A solution needs several AI capabilities behind one key and endpoint. You should provision:", options: ["A single-service resource per capability", "A multi-service (Azure AI services/Foundry) resource", "A separate subscription each", "A virtual machine"], answer: 1, explain: "A multi-service resource shares one key and endpoint across capabilities, simplifying management for combined solutions." },
        { q: "For production authentication to Azure AI services, the recommended approach is:", options: ["Hardcoded API keys", "Microsoft Entra ID with managed identities", "Keys in the URL", "A public config file"], answer: 1, explain: "Entra ID with managed identities avoids storing secrets in code and supports least-privilege RBAC." },
        { q: "Training a model on labeled historical data to predict a category is an example of:", options: ["Unsupervised learning", "Supervised learning", "Reinforcement learning", "Clustering"], answer: 1, explain: "Supervised learning uses labeled examples; predicting a category is classification." },
        { q: "Which capability automatically tries many algorithms to find the best model with minimal code?", options: ["Automated ML (AutoML)", "MLflow", "Content Safety", "Data drift"], answer: 0, explain: "Automated ML in Azure Machine Learning searches algorithms and hyperparameters and ranks candidate models." },
        { q: "Converting speech in a meeting recording into text uses which service?", options: ["Azure AI Speech", "Azure AI Vision", "Azure AI Translator", "Azure AI Search"], answer: 0, explain: "Azure AI Speech provides speech-to-text (and text-to-speech) capabilities." },
        { q: "Grounding a generative model on your own data primarily aims to:", options: ["Increase token cost", "Reduce hallucinations with trusted context", "Retrain the base model", "Disable content filters"], answer: 1, explain: "Grounding supplies relevant, trusted context (often via Azure AI Search) so answers are accurate and less prone to hallucination." },
        { q: "Which Responsible AI principle means humans remain answerable for AI outcomes?", options: ["Transparency", "Accountability", "Inclusiveness", "Reliability & safety"], answer: 1, explain: "Accountability means people, not the AI, remain responsible for the system's outcomes." },
        { q: "Identifying the language of incoming text so it can be routed correctly is:", options: ["Language detection", "Object detection", "Clustering", "Tokenization"], answer: 0, explain: "Language detection identifies the language of text, a prebuilt feature of Azure AI Language/Translator." },
        { q: "A chatbot that extracts a user's intent and key details (entities) uses:", options: ["Conversational Language Understanding (CLU)", "OCR", "Anomaly detection", "Vector search"], answer: 0, explain: "CLU extracts intents and entities from utterances - the brain behind chatbots and voice assistants." }
    ],
    'panel-ab-731': [
        { q: "For AB-731, what should an AI transformation roadmap start with?", options: ["Selecting a model provider", "Buying GPU capacity", "Business outcomes and measurable value", "Building a chatbot pilot"], answer: 2, explain: "AB-731 focuses on outcome-first strategy: define business value first, then choose technology and delivery patterns." },
        { q: "Which operating model commonly balances enterprise standards with domain delivery speed?", options: ["Fully centralized", "Fully decentralized", "Hybrid model", "Vendor-managed only"], answer: 2, explain: "A hybrid model combines a shared AI platform/governance layer with domain-aligned product teams." },
        { q: "In AI governance, which control best supports auditability?", options: ["Informal approvals", "Undocumented prompt edits", "Comprehensive logging and traceability", "Disabling user feedback"], answer: 2, explain: "Auditability requires clear logs, decision trails, and evidence of policy enforcement across workflows." },
        { q: "What is the main purpose of an AI use-case portfolio rubric?", options: ["To maximize model size", "To prioritize initiatives by value, feasibility, speed, and risk", "To remove governance reviews", "To standardize job titles"], answer: 1, explain: "Portfolio rubrics help leaders consistently choose high-value, feasible use cases while managing risk." },
        { q: "Which metric is most directly tied to transformation adoption?", options: ["Cloud region count", "Weekly active users and task success by persona", "Total lines of prompt text", "Number of data sources connected"], answer: 1, explain: "AB-731 emphasizes adoption and value realization, so usage and task-level outcomes are key indicators." },
        { q: "The recommended way to launch AI initiatives at scale is to:", options: ["Deploy org-wide on day one", "Run measured pilots, then scale based on evidence", "Avoid metrics", "Skip governance"], answer: 1, explain: "Pilot-then-scale with measurement reduces risk and builds confidence before broad rollout." },
        { q: "An AI value case should always include:", options: ["Only the model name", "Target outcome, baseline, costs, and risks", "GPU brand", "The office location"], answer: 1, explain: "A defensible value case ties outcome and baseline to costs and risks." },
        { q: "Pairing value metrics with guardrail metrics ensures that:", options: ["Speed erodes trust", "Delivery speed does not compromise safety and satisfaction", "Costs are ignored", "Governance is removed"], answer: 1, explain: "Guardrail metrics (safety, satisfaction, error rates) keep speed from eroding trust." },
        { q: "A right-sized AI governance model should:", options: ["Block all initiatives", "Enable safe speed with clear decision rights and risk tiers", "Avoid documentation", "Remove approvals"], answer: 1, explain: "Good governance accelerates safe delivery via decision rights, risk tiers, and review gates." },
        { q: "FinOps for AI primarily helps a leader:", options: ["Increase token usage", "Govern and optimize AI spend and ROI", "Remove monitoring", "Hide costs"], answer: 1, explain: "FinOps brings budgets, showback, and right-sizing to control AI cost and improve ROI." },
        { q: "When a pilot underperforms against its metrics, a leader should:", options: ["Scale it anyway", "Pivot or stop based on evidence", "Ignore the result", "Remove the metrics"], answer: 1, explain: "Portfolio decisions (fund, scale, pivot, stop) should be evidence-based." },
        { q: "Sustained AI adoption depends most on:", options: ["Larger models", "People: change management, enablement, and champions", "More regions", "Longer prompts"], answer: 1, explain: "Transformation is sustained by people through change management and enablement." },
        { q: "A strong executive value update leads with:", options: ["Technical architecture diagrams", "Outcome vs target, ROI, risks controlled, and next funding", "Token counts", "Server names"], answer: 1, explain: "Outcome-first executive communication sustains sponsorship and funding." },
        { q: "A center of excellence in an AI operating model primarily provides:", options: ["Standards, reuse, and governance support", "Cheaper GPUs", "A CDN", "A DNS zone"], answer: 0, explain: "A CoE sets standards, promotes reuse, and supports governance across teams." }
    ],
    'panel-ai-103': [
        { q: "In a Retrieval-Augmented Generation (RAG) pattern, what does the retrieval step do?", options: ["Fine-tunes the model weights", "Fetches relevant context to ground the model's response", "Deploys the model to an endpoint", "Encrypts the prompt"], answer: 1, explain: "RAG retrieves relevant documents (often via a vector search) and injects them as context so the model answers from grounded facts, reducing hallucination." },
        { q: "Which Azure service is commonly used as the knowledge/vector store for RAG?", options: ["Azure Key Vault", "Azure AI Search", "Azure Monitor", "Azure DevOps"], answer: 1, explain: "Azure AI Search provides vector and hybrid search, making it the typical retrieval layer for RAG solutions." },
        { q: "What does tool calling (function calling) allow a large language model to do?", options: ["Increase its token limit", "Invoke external functions or APIs to take actions", "Automatically reduce latency", "Encrypt its outputs"], answer: 1, explain: "Tool/function calling lets the model request that your defined functions or APIs run, enabling agents to take real actions." },
        { q: "To keep credentials out of code when authenticating to Azure AI services, you should use:", options: ["Hardcoded API keys", "Microsoft Entra ID / managed identities", "A public config file", "Keys in the query string"], answer: 1, explain: "Microsoft Entra ID with managed identities avoids storing secrets in code and is the recommended authentication approach." },
        { q: "Which Azure AI Foundry capability helps you measure the quality and groundedness of generative responses?", options: ["Load balancing", "Evaluations", "Autoscaling", "Blob storage"], answer: 1, explain: "Foundry Evaluations score responses on metrics like groundedness, relevance, and coherence so you can compare and improve solutions." },
        { q: "Which service extracts fields and tables from invoices and forms?", options: ["Azure AI Document Intelligence", "Azure AI Translator", "Azure Key Vault", "Azure Monitor"], answer: 0, explain: "Document Intelligence extracts structured data (fields, tables, key-value pairs) from documents." },
        { q: "To transcribe large volumes of recorded call audio, use:", options: ["Batch transcription in Azure AI Speech", "Text-to-speech", "Azure AI Translator", "Azure AI Vision"], answer: 0, explain: "Batch transcription handles large volumes of recorded audio; real-time is for live scenarios." },
        { q: "To extract transcripts, faces, and topics from recorded video, use:", options: ["Azure AI Video Indexer", "Azure AI Search", "Azure Functions", "Azure DNS"], answer: 0, explain: "Azure AI Video Indexer extracts rich insights (transcripts, faces, topics, scenes) from recorded media." },
        { q: "The Foundry Agent Service primarily manages:", options: ["Threads, tool calls, and run state for agents", "Virtual networks", "Blob lifecycle", "DNS records"], answer: 0, explain: "The Foundry Agent Service handles conversation threads, tool invocation, and run state so you focus on agent behavior." },
        { q: "When your generic translations miss industry terms, you should use:", options: ["Custom Translator", "Text-to-speech", "OCR", "Anomaly detection"], answer: 0, explain: "Custom Translator adapts translation to domain-specific and brand terminology." },
        { q: "To handle Azure AI throttling (HTTP 429) gracefully, implement:", options: ["Exponential backoff and retries", "A larger font", "A public endpoint", "A second subscription only"], answer: 0, explain: "Retry with exponential backoff handles rate limiting (429) robustly." },
        { q: "For observability of an AI solution in production, enable:", options: ["Diagnostic settings to Azure Monitor / Log Analytics", "Cloud CDN", "A static website", "A new tenant"], answer: 0, explain: "Diagnostic settings send logs/metrics to Azure Monitor and Log Analytics for monitoring and alerts." },
        { q: "A prompt-shield guardrail primarily defends against:", options: ["Jailbreak/prompt-injection attempts", "Slow networks", "High storage cost", "DNS failures"], answer: 0, explain: "Prompt shields detect and block jailbreak and prompt-injection attempts on model input." },
        { q: "Vector (semantic) search retrieves content by:", options: ["Exact keyword only", "Meaning via nearest-neighbor on embeddings", "File size", "Alphabetical order"], answer: 1, explain: "Vector search compares embeddings by nearest-neighbor to retrieve semantically relevant content, often combined with keywords in hybrid search." }
    ],
    'panel-dp-100': [
        { q: "Which Azure Machine Learning concept packages your training code to run on remote compute?", options: ["Endpoint", "Job (experiment run)", "Datastore", "Workspace"], answer: 1, explain: "A job submits your training script to run on a specified compute target and tracks its outputs." },
        { q: "What is MLflow primarily used for in Azure ML?", options: ["Network security", "Tracking experiments, parameters, and models", "Billing", "Provisioning virtual machines"], answer: 1, explain: "MLflow provides experiment tracking, model logging, and packaging, and is integrated into Azure ML." },
        { q: "To serve a model for low-latency, real-time inference you deploy to a:", options: ["Batch endpoint", "Online (real-time) endpoint", "Datastore", "Pipeline"], answer: 1, explain: "Online endpoints provide scalable, low-latency REST inference; batch endpoints are for asynchronous scoring of large datasets." },
        { q: "Which capability automatically tries many algorithms and hyperparameters to find the best model?", options: ["Automated ML (AutoML)", "MLflow", "ONNX", "Data drift"], answer: 0, explain: "Automated ML searches across algorithms and hyperparameters and ranks candidate models by your chosen metric." },
        { q: "Monitoring how input data distribution changes over time is known as detecting:", options: ["Overfitting", "Data drift", "Underfitting", "Data leakage"], answer: 1, explain: "Data drift is a shift in the input distribution after deployment, which can degrade model performance and should be monitored." },
        { q: "DP-100 retired in June 2026. The recommended successor certification is:", options: ["AI-900", "AI-300 (ML Operations Engineer)", "DP-203", "AZ-900"], answer: 1, explain: "AI-300 (ML Operations Engineer Associate) is the modern successor covering MLOps and GenAIOps on Azure." },
        { q: "To tune hyperparameters natively in Azure ML, you run a:", options: ["Sweep job", "Datastore", "Workspace", "Blob container"], answer: 0, explain: "A sweep job wraps a command job to search a hyperparameter space and optimize a primary metric." },
        { q: "To measure and mitigate model unfairness across groups, use:", options: ["Fairlearn (in the Responsible AI dashboard)", "Cloud CDN", "Key Vault", "Load Balancer"], answer: 0, explain: "Fairlearn assesses disaggregated metrics and supports mitigation for fairness." },
        { q: "For cost-effective scoring of a large dataset that is not time-sensitive, use a:", options: ["Batch endpoint", "Real-time endpoint", "Datastore", "Notebook"], answer: 0, explain: "Batch endpoints score large datasets asynchronously and are cheaper than always-on online endpoints." },
        { q: "A reusable, versioned pipeline step (code + inputs + outputs + environment) is a:", options: ["Component", "Datastore", "Secret", "Region"], answer: 0, explain: "Components are self-contained, reusable pipeline steps that improve reproducibility." },
        { q: "Explaining which features drove a model's predictions is called:", options: ["Interpretability", "Sharding", "Caching", "Throttling"], answer: 0, explain: "Model interpretability produces feature-importance explanations for transparency and debugging." },
        { q: "A model with high overall accuracy but poor results for one subgroup is best surfaced by:", options: ["Error analysis", "A bigger VM", "A CDN", "A new region"], answer: 0, explain: "Error analysis reveals cohorts with high error rates that overall accuracy hides." },
        { q: "Safe rollout of a new model version alongside the old one uses:", options: ["Blue/green (traffic split) deployment", "A single variant only", "Manual file copy", "A DNS change"], answer: 0, explain: "Blue/green deployments route traffic gradually between versions for safe rollouts." },
        { q: "Registering a model in Azure ML primarily enables:", options: ["Versioning and governed, repeatable deployment", "Cheaper storage", "Faster DNS", "Automatic labeling"], answer: 0, explain: "The model registry versions models with lineage and provides a governed path to deployment." }
    ],
    'panel-aif': [
        { q: "Which AWS service provides access to multiple foundation models through a single API?", options: ["Amazon SageMaker", "Amazon Bedrock", "Amazon Comprehend", "Amazon Rekognition"], answer: 1, explain: "Amazon Bedrock offers serverless access to foundation models from multiple providers via one API, plus features like Knowledge Bases and Guardrails." },
        { q: "RAG is generally preferred over fine-tuning when you want to:", options: ["Permanently change the model's weights", "Ground responses on up-to-date proprietary data cost-effectively", "Train a model from scratch", "Shrink the model size"], answer: 1, explain: "RAG grounds answers on your current data without retraining, and is usually cheaper and faster to update than fine-tuning." },
        { q: "Which AWS service extracts text and structured data from scanned documents?", options: ["Amazon Polly", "Amazon Textract", "Amazon Translate", "Amazon Lex"], answer: 1, explain: "Amazon Textract extracts text, forms, and tables from documents; Polly is text-to-speech, Lex is chatbots, Translate is language translation." },
        { q: "Amazon Bedrock Guardrails are used to:", options: ["Speed up model training", "Filter harmful content and enforce safety policies", "Reduce storage costs", "Manage IAM roles"], answer: 1, explain: "Guardrails apply content filters, denied topics, and PII controls to generative AI applications." },
        { q: "Which of these is a generative AI use case?", options: ["Fraud classification", "Summarizing long documents", "Forecasting next month's sales", "Detecting anomalies"], answer: 1, explain: "Summarization generates new text from input, making it generative; the others are predictive/classification tasks." },
        { q: "To build and train a custom ML model with full control on AWS, you use:", options: ["Amazon SageMaker", "Amazon Rekognition", "Amazon Polly", "Amazon Translate"], answer: 0, explain: "Amazon SageMaker is the ML platform for building, training, and deploying custom models; the others are prebuilt AI services." },
        { q: "A managed way to implement retrieval-augmented generation on AWS is:", options: ["Amazon Bedrock Knowledge Bases", "Amazon SQS", "AWS Glue", "Amazon Route 53"], answer: 0, explain: "Bedrock Knowledge Bases handle chunking, embeddings, and retrieval to ground foundation models on your data." },
        { q: "Which Bedrock feature lets a model plan multi-step tasks and call APIs/tools?", options: ["Bedrock Agents", "Bedrock Guardrails", "Amazon S3", "Amazon CloudWatch"], answer: 0, explain: "Bedrock Agents orchestrate reasoning steps and invoke action groups (APIs/tools) to complete tasks." },
        { q: "For image and video analysis or content moderation, the managed service is:", options: ["Amazon Rekognition", "Amazon Comprehend", "Amazon Transcribe", "Amazon Kendra"], answer: 0, explain: "Amazon Rekognition analyzes images and video, including labels, faces, and moderation." },
        { q: "To convert customer-call audio into text, use:", options: ["Amazon Transcribe", "Amazon Polly", "Amazon Lex", "Amazon Personalize"], answer: 0, explain: "Amazon Transcribe is speech-to-text; Polly is text-to-speech, Lex builds chatbots, Personalize does recommendations." },
        { q: "Intelligent enterprise search across company documents is provided by:", options: ["Amazon Kendra", "Amazon SQS", "Amazon EC2", "Amazon VPC"], answer: 0, explain: "Amazon Kendra provides ML-powered intelligent search over enterprise content." },
        { q: "When choosing a foundation model, a key cost driver is:", options: ["The AWS Region name", "Tokens processed (input + output)", "The IAM role name", "The bucket name"], answer: 1, explain: "Generative AI cost is largely driven by tokens processed; smaller models are cheaper and faster." },
        { q: "Which is a limitation of large language models to plan for?", options: ["They never make mistakes", "They can hallucinate confident but wrong answers", "They have unlimited context", "They cost nothing to run"], answer: 1, explain: "LLMs can hallucinate, so grounding (RAG), guardrails, and human oversight matter." },
        { q: "The AWS shared responsibility model means, for AI services, AWS is responsible for:", options: ["Your prompt content", "Security of the underlying cloud infrastructure", "Your IAM policy design choices", "Your data classification"], answer: 1, explain: "AWS secures the cloud infrastructure; customers remain responsible for how they configure access, data, and usage." },
        { q: "To control who can call Bedrock or other AWS AI services, you primarily use:", options: ["AWS IAM", "Amazon CloudFront", "Amazon SNS", "AWS Batch"], answer: 0, explain: "AWS IAM controls authentication and least-privilege authorization for AWS services." },
        { q: "Providing a few examples inside the prompt to guide output is called:", options: ["Few-shot prompting", "Fine-tuning", "Distillation", "Sharding"], answer: 0, explain: "Few-shot prompting includes examples in the prompt to steer format and behavior without training." }
    ],
    'panel-mla': [
        { q: "Which SageMaker feature stores and serves curated ML features for training and inference?", options: ["SageMaker Feature Store", "SageMaker Ground Truth", "SageMaker Clarify", "SageMaker Neo"], answer: 0, explain: "Feature Store centralizes features so the same definitions are used consistently in training and serving, avoiding skew." },
        { q: "To detect bias and explain model predictions in SageMaker, you use:", options: ["SageMaker Clarify", "SageMaker Pipelines", "SageMaker Edge", "SageMaker Canvas"], answer: 0, explain: "SageMaker Clarify measures bias and produces feature-importance explanations for models." },
        { q: "Which service orchestrates repeatable, CI/CD-style ML workflows in SageMaker?", options: ["SageMaker Pipelines", "SageMaker Studio Lab", "Amazon Q", "SageMaker JumpStart"], answer: 0, explain: "SageMaker Pipelines defines and automates end-to-end ML workflows for reproducibility and CI/CD." },
        { q: "For cost-effective inference over a large dataset that is not time-sensitive, use:", options: ["A real-time endpoint", "Batch transform", "A serverless websocket", "Edge deployment"], answer: 1, explain: "Batch transform processes large datasets asynchronously and is more cost-effective than keeping a real-time endpoint running." },
        { q: "Large-scale data labeling for supervised learning is provided by:", options: ["SageMaker Ground Truth", "SageMaker Neo", "Feature Store", "Model Monitor"], answer: 0, explain: "SageMaker Ground Truth manages human and automated data labeling workflows." },
        { q: "To detect data and quality drift on a deployed SageMaker endpoint, use:", options: ["SageMaker Model Monitor", "SageMaker Neo", "SageMaker Canvas", "Amazon Q"], answer: 0, explain: "Model Monitor watches for data quality and drift so you can retrain before performance drops." },
        { q: "To govern which model versions can be deployed, use:", options: ["SageMaker Model Registry with approval status", "An S3 bucket policy only", "A CloudFront distribution", "A Route 53 record"], answer: 0, explain: "The Model Registry versions models and gates deployment via approval status." },
        { q: "To reduce training cost for interruptible jobs, use:", options: ["Spot (managed spot) training", "Always on-demand only", "A bigger endpoint", "A CDN"], answer: 0, explain: "Managed spot training uses spare capacity at lower cost for interruptible training jobs." },
        { q: "For spiky, unpredictable inference traffic with no idle cost, choose:", options: ["Serverless inference", "A 24/7 real-time endpoint", "Batch only", "Edge only"], answer: 0, explain: "Serverless inference scales to zero and suits spiky traffic without paying for idle endpoints." },
        { q: "Native CI/CD templates for ML on AWS are provided by:", options: ["SageMaker Projects (+ CodePipeline)", "Amazon Route 53", "Amazon CloudFront", "AWS Shield"], answer: 0, explain: "SageMaker Projects provision CI/CD (often with CodePipeline/CodeBuild) for MLOps." },
        { q: "For very large inference payloads processed asynchronously, use:", options: ["Asynchronous inference", "A websocket", "A CDN", "A DNS record"], answer: 0, explain: "Asynchronous inference handles large payloads and long processing without blocking." },
        { q: "Consistent features in training and serving are ensured by:", options: ["SageMaker Feature Store", "Amazon SQS", "Amazon SNS", "AWS Batch"], answer: 0, explain: "Feature Store centralizes feature definitions, preventing training-serving skew." },
        { q: "To control access to SageMaker resources with least privilege, use:", options: ["AWS IAM roles and policies", "A public S3 bucket", "A CDN", "A DNS zone"], answer: 0, explain: "IAM enforces least-privilege access to SageMaker and related resources." },
        { q: "A pipeline step that only registers a model if accuracy exceeds a threshold uses a:", options: ["Condition step", "DNS step", "CDN step", "Cache step"], answer: 0, explain: "A condition step gates registration/deployment on evaluation metrics." }
    ],
    'panel-genai-leader': [
        { q: "Google Cloud's unified platform for building and deploying ML and generative AI is:", options: ["BigQuery", "Vertex AI", "Cloud Spanner", "Dataproc"], answer: 1, explain: "Vertex AI is Google Cloud's end-to-end platform for training, tuning, and deploying models, including Gemini." },
        { q: "Grounding a Gemini model on your enterprise data primarily helps to:", options: ["Increase token cost", "Reduce hallucinations by supplying factual context", "Retrain the model from scratch", "Bypass safety filters"], answer: 1, explain: "Grounding supplies trusted, current data as context so responses are more accurate and less prone to hallucination." },
        { q: "Which is a key limitation of large language models that leaders should plan for?", options: ["Guaranteed factual accuracy", "The potential to hallucinate", "Unlimited context length", "Zero operating cost"], answer: 1, explain: "LLMs can generate confident but incorrect answers (hallucinations), so grounding and human oversight matter." },
        { q: "Converting text into numeric vectors that capture meaning for search and similarity uses:", options: ["Embeddings", "Tokenizers only", "Quantization", "Sharding"], answer: 0, explain: "Embeddings map text to vectors so semantically similar items are close together, powering semantic search and RAG." },
        { q: "From a governance standpoint, responsible use of generative AI on Google Cloud is supported by:", options: ["Cloud CDN", "Vertex AI safety filters and Responsible AI tooling", "Cloud NAT", "Pub/Sub"], answer: 1, explain: "Vertex AI provides safety filters and Responsible AI tooling to help enforce policies and reduce harmful outputs." },
        { q: "Google's flagship multimodal foundation model family is:", options: ["BERT", "Gemini", "PaLM 1", "ResNet"], answer: 1, explain: "Gemini is Google's multimodal model family, handling text, images, audio, and video." },
        { q: "A team needs image labeling for a common use case with no ML expertise. The best first choice is:", options: ["Custom training on Vertex AI", "The pre-trained Cloud Vision API", "Training a model in BigQuery ML", "Building a new neural network"], answer: 1, explain: "Pre-trained APIs like Cloud Vision solve common tasks instantly with no training - always prefer the simplest option that meets the need." },
        { q: "An analyst wants to build a churn model using data already in BigQuery, with SQL. Recommend:", options: ["BigQuery ML", "Dataproc", "Cloud Spanner", "Cloud CDN"], answer: 0, explain: "BigQuery ML lets you train and run models with SQL directly where the data lives, ideal for analysts and fast baselines." },
        { q: "In Vertex AI, the interface for designing and testing prompts without writing code is:", options: ["Cloud Shell", "Vertex AI Studio", "Cloud Build", "Artifact Registry"], answer: 1, explain: "Vertex AI Studio is where you design, test, and iterate on prompts before moving to code." },
        { q: "A catalog of Google, partner, and open models to discover and deploy is called:", options: ["Model Garden", "Cloud Storage", "Looker", "Firestore"], answer: 0, explain: "Model Garden in Vertex AI is the catalog for discovering, testing, and deploying models." },
        { q: "The prompting technique of asking the model to reason step by step is:", options: ["Zero-shot", "Chain-of-thought", "Quantization", "Sharding"], answer: 1, explain: "Chain-of-thought prompting asks the model to show step-by-step reasoning, improving results on complex tasks." },
        { q: "Providing a few input/output examples inside the prompt is known as:", options: ["Fine-tuning", "Few-shot prompting", "Distillation", "Grounding"], answer: 1, explain: "Few-shot prompting includes examples in the prompt to steer the model's format and behavior - no training required." },
        { q: "Which parameter most directly controls how random or creative model output is?", options: ["Temperature", "Replica count", "Region", "IAM role"], answer: 0, explain: "Temperature controls randomness: lower is more deterministic/factual, higher is more creative." },
        { q: "To let one small model learn from a larger model to cut cost and latency, you use:", options: ["Distillation", "Sharding", "Quantized indexing", "Federation"], answer: 0, explain: "Distillation trains a smaller, cheaper model to mimic a larger one, reducing cost and latency." },
        { q: "For fast semantic retrieval behind RAG on Google Cloud, use:", options: ["Vertex AI Vector Search", "Cloud NAT", "Cloud DNS", "Cloud Armor"], answer: 0, explain: "Vertex AI Vector Search performs fast nearest-neighbor lookups over embeddings, powering RAG retrieval." },
        { q: "To prevent data exfiltration around AI services, a leader should apply:", options: ["VPC Service Controls", "Cloud CDN", "Pub/Sub", "Cloud Scheduler"], answer: 0, explain: "VPC Service Controls create a security perimeter that helps prevent data exfiltration from Google Cloud services." },
        { q: "Enterprise customers often require that their prompts and data are NOT used to train models. On Vertex AI this is:", options: ["Impossible", "Provided via enterprise data governance and no-training guarantees", "Only on the free tier", "Handled by Cloud CDN"], answer: 1, explain: "Vertex AI offers enterprise data governance, including data residency and guarantees that customer data is not used to train foundation models." },
        { q: "Which is a Google AI Principle a leader should uphold?", options: ["Maximize engagement at all costs", "Avoid creating or reinforcing unfair bias", "Hide model limitations", "Skip human oversight"], answer: 1, explain: "Google's AI Principles include avoiding unfair bias, safety, accountability, and privacy." },
        { q: "When prioritizing generative AI use cases, a common leadership framework weighs:", options: ["Color scheme vs font", "Business value vs feasibility", "Region vs zone", "CPU vs GPU only"], answer: 1, explain: "Leaders prioritize use cases by business value against feasibility, then define clear success metrics." },
        { q: "The recommended way to scale a new generative AI initiative is to:", options: ["Deploy org-wide on day one", "Start with a measured pilot, then scale", "Avoid measuring outcomes", "Skip change management"], answer: 1, explain: "Start with pilots, measure outcomes, then scale - paired with executive sponsorship and change management." },
        { q: "To embed generative AI into everyday productivity apps with no build effort, an organization can use:", options: ["Gemini for Google Workspace", "Cloud Functions", "Bigtable", "Cloud Interconnect"], answer: 0, explain: "Gemini for Google Workspace adds AI directly into Gmail, Docs, Sheets, Slides, and Meet - fast value with nothing to build." }
    ],
    'panel-gcp-mle': [
        { q: "Which Vertex AI feature manages features for training/serving consistency?", options: ["Vertex AI Feature Store", "Cloud Armor", "Dataflow", "Looker"], answer: 0, explain: "Vertex AI Feature Store centralizes feature definitions to keep training and serving consistent." },
        { q: "To serve a model as an autoscaling REST endpoint on Vertex AI, you deploy it to a:", options: ["BigQuery table", "Vertex AI Endpoint", "Cloud Storage bucket", "Pub/Sub topic"], answer: 1, explain: "A Vertex AI Endpoint hosts a deployed model for online predictions with autoscaling." },
        { q: "Which managed service runs large-scale Apache Beam data pipelines on Google Cloud?", options: ["Dataflow", "Cloud SQL", "Firestore", "Memorystore"], answer: 0, explain: "Dataflow runs Apache Beam pipelines for scalable batch and streaming data processing." },
        { q: "Vertex AI Pipelines are used to:", options: ["Manage DNS", "Orchestrate reproducible ML workflows", "Host static websites", "Store secrets"], answer: 1, explain: "Vertex AI Pipelines orchestrate and track reproducible, containerized ML workflows." },
        { q: "To detect training-serving skew and data drift on a deployed model, use:", options: ["Vertex AI Model Monitoring", "Cloud Trace", "Cloud Build", "Artifact Registry"], answer: 0, explain: "Vertex AI Model Monitoring watches for skew and drift so you can retrain before performance degrades." },
        { q: "An analyst wants a model with SQL on data already in BigQuery. Recommend:", options: ["BigQuery ML", "Custom training containers", "Dataproc Spark", "Compute Engine VMs"], answer: 0, explain: "BigQuery ML trains models with SQL where the data lives - ideal for analysts and fast baselines." },
        { q: "For cost-effective scoring of millions of records that is not time-sensitive, use:", options: ["A real-time endpoint", "Vertex AI batch prediction", "A websocket", "Edge deployment"], answer: 1, explain: "Batch prediction processes large datasets asynchronously and is cheaper than keeping an online endpoint running." },
        { q: "Google's custom accelerators optimized for large-scale training are:", options: ["TPUs", "HDDs", "CDNs", "NICs"], answer: 0, explain: "TPUs (Tensor Processing Units) are Google's accelerators for large-scale ML training and inference." },
        { q: "To automate build-test-deploy of ML pipelines on Google Cloud, you typically use:", options: ["Cloud Build (or GitHub Actions)", "Cloud CDN", "Cloud NAT", "Cloud DNS"], answer: 0, explain: "Cloud Build (or GitHub Actions) triggers Vertex AI Pipelines for CI/CD in MLOps." },
        { q: "Choosing the simplest build approach that meets requirements usually means preferring:", options: ["Custom training first", "Pre-trained APIs / BQML / AutoML before full custom", "Always the largest model", "Manual notebooks only"], answer: 1, explain: "Prefer the lowest-effort option (APIs, BQML, AutoML) before custom training to reduce cost and effort." },
        { q: "To reproduce a deployed model's exact data and code lineage, rely on:", options: ["Vertex ML Metadata", "Cloud Armor", "Memorystore", "Cloud Interconnect"], answer: 0, explain: "Vertex ML Metadata records artifacts and executions, enabling lineage and reproducibility." },
        { q: "Applying the same feature transformations in training and serving prevents:", options: ["Training-serving skew", "Cold starts", "DNS caching", "Sharding"], answer: 0, explain: "Consistent transformations (e.g., via Feature Store) prevent training-serving skew." },
        { q: "Native hyperparameter tuning on Vertex AI is done with:", options: ["Vertex AI hyperparameter tuning (sweep) jobs", "Cloud DNS", "Pub/Sub", "Cloud CDN"], answer: 0, explain: "Vertex AI runs automated hyperparameter search over defined ranges to optimize a metric." },
        { q: "To secure ML workloads and prevent data exfiltration, a strong control is:", options: ["VPC Service Controls", "Cloud CDN", "Cloud Scheduler", "Cloud Tasks"], answer: 0, explain: "VPC Service Controls create a security perimeter that helps prevent data exfiltration." }
    ],
    'panel-agentforce': [
        { q: "In Agentforce, a topic primarily defines:", options: ["The UI colour theme", "A job the agent can handle, grouping related actions", "The billing tier", "The data center region"], answer: 1, explain: "A topic scopes a set of jobs the agent can do and groups the instructions and actions that handle them." },
        { q: "Agentforce actions can be built from:", options: ["Only Apex", "Flows, Apex, and prompt templates", "Only external REST APIs", "Only reports"], answer: 1, explain: "Actions can be backed by Flows, Apex, prompt templates, and other invocable capabilities." },
        { q: "The Einstein Trust Layer is designed to:", options: ["Speed up flows", "Provide data masking, grounding, and toxicity/audit controls", "Store files", "Manage user licenses"], answer: 1, explain: "The Einstein Trust Layer adds security and trust features such as data masking, secure grounding, zero data retention, and toxicity/audit logging." },
        { q: "Grounding a prompt template in Salesforce means:", options: ["Deploying it to production", "Injecting CRM or Data Cloud data into the prompt for context", "Deleting related records", "Scheduling a batch job"], answer: 1, explain: "Grounding merges relevant Salesforce/Data Cloud data into the prompt so the response is specific and accurate." },
        { q: "Which tool do you use to build and test prompt templates in Salesforce?", options: ["Flow Builder", "Prompt Builder", "Schema Builder", "Report Builder"], answer: 1, explain: "Prompt Builder is where you create, ground, and test reusable prompt templates." },
        { q: "Which Salesforce product unifies data into a real-time customer profile for grounding?", options: ["Data Cloud", "Experience Cloud", "Commerce Cloud", "Marketing Cloud"], answer: 0, explain: "Data Cloud unifies data via data streams, DMOs, and identity resolution to ground agents and prompts." },
        { q: "A retriever in Agentforce is used to:", options: ["Deploy code", "Find the most relevant content to ground a response", "Manage user permissions", "Schedule jobs"], answer: 1, explain: "Retrievers search a vector/search index and return relevant chunks that ground the model's answer (RAG)." },
        { q: "The Einstein Trust Layer feature that prevents the LLM provider from keeping your data is:", options: ["Data masking", "Zero data retention", "Toxicity scoring", "Prompt defense"], answer: 1, explain: "Zero data retention ensures the LLM provider does not store your prompts or responses." },
        { q: "To validate an agent's topic classification and action selection before launch, use:", options: ["Report Builder", "Agentforce Testing Center", "Schema Builder", "Data Loader"], answer: 1, explain: "The Agentforce Testing Center runs test cases to confirm the agent picks the right topics and actions." },
        { q: "When both fit a requirement, Salesforce best practice generally prefers:", options: ["Custom Apex over everything", "Declarative tools (Flow, prompt templates) before custom code", "External APIs only", "Manual processes"], answer: 1, explain: "Declarative building blocks are preferred before custom Apex when both can meet the requirement." },
        { q: "The Agentforce Service Agent is designed primarily to:", options: ["Write Apex triggers", "Resolve customer service cases autonomously", "Manage sandboxes", "Build dashboards"], answer: 1, explain: "The Agentforce Service Agent handles customer service interactions, grounded in Knowledge and case data, escalating to humans when needed." },
        { q: "Instructions within a topic are used to:", options: ["Set the org's currency", "Guide how the agent behaves for that job", "Define page layouts", "Control license counts"], answer: 1, explain: "Instructions guide the agent's behavior within a topic, improving accuracy and appropriate action selection." },
        { q: "Sensitive fields being hidden before a prompt reaches the LLM is an example of:", options: ["Data masking", "Sharding", "Indexing", "Caching"], answer: 0, explain: "Data masking in the Einstein Trust Layer protects sensitive/PII fields before content reaches the model." },
        { q: "Grounding an agent in Salesforce Knowledge articles primarily improves:", options: ["Deployment speed", "Answer accuracy and relevance", "License usage", "Report formatting"], answer: 1, explain: "Grounding in curated Knowledge (via retrievers) makes agent responses accurate and relevant." }
    ],
    'panel-ai-300': [
        { q: "What is the primary goal of MLOps?", options: ["To build more models faster only", "To standardize and automate the ML lifecycle: build, deploy, and monitor", "To replace data scientists", "To reduce the number of cloud regions"], answer: 1, explain: "MLOps applies DevOps practices to machine learning so models are reproducibly built, deployed, monitored, and retrained." },
        { q: "Which tool would you use to automate build-test-deploy (CI/CD) workflows for Azure Machine Learning?", options: ["Power BI", "GitHub Actions", "Azure Monitor alone", "Microsoft Excel"], answer: 1, explain: "GitHub Actions (or Azure Pipelines) can trigger Azure ML jobs to automate training and deployment as part of CI/CD." },
        { q: "After a model is deployed, detecting data drift primarily tells you when to:", options: ["Delete the endpoint", "Retrain or update the model", "Raise the price", "Change the Azure region"], answer: 1, explain: "Drift signals that incoming data no longer matches the training data, so the model should be retrained to stay accurate." },
        { q: "In GenAIOps, evaluating a generative model for 'groundedness' measures whether:", options: ["Responses are fast", "Responses are supported by the provided source data", "The model is small", "Tokens are cheap"], answer: 1, explain: "Groundedness checks that the model's answer is backed by the supplied context rather than fabricated." },
        { q: "Registering a model in the Azure ML model registry mainly enables:", options: ["Faster GPUs", "Versioning and governed, repeatable deployment", "Cheaper blob storage", "Automatic data labeling"], answer: 1, explain: "The model registry versions models and provides a governed path to deploy a specific, tracked version." },
        { q: "To provision repeatable dev/test/prod ML environments, AI-300 emphasizes:", options: ["Manual portal clicks", "Infrastructure as code with Bicep and the Azure CLI", "Copying VMs by hand", "Editing DNS"], answer: 1, explain: "IaC (Bicep + Azure CLI) gives repeatable, reviewable environments - a core AI-300 practice." },
        { q: "For keyless, least-privilege access between Azure services, use:", options: ["Managed identities with RBAC", "Shared passwords", "Public keys in code", "A single admin account"], answer: 0, explain: "Managed identities plus RBAC avoid secrets and enforce least privilege." },
        { q: "To isolate an ML workspace from the public internet, use:", options: ["Private endpoints and VNet integration", "A larger VM", "A CDN", "A new subscription"], answer: 0, explain: "Private endpoints and VNet isolation restrict network access for sensitive workloads." },
        { q: "GenAIOps orchestration of prompts, retrieval, and post-processing uses:", options: ["Prompt flow in Microsoft Foundry", "Cloud CDN", "Blob lifecycle rules", "DNS zones"], answer: 0, explain: "Prompt flow orchestrates and evaluates the steps of a generative AI application." },
        { q: "When compliance requires you to control encryption keys, choose:", options: ["Customer-managed keys (CMEK)", "No encryption", "Keys in source control", "Plaintext storage"], answer: 0, explain: "CMEK lets you control encryption keys to meet regulatory requirements." },
        { q: "To catch harmful generative output in production, apply:", options: ["Azure AI Content Safety filters", "A bigger context window", "More tokens", "A CDN"], answer: 0, explain: "Content Safety screens input and output; combined with monitoring it enforces guardrails." },
        { q: "Tracing a generative request end to end for debugging uses:", options: ["Foundry tracing + Application Insights", "Cloud NAT", "Blob storage", "Pub/Sub"], answer: 0, explain: "Foundry tracing (OpenTelemetry) with Application Insights makes generative systems observable." },
        { q: "Automated retraining should typically be triggered by:", options: ["Drift detection or new data", "A manual email", "Changing the region", "A larger font"], answer: 0, explain: "Retraining is triggered by drift, new data, or schedules - closing the MLOps loop." },
        { q: "To reduce generative AI cost in production, a leader/engineer should:", options: ["Right-size models, cache, and trim prompts", "Always use the largest model", "Disable monitoring", "Ignore token usage"], answer: 0, explain: "Right-sizing models, caching, and trimming prompts/context reduce token cost and latency." }
    ],
    'panel-ai-200': [
        { q: "Which Azure service is a serverless container platform well-suited to hosting microservices and AI apps?", options: ["Azure Container Apps", "Azure Blob Storage", "Azure Monitor", "Azure Key Vault"], answer: 0, explain: "Azure Container Apps runs containerized microservices serverlessly with built-in scaling, ideal for AI back-ends." },
        { q: "For a globally distributed, low-latency NoSQL data store for your app, you would use:", options: ["Azure Cosmos DB", "Azure Files", "Azure Queue Storage", "Azure DevOps"], answer: 0, explain: "Azure Cosmos DB is a globally distributed, low-latency NoSQL database with multiple APIs." },
        { q: "Which architectural pattern decouples services so they communicate through messages/events?", options: ["A single monolith", "Event-driven architecture", "Synchronous-only calls", "Batch-only processing"], answer: 1, explain: "Event-driven architectures use events/messages (e.g., via Event Grid or Service Bus) to loosely couple and scale services independently." },
        { q: "To store vector embeddings for RAG inside a relational database on Azure, you can use:", options: ["Azure Database for PostgreSQL with the pgvector extension", "Azure Table Storage", "Azure Queue Storage", "Azure CDN"], answer: 0, explain: "Azure Database for PostgreSQL supports the pgvector extension for storing and searching embeddings." },
        { q: "Which service securely stores secrets, keys, and connection strings for a cloud app?", options: ["Azure Key Vault", "Azure Container Registry", "Azure Monitor", "Azure Front Door"], answer: 0, explain: "Azure Key Vault centrally and securely stores secrets, keys, and certificates, keeping them out of code." },
        { q: "When calling Azure OpenAI from an app, you invoke the:", options: ["Raw model name", "Deployment name and endpoint", "Storage account", "Resource group"], answer: 1, explain: "Applications call the deployment name at the resource endpoint, secured with managed identity in production." },
        { q: "To handle throttling (HTTP 429) from an AI endpoint under load, implement:", options: ["Exponential backoff and retries", "A larger font", "A public bucket", "A DNS change"], answer: 0, explain: "Retry with exponential backoff (and queue-based load leveling) keeps AI apps reliable under load." },
        { q: "To smooth traffic spikes and decouple slow AI processing, use:", options: ["Azure Service Bus queues", "Azure CDN", "Azure DNS", "Azure Policy"], answer: 0, explain: "Service Bus queues buffer work and decouple producers from slower consumers." },
        { q: "To react automatically when a document is uploaded to Blob Storage, use:", options: ["Azure Event Grid", "Azure Key Vault", "Azure Monitor", "Azure Front Door"], answer: 0, explain: "Event Grid routes events (like blob-created) to handlers such as Functions for reactive processing." },
        { q: "Streaming an LLM response to the client mainly improves:", options: ["Perceived latency for chat UIs", "Storage cost", "DNS resolution", "Disk IO"], answer: 0, explain: "Streaming tokens as they arrive improves perceived responsiveness in conversational apps." },
        { q: "For very complex, large-scale orchestration with GPUs, the best host is:", options: ["Azure Kubernetes Service (AKS)", "Azure Table Storage", "Azure DNS", "Azure CDN"], answer: 0, explain: "AKS provides full orchestration and GPU support for complex, large-scale workloads." },
        { q: "To cache frequent model responses/embeddings and cut cost/latency, use:", options: ["Azure Managed Redis", "Azure DNS", "Azure Policy", "Azure Front Door"], answer: 0, explain: "Azure Managed Redis caches responses/embeddings and provides fast lookups." },
        { q: "To keep AI credentials out of code, an app should use:", options: ["Managed identity + Key Vault references", "Hardcoded keys", "A public config file", "Keys in the URL"], answer: 0, explain: "Managed identities with Key Vault references eliminate secrets in code." },
        { q: "To instrument an AI app for latency, dependencies, and token usage, use:", options: ["Application Insights (Azure Monitor)", "Azure CDN", "Azure DNS", "Azure Policy"], answer: 0, explain: "Application Insights captures requests, dependencies, exceptions, and custom metrics for observability." }
    ],
    'panel-genai-dev': [
        { q: "Amazon Bedrock Knowledge Bases are primarily used to implement:", options: ["Model pre-training", "Retrieval-Augmented Generation (RAG)", "Billing reports", "IAM policies"], answer: 1, explain: "Knowledge Bases connect foundation models to your data sources to power managed RAG workflows." },
        { q: "Which Bedrock feature lets a model plan multi-step tasks and call tools/APIs to complete them?", options: ["Bedrock Agents", "Bedrock Guardrails", "Amazon S3", "Amazon CloudWatch"], answer: 0, explain: "Bedrock Agents orchestrate reasoning steps and invoke actions/APIs to complete tasks." },
        { q: "To store and query vector embeddings on AWS for semantic search, you could use:", options: ["Amazon OpenSearch Service vector engine", "Amazon SQS", "AWS Glue only", "Amazon Route 53"], answer: 0, explain: "Amazon OpenSearch Service supports vector search, a common vector store for RAG on AWS." },
        { q: "Fine-tuning is more appropriate than RAG when you need to:", options: ["Add frequently-changing facts", "Adapt the model's style or behavior to a domain using labeled examples", "Avoid any training entirely", "Only shorten prompts"], answer: 1, explain: "Fine-tuning adjusts model weights to specialize tone/behavior; RAG is better for injecting current, changing knowledge." },
        { q: "Amazon Bedrock Guardrails help a generative application by:", options: ["Increasing token limits", "Blocking disallowed topics and filtering harmful content and PII", "Speeding up GPUs", "Managing VPC networking"], answer: 1, explain: "Guardrails enforce content policies, denied topics, and PII redaction on inputs and outputs." },
        { q: "To let a Bedrock Agent call your own API or Lambda, you configure a(n):", options: ["Action group", "Route table", "Security group", "Placement group"], answer: 0, explain: "Action groups define the APIs/Lambdas an agent can invoke to take actions." },
        { q: "To measure whether generated answers are supported by retrieved context, evaluate:", options: ["Groundedness", "CPU usage", "Disk IO", "DNS latency"], answer: 0, explain: "Groundedness measures whether the response is backed by the provided context, not fabricated." },
        { q: "For end-to-end tracing and metrics of a GenAI app on AWS, use:", options: ["CloudWatch and X-Ray", "Route 53", "CloudFront", "Shield"], answer: 0, explain: "CloudWatch (logs/metrics) and X-Ray (tracing) provide observability for production apps." },
        { q: "To audit who invoked Bedrock and when, use:", options: ["AWS CloudTrail", "Amazon SQS", "Amazon SNS", "AWS Batch"], answer: 0, explain: "CloudTrail records API activity for auditing and governance." },
        { q: "A cost-effective default when a smaller model meets quality targets is to:", options: ["Use the smaller/cheaper model and cache results", "Always use the largest model", "Disable monitoring", "Increase max tokens"], answer: 0, explain: "Right-sizing models and caching reduce cost and latency while meeting quality targets." },
        { q: "To protect data and restrict access for a Bedrock app, apply:", options: ["IAM least privilege, KMS encryption, and VPC endpoints", "A public S3 bucket", "A CDN only", "A DNS record"], answer: 0, explain: "IAM, KMS, and private networking (VPC endpoints) secure the generative application." },
        { q: "Streaming responses in a chat app primarily improves:", options: ["Perceived latency", "Storage cost", "IAM policy size", "DNS caching"], answer: 0, explain: "Streaming tokens as they generate improves perceived responsiveness." },
        { q: "To compare two prompts or models objectively, you should:", options: ["Run an evaluation set with scored metrics", "Guess", "Pick the newer one", "Choose at random"], answer: 0, explain: "Evaluation datasets with metrics (relevance, groundedness) enable objective comparison." },
        { q: "Provisioned throughput on Bedrock is best used when:", options: ["You need guaranteed capacity/latency at scale", "Traffic is near zero", "Only for testing", "To reduce security"], answer: 0, explain: "Provisioned throughput guarantees capacity for high, steady workloads; on-demand suits variable traffic." }
    ],
    'panel-mls': [
        { q: "Which SageMaker built-in algorithm is well-suited to classification and regression on tabular data?", options: ["XGBoost", "BlazingText", "Seq2Seq", "Object2Vec"], answer: 0, explain: "XGBoost is a gradient-boosted tree algorithm that excels on structured/tabular problems." },
        { q: "Which technique helps reduce overfitting?", options: ["Increasing the learning rate indefinitely", "Regularization, dropout, and gathering more data", "Removing the validation set", "Training for unlimited epochs"], answer: 1, explain: "Regularization, dropout, early stopping, and more data all reduce variance and overfitting." },
        { q: "Which metric is most appropriate for a highly imbalanced binary classifier?", options: ["Raw accuracy", "F1 score (or AUC-PR)", "Mean squared error", "R-squared"], answer: 1, explain: "With class imbalance, accuracy is misleading; F1 (and precision-recall AUC) better reflect performance on the minority class." },
        { q: "SageMaker automatic model tuning finds good hyperparameters by:", options: ["Only manual trial and error", "Automated search (e.g., Bayesian or random) over defined ranges", "Never searching", "Deleting features at random"], answer: 1, explain: "SageMaker hyperparameter tuning runs an automated search strategy over the ranges you specify." },
        { q: "To deploy two model versions and split live traffic between them, you use:", options: ["A single production variant", "Multiple production variants (A/B testing) on one endpoint", "Batch transform", "SageMaker Ground Truth"], answer: 1, explain: "An endpoint can host multiple production variants with weighted traffic to A/B test models in production." }
    ],
    'panel-sap-caig': [
        { q: "Which SAP BTP service provides governed access to multiple foundation models through one harmonized interface?", options: ["Generative AI Hub in SAP AI Core", "SAP Fiori", "SAP Build Apps", "SAP Signavio"], answer: 0, explain: "The Generative AI Hub in SAP AI Core gives trusted, governed access to leading foundation models behind one API and SDK." },
        { q: "In SAP AI Core, what is the difference between a configuration and a deployment?", options: ["They are identical", "A configuration is the reusable recipe; a deployment is the running, callable instance", "A deployment is the recipe; a configuration is the endpoint", "Neither is used for models"], answer: 1, explain: "A configuration defines a workload (model, scenario, parameters); a deployment is the running instance that exposes an inference endpoint." },
        { q: "SAP AI Launchpad is primarily used to:", options: ["Host your application backend code", "Operate, monitor, and experiment with AI across AI Core instances", "Replace SAP HANA Cloud", "Store source code repositories"], answer: 1, explain: "AI Launchpad is the central operations and experimentation UI, not a place to host application code." },
        { q: "SAP AI Core isolates artifacts and secrets per tenant using:", options: ["Resource groups", "Route tables", "Fiori tiles", "Transport requests"], answer: 0, explain: "Resource groups provide tenant-level isolation of artifacts, configurations, deployments, and secrets." },
        { q: "In chat-style prompting, durable rules and persona should go in the:", options: ["User message", "System message", "Assistant message", "Function result"], answer: 1, explain: "The system message sets behavior, persona, and guardrails; the user message carries the specific task." },
        { q: "For factual, deterministic output you should generally set temperature:", options: ["High", "Low", "It has no effect", "To max_tokens"], answer: 1, explain: "Lower temperature reduces randomness, which suits factual and deterministic tasks." },
        { q: "Embeddings are best described as:", options: ["Compressed images", "Numeric vectors that capture the meaning of text for similarity search", "Encrypted passwords", "SQL indexes only"], answer: 1, explain: "Embeddings map text to vectors where similar meanings are near each other, enabling semantic search." },
        { q: "Where can you natively store and search vector embeddings alongside business data on SAP?", options: ["SAP HANA Cloud Vector Engine", "SAP GUI", "SAP Web IDE", "ABAP Dictionary"], answer: 0, explain: "SAP HANA Cloud provides a native REAL_VECTOR type and similarity search for embeddings." },
        { q: "Retrieval-Augmented Generation (RAG) primarily helps by:", options: ["Increasing token limits", "Grounding responses in retrieved, trusted content to reduce hallucinations", "Training the model from scratch", "Encrypting the network"], answer: 1, explain: "RAG retrieves relevant context and adds it to the prompt so answers stay accurate and current." },
        { q: "The SAP AI Core orchestration service lets you combine, in one call:", options: ["Only model inference", "Templating, model configuration, grounding, and content filtering", "Only data masking", "Networking and DNS"], answer: 1, explain: "Orchestration composes templating, grounding, model calls, and filtering into a single managed workflow." },
        { q: "Function calling enables a model to:", options: ["Retrain itself", "Return a structured request so your app can run a tool/API and feed results back", "Delete deployments", "Bypass authentication"], answer: 1, explain: "Function calling produces structured tool calls, the foundation of agentic workflows that take real actions." },
        { q: "To prevent sensitive PII from reaching a foundation model, you use:", options: ["Data masking", "Provisioned throughput", "A larger context window", "A CDN"], answer: 0, explain: "Data masking removes or obfuscates PII before prompts reach the model, part of SAP's trust guardrails." },
        { q: "Content filtering in the Generative AI Hub is applied to:", options: ["Only outputs", "Only inputs", "Both inputs and outputs", "Neither"], answer: 2, explain: "Content filtering can screen both prompts (inputs) and generated responses (outputs) for harmful content." },
        { q: "Which developer option lets you switch models with minimal code changes?", options: ["The harmonized API/SDK of the Generative AI Hub", "Hard-coding one provider's SDK", "Editing ABAP reports", "Rewriting the app per model"], answer: 0, explain: "The harmonized interface abstracts providers, so changing models requires little code change." },
        { q: "SAP's responsible AI principles emphasize:", options: ["Maximum automation with no oversight", "Human oversight, transparency, fairness, and data privacy", "Ignoring bias", "Removing audit logs"], answer: 1, explain: "SAP's ethical AI principles stress human agency, transparency, fairness/bias mitigation, and privacy." }
    ],
    'panel-java-programming': [
        { q: "In Java, which keyword is used to define behavior shared by implementing classes?", options: ["extends", "interface", "package", "record"], answer: 1, explain: "An interface defines a contract that classes implement." },
        { q: "Which collection preserves insertion order and allows duplicates?", options: ["HashSet", "ArrayList", "TreeSet", "HashMap"], answer: 1, explain: "ArrayList keeps insertion order and allows duplicates." },
        { q: "What is the primary purpose of try-catch in Java?", options: ["Increase compilation speed", "Handle runtime exceptions safely", "Create threads", "Define generics"], answer: 1, explain: "try-catch handles exceptional runtime conditions." },
        { q: "In stream processing, filter is typically used to:", options: ["Transform each element", "Sort a collection", "Keep elements matching a condition", "Convert to Optional"], answer: 2, explain: "filter retains only elements that satisfy a predicate." },
        { q: "Which executor utility is preferred over manual thread creation for task management?", options: ["Thread.sleep", "ExecutorService", "System.gc", "StringBuilder"], answer: 1, explain: "ExecutorService manages thread pools and task execution." },
        { q: "For production backend Java, what is usually the safest default for shared data?", options: ["Global mutable state", "Immutability where possible", "Public fields everywhere", "Single giant class"], answer: 1, explain: "Immutability reduces concurrency and state bugs." },
        { q: "What does JVM bytecode enable?", options: ["Direct SQL execution", "Platform portability", "No compilation", "Automatic UI rendering"], answer: 1, explain: "Bytecode runs on any platform with a compatible JVM." },
        { q: "Which design choice improves maintainability most in large Java systems?", options: ["Long methods with many branches", "Clear package boundaries and small cohesive classes", "No interfaces", "All logic in main()"], answer: 1, explain: "Separation of concerns and cohesion improve maintainability." }
    ],
    'panel-python-programming': [
        { q: "Why is indentation important in Python?", options: ["Only for style", "It defines code blocks syntactically", "It improves runtime speed", "It replaces imports"], answer: 1, explain: "Python uses indentation to define block structure." },
        { q: "Which structure is best for key-value pairs?", options: ["list", "tuple", "dict", "set"], answer: 2, explain: "dict is Python's key-value mapping type." },
        { q: "What is a major advantage of using with open(...) as f:?", options: ["Automatic file cleanup", "Faster CPU execution", "Skips encoding", "Disables exceptions"], answer: 0, explain: "with ensures resources are released safely." },
        { q: "Which statement about exceptions is most professional?", options: ["Catch every exception silently", "Use precise exceptions and clear handling", "Avoid exceptions entirely", "Use print instead of errors"], answer: 1, explain: "Specific and meaningful exception handling improves reliability." },
        { q: "Virtual environments are used primarily to:", options: ["Increase internet speed", "Isolate dependencies per project", "Replace testing", "Compile Python to C automatically"], answer: 1, explain: "Virtual environments prevent dependency conflicts across projects." },
        { q: "In API-heavy Python work, which pattern is essential?", options: ["Ignore status codes", "Parse JSON and handle failures explicitly", "Always retry forever", "Disable logging"], answer: 1, explain: "Reliable API integrations require response validation and failure handling." },
        { q: "Comprehensions are best used when:", options: ["Readability remains clear", "You want maximum nesting", "You need side effects", "You avoid functions"], answer: 0, explain: "Comprehensions should remain readable and focused." },
        { q: "What separates production Python from casual scripting?", options: ["Longer file names", "Testing, logging, structure, and validation", "Only using classes", "Avoiding libraries"], answer: 1, explain: "Professional code emphasizes reliability, observability, and maintainability." }
    ],
    'panel-csharp-programming': [
        { q: "C# code typically runs under which runtime component?", options: ["JVM", "CLR", "V8", "CPython"], answer: 1, explain: "C# runs on the .NET Common Language Runtime (CLR)." },
        { q: "What is a common purpose of interfaces in C#?", options: ["Store files", "Define contracts for implementations", "Replace constructors", "Disable dependency injection"], answer: 1, explain: "Interfaces define behavior contracts and support abstraction." },
        { q: "LINQ mainly helps by:", options: ["Creating DNS records", "Querying collections declaratively", "Compiling C# faster", "Managing NuGet packages"], answer: 1, explain: "LINQ provides expressive querying and transformations over data." },
        { q: "Which async pattern is preferred in modern C#?", options: ["Thread.Abort", "async/await with Task", "Blocking on .Result everywhere", "Polling loops"], answer: 1, explain: "async/await with Task is the recommended asynchronous model." },
        { q: "Dependency injection primarily reduces:", options: ["Typing speed", "Coupling between components", "Need for tests", "Compilation errors"], answer: 1, explain: "DI decouples dependencies and improves testability." },
        { q: "A strong habit for API development in C# is:", options: ["Use DTOs and validation boundaries", "Return raw database entities always", "Keep all logic in controllers", "Avoid service layers"], answer: 0, explain: "DTOs and layered boundaries keep APIs stable and maintainable." },
        { q: "What is a key risk of using blocking calls in async flows?", options: ["Better scalability", "Deadlocks and thread starvation", "Automatic retries", "Lower memory usage"], answer: 1, explain: "Blocking in async code can cause deadlocks and reduce throughput." },
        { q: "Which approach aligns with maintainable .NET architecture?", options: ["Single-file monolith logic", "Clear layers with focused responsibilities", "No interfaces", "Public mutable state everywhere"], answer: 1, explain: "Layered architecture with explicit responsibilities is easier to evolve." }
    ],
    'panel-cpp-programming': [
        { q: "Why is C++ often chosen for systems programming?", options: ["Automatic scaling", "High performance and low-level control", "Built-in cloud APIs", "No compilation required"], answer: 1, explain: "C++ offers strong performance with direct control over resources." },
        { q: "What does a pointer store?", options: ["A function body", "A memory address", "A namespace", "A thread handle only"], answer: 1, explain: "Pointers store memory addresses." },
        { q: "RAII helps primarily by:", options: ["Skipping destructors", "Binding resource cleanup to object lifetime", "Avoiding classes", "Removing templates"], answer: 1, explain: "RAII ensures deterministic cleanup through object scope/lifetime." },
        { q: "Which STL container is a dynamic array-like sequence?", options: ["std::vector", "std::map", "std::set", "std::stack only"], answer: 0, explain: "std::vector is the standard dynamic contiguous sequence container." },
        { q: "When possible, which allocation style is safer in modern C++?", options: ["Manual new/delete everywhere", "Smart pointers and value semantics", "Global raw pointers", "malloc in classes"], answer: 1, explain: "Modern C++ favors smart pointers and value-based ownership." },
        { q: "What is a major concurrency concern in C++?", options: ["Only syntax errors", "Shared mutable state race conditions", "Header guards", "Linker flags"], answer: 1, explain: "Race conditions arise when shared state is mutated across threads." },
        { q: "Why reserve vector capacity in performance-critical loops?", options: ["To increase recursion", "To reduce reallocations", "To disable iterators", "To force heap fragmentation"], answer: 1, explain: "reserve avoids repeated reallocations and copies." },
        { q: "What best defines professional C++ optimization practice?", options: ["Guess first", "Profile, measure, then optimize", "Inline everything", "Avoid STL"], answer: 1, explain: "Evidence-driven optimization prevents premature complexity." }
    ]
};

const CERTIFICATE_PASS_RATIO = 0.7;
const CERTIFICATE_STORAGE_KEY = 'cloudvana-issued-certificates-v1';
const CERTIFICATE_NAME_KEY = 'cloudvana-learner-name-v1';
const COURSE_CREDENTIAL_SECTIONS = [
    'courses',
    'track-ai-eng',
    'track-dsa-sd',
    'track-os-net'
];
const COURSE_CREDENTIAL_PANELS = [
    'panel-java-programming',
    'panel-python-programming',
    'panel-csharp-programming',
    'panel-cpp-programming'
];

function getIssuedCertificates() {
    return window.CloudvanaStorage.readObject(CERTIFICATE_STORAGE_KEY);
}

function getIssuedCertificate(panelId) {
    const record = getIssuedCertificates()[panelId];
    if (record === undefined) return null;
    if (record && typeof record.learnerName === 'string' && record.learnerName.trim() &&
        typeof record.certificateId === 'string' && record.certificateId &&
        typeof record.issuedAt === 'string' && !Number.isNaN(Date.parse(record.issuedAt))) return record;
    window.CloudvanaStorage.reportInvalidData(new TypeError('Invalid completion record for ' + panelId));
    return null;
}

function setIssuedCertificate(panelId, certificate) {
    const issued = getIssuedCertificates();
    issued[panelId] = certificate;
    window.CloudvanaStorage.set(CERTIFICATE_STORAGE_KEY, JSON.stringify(issued));
}

function getSavedLearnerName() {
    return window.CloudvanaStorage.get(CERTIFICATE_NAME_KEY) || '';
}

function setSavedLearnerName(name) {
    window.CloudvanaStorage.set(CERTIFICATE_NAME_KEY, name);
}

function getCertificatePanel(panelId) {
    return document.getElementById(panelId);
}

function getCertificateTitle(panel) {
    if (!panel) return 'Certification Track';
    const sectionTitle = panel.querySelector(':scope > .container > .section-title, :scope > .section-title');
    if (sectionTitle) {
        return sectionTitle.textContent.replace(/\s+/g, ' ').trim();
    }
    const title = panel.querySelector('.category-title');
    return title ? title.textContent.replace(/\s+/g, ' ').trim() : 'Certification Track';
}

function getCertificateProvider(panel) {
    const section = panel ? panel.closest('section[data-provider]') : null;
    return section ? (section.getAttribute('data-provider') || '').toLowerCase() : '';
}

function getCredentialKind(panelId) {
    // Keep language panels as course certificates even when quizzes exist.
    if (isCoursePanelCredential(panelId)) {
        return 'certificate';
    }
    return quizData[panelId] ? 'readiness' : 'certificate';
}

function requiresQuizForCredential(panelId) {
    return !!quizData[panelId];
}

function isCoursePanelCredential(panelId) {
    return COURSE_CREDENTIAL_PANELS.indexOf(panelId) >= 0;
}

function getAllCredentialPanelIds() {
    return Array.from(new Set(Object.keys(quizData).concat(COURSE_CREDENTIAL_SECTIONS, COURSE_CREDENTIAL_PANELS)));
}

function getCredentialTarget(panelId) {
    if (isCoursePanelCredential(panelId) || quizData[panelId]) {
        return document.getElementById(panelId);
    }
    return document.getElementById(panelId);
}

function getCertificateBadgeText(panelId, title, provider, kind) {
    var normalizedTitle = String(title || '').toLowerCase();
    var byProvider = {
        azure: 'Azure AI Track',
        aws: 'AWS AI Track',
        gcp: 'Google Cloud AI Track',
        salesforce: 'Salesforce AI Track',
        sap: 'SAP AI Track'
    };

    if (kind === 'readiness') {
        if (/fundamentals|practitioner|leader|foundational|beginner/.test(normalizedTitle)) {
            return 'Foundation Readiness';
        }
        return byProvider[provider] || 'Exam Readiness';
    }

    if (/foundation|fundamentals|beginner/.test(normalizedTitle)) {
        return 'Course Completion';
    }
    if (/java|python|c#|c\+\+/.test(normalizedTitle)) {
        return 'Language Certificate';
    }
    if (/advanced|professional|system design|agents|frameworks|engineering/.test(normalizedTitle)) {
        return 'Advanced Course';
    }
    if (/dsa|networking|operating systems|os|machine learning|data/.test(normalizedTitle)) {
        return 'Learning Track';
    }

    if (/fundamentals|practitioner|leader|foundational|beginner/.test(normalizedTitle)) {
        return 'Foundation Credential';
    }
    if (/professional|expert|ml engineer|engineer|developer|associate|specialist/.test(normalizedTitle)) {
        return byProvider[provider] || 'Professional Credential';
    }
    return byProvider[provider] || 'Cloud Credential';
}

function getCertificateSummary(panelId) {
    const panel = getCredentialTarget(panelId);
    const moduleCards = panel ? Array.from(panel.querySelectorAll('.module-card')) : [];
    const completedModules = moduleCards.filter(card => isModuleRead(card.dataset.moduleId)).length;
    const totalModules = moduleCards.length;
    const kind = getCredentialKind(panelId);
    const totalQuestions = (quizData[panelId] || []).length;
    const bestScore = getQuizBest(panelId);
    const passScore = totalQuestions ? Math.ceil(totalQuestions * CERTIFICATE_PASS_RATIO) : 0;
    const provider = getCertificateProvider(panel);
    const title = getCertificateTitle(panel);
    const requiresQuiz = requiresQuizForCredential(panelId);
    const eligible = totalModules > 0 && completedModules === totalModules && (!requiresQuiz || bestScore >= passScore);

    return {
        panel: panel,
        kind: kind,
        title: title,
        provider: provider,
        badgeText: getCertificateBadgeText(panelId, title, provider, kind),
        completedModules: completedModules,
        totalModules: totalModules,
        bestScore: bestScore,
        totalQuestions: totalQuestions,
        passScore: passScore,
        requiresQuiz: requiresQuiz,
        eligible: eligible
    };
}

function buildCertificateId(panelId) {
    const panelCode = panelId.replace('panel-', '').replace(/[^a-z0-9]+/gi, '').toUpperCase();
    return 'CV-' + panelCode + '-' + Date.now().toString(36).toUpperCase();
}

function formatCertificateDate(isoString) {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return isoString;
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function renderCertificateSection(panelId) {
    const summary = getCertificateSummary(panelId);
    const panel = summary.panel;
    if (!panel) return;

    const sectionContainer = summary.kind === 'readiness' || isCoursePanelCredential(panelId)
        ? panel
        : (panel.querySelector('.container') || panel);

    let section = sectionContainer.querySelector(':scope > .cert-issue-section');
    if (!section) {
        section = document.createElement('div');
        section.className = 'cert-issue-section';
        if (summary.kind === 'readiness' || isCoursePanelCredential(panelId)) {
            panel.appendChild(section);
        } else {
            const tabs = sectionContainer.querySelector('.section-tabs');
            if (tabs && tabs.nextSibling) {
                sectionContainer.insertBefore(section, tabs.nextSibling);
            } else if (tabs) {
                sectionContainer.appendChild(section);
            } else {
                sectionContainer.appendChild(section);
            }
        }
    }

    const issued = getIssuedCertificate(panelId);
    const moduleState = summary.totalModules
        ? summary.completedModules + '/' + summary.totalModules + ' modules completed'
        : 'No modules found';
    const quizState = summary.requiresQuiz
        ? 'Best quiz score ' + summary.bestScore + '/' + summary.totalQuestions + ' (need ' + summary.passScore + '+' + ')'
        : 'All modules must be completed to unlock the certificate';
    const artifactName = summary.kind === 'readiness' ? 'Readiness Badge' : 'Certificate';
    const artifactHeading = summary.kind === 'readiness' ? 'Exam Readiness Badge' : 'Course Completion Certificate';
    const issuedStateText = summary.kind === 'readiness' ? 'Badge already claimed on this device' : 'Certificate already issued on this device';
    const readyStateText = summary.kind === 'readiness' ? 'Eligible to claim readiness badge' : 'Eligible to issue certificate';
    const lockedStateText = summary.kind === 'readiness'
        ? 'Complete the prep track to unlock your readiness badge'
        : (summary.requiresQuiz
            ? 'Complete the course track and pass the quiz to unlock certificate issuance'
            : 'Complete the course track to unlock certificate issuance');
    const statusText = issued
        ? issuedStateText
        : summary.eligible
            ? readyStateText
            : lockedStateText;
    const buttonText = issued
        ? 'View Issued ' + artifactName
        : (summary.kind === 'readiness' ? 'Claim Readiness Badge' : 'Issue Certificate');
    const buttonDisabled = issued || summary.eligible ? '' : 'disabled';
    const claimNoun = summary.kind === 'readiness' ? 'readiness badge' : 'certificate';
    const formId = panelId + '-claim-form';
    const nameId = panelId + '-learner-name';
    const hintId = panelId + '-claim-hint';
    const feedbackId = panelId + '-claim-feedback';
    const draftName = section.querySelector('.cert-claim-name')?.value ?? getSavedLearnerName();
    const formOpen = !issued && summary.eligible && section.querySelector('.cert-claim-form')?.hidden === false;
    const claimFormHtml = issued ? '' : `
        <form class="cert-claim-form" id="${formId}" novalidate ${formOpen ? '' : 'hidden'}>
            <label for="${nameId}">Name on ${claimNoun}</label>
            <input class="cert-claim-name" id="${nameId}" name="learnerName" type="text" autocomplete="name" value="${escapeHtml(draftName)}" aria-describedby="${hintId} ${feedbackId}" required>
            <p class="cert-claim-hint" id="${hintId}">Use the name you want displayed on your completion record.</p>
            <div class="cert-claim-actions">
                <button class="btn btn-primary" type="submit">Create ${claimNoun}</button>
                <button class="btn btn-secondary cert-claim-cancel" type="button">Cancel</button>
            </div>
        </form>`;
    const storageNoteHtml = '<p class="cert-storage-note">Cloudvana completion records are self-reported learning milestones, not official vendor certifications or proof of exam readiness.' +
        (issued ? (!window.CloudvanaStorage.persistent ? ' Browser storage is unavailable; this record is only available on this page.' : ' This record is saved in this browser, not synced to an account.') : ' Progress is stored in this browser when browser storage is available.') + '</p>';
    const issuedHtml = issued ? `
        <div class="cert-issued-card">
            <div class="cert-issued-ornament cert-issued-ornament--tl" aria-hidden="true"></div>
            <div class="cert-issued-ornament cert-issued-ornament--tr" aria-hidden="true"></div>
            <div class="cert-issued-ornament cert-issued-ornament--bl" aria-hidden="true"></div>
            <div class="cert-issued-ornament cert-issued-ornament--br" aria-hidden="true"></div>
            <div class="cert-issued-watermark" aria-hidden="true">
                <span class="cert-issued-watermark-mark">CV</span>
                <span class="cert-issued-watermark-text">Cloudvana</span>
            </div>
            <div class="cert-issued-topline">
                <span class="cert-issued-kicker">Cloudvana Learning Milestone</span>
                <span class="cert-issued-badge">${escapeHtml(summary.badgeText)}</span>
            </div>
            <div class="cert-issued-body">
                <span class="cert-issued-caption">${summary.kind === 'readiness' ? 'Readiness Badge' : 'Certificate of Completion'}</span>
                <h4 class="cert-issued-recipient">${escapeHtml(issued.learnerName)}</h4>
                <p class="cert-issued-copy">${summary.kind === 'readiness' ? 'has successfully completed the exam readiness requirements for' : 'has successfully completed the Cloudvana course track'}</p>
                <p class="cert-issued-track">${escapeHtml(summary.title)}</p>
                <div class="cert-issued-divider" aria-hidden="true"></div>
            </div>
            <div class="cert-issued-grid">
                <div>
                    <span class="cert-issued-label">Certificate ID</span>
                    <strong class="cert-issued-value cert-issued-id">${escapeHtml(issued.certificateId)}</strong>
                </div>
                <div>
                    <span class="cert-issued-label">Issued On</span>
                    <strong class="cert-issued-value">${escapeHtml(formatCertificateDate(issued.issuedAt))}</strong>
                </div>
            </div>
            <div class="cert-issued-footer">
                <div class="cert-issued-signature">
                    <span class="cert-issued-signature-mark">Cloudvana</span>
                    <span class="cert-issued-signature-role">Recorded by Cloudvana Learning</span>
                </div>
                <div class="cert-issued-seal" aria-hidden="true">
                    <span class="cert-issued-seal-ring"></span>
                    <span class="cert-issued-seal-text">Completed</span>
                </div>
            </div>
        </div>` : '';

    section.innerHTML = `
        <div class="cert-issue-head">
            <div>
                <h3 class="cert-issue-title"><i class="fas fa-award"></i> ${artifactHeading}</h3>
                <p class="cert-issue-sub">${statusText}</p>
            </div>
            <span class="cert-issue-pill ${issued ? 'is-issued' : summary.eligible ? 'is-ready' : 'is-locked'}">${issued ? 'Issued' : summary.eligible ? 'Ready' : 'Locked'}</span>
        </div>
        <div class="cert-issue-progress">
            <span>${moduleState}</span>
            <span>${quizState}</span>
        </div>
        <div class="cert-issue-actions">
            <button type="button" class="btn btn-primary cert-issue-btn" data-panel-id="${escapeHtml(panelId)}" ${issued ? '' : `aria-controls="${formId}" aria-expanded="${formOpen}"`} ${buttonDisabled}>${buttonText}</button>
        </div>
        <p class="cert-claim-feedback" id="${feedbackId}" role="alert"></p>
        ${claimFormHtml}
        ${issuedHtml}
        ${storageNoteHtml}
    `;

    const button = section.querySelector('.cert-issue-btn');
    const form = section.querySelector('.cert-claim-form');
    const nameInput = section.querySelector('.cert-claim-name');
    const feedback = section.querySelector('.cert-claim-feedback');
    enhanceExternalLinks(section);
    if (!button) return;

    button.addEventListener('click', function () {
        const existing = getIssuedCertificate(panelId);
        if (existing) {
            const issuedCard = section.querySelector('.cert-issued-card');
            if (issuedCard) issuedCard.scrollIntoView({ behavior: prefersReducedMotion() ? 'instant' : 'smooth', block: 'nearest' });
            return;
        }

        const current = getCertificateSummary(panelId);
        if (!current.eligible) {
            feedback.textContent = lockedStateText;
            return;
        }

        form.hidden = false;
        button.setAttribute('aria-expanded', 'true');
        feedback.textContent = '';
        nameInput.removeAttribute('aria-invalid');
        nameInput.focus();
        nameInput.select();
    });

    if (form) {
        nameInput.addEventListener('input', function () {
            nameInput.removeAttribute('aria-invalid');
            feedback.textContent = '';
        });
        form.querySelector('.cert-claim-cancel').addEventListener('click', function () {
            form.hidden = true;
            button.setAttribute('aria-expanded', 'false');
            feedback.textContent = '';
            button.focus();
        });
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            if (!getCertificateSummary(panelId).eligible) {
                feedback.textContent = lockedStateText;
                return;
            }

            const learnerName = nameInput.value.trim();
            if (!learnerName) {
                feedback.textContent = 'Enter your name to create the ' + claimNoun + '.';
                nameInput.setAttribute('aria-invalid', 'true');
                nameInput.focus();
                return;
            }

            setSavedLearnerName(learnerName);
            setIssuedCertificate(panelId, {
                learnerName: learnerName,
                certificateId: buildCertificateId(panelId),
                issuedAt: new Date().toISOString()
            });

            renderCertificateSection(panelId);
            section.querySelector('.cert-issue-btn').focus({ preventScroll: true });
        });
    }
}

function renderAllCertificateSections() {
    getAllCredentialPanelIds().forEach(renderCertificateSection);
}

function refreshCertificateSectionForModule(moduleId) {
    const card = document.querySelector('.module-card[data-module-id="' + moduleId.replace(/"/g, '\\"') + '"]');
    const panel = card ? card.closest('.tab-panel') : null;
    const section = card ? card.closest('section') : null;
    if (panel && panel.id && quizData[panel.id]) {
        renderCertificateSection(panel.id);
        return;
    }
    if (panel && panel.id && isCoursePanelCredential(panel.id)) {
        renderCertificateSection(panel.id);
        return;
    }
    if (section && section.id && COURSE_CREDENTIAL_SECTIONS.indexOf(section.id) >= 0) {
        renderCertificateSection(section.id);
    }
}

function loadQuizzes() {
    Object.keys(quizData).forEach(panelId => {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        const questions = quizData[panelId];
        const best = getQuizBest(panelId);

        const section = document.createElement('div');
        section.className = 'quiz-section';
        section.innerHTML = `
            <div class="quiz-head">
                <h3 class="quiz-heading"><i class="fas fa-circle-question"></i> Practice Questions</h3>
                <span class="quiz-score" role="status">Best: ${best}/${questions.length}</span>
            </div>
            <p class="quiz-sub">Test your understanding with ${questions.length} practice questions. Finish every question to save your best score in this browser. These are not official exam questions.</p>
            <div class="quiz-list"></div>
            <button type="button" class="btn btn-secondary btn-sm quiz-reset">Reset answers</button>
        `;

        const list = section.querySelector('.quiz-list');
        questions.forEach((item, qi) => {
            const q = document.createElement('fieldset');
            q.className = 'quiz-q';
            q.dataset.answer = item.answer;
            const opts = item.options.map((opt, oi) =>
                `<button type="button" class="quiz-opt" data-i="${oi}">${String.fromCharCode(65 + oi)}. ${escapeHtml(opt)}</button>`
            ).join('');
            q.innerHTML = `
                <legend class="quiz-q-text">${qi + 1}. ${escapeHtml(item.q)}</legend>
                <div class="quiz-opts">${opts}</div>
                <p class="quiz-explain" id="${panelId}-feedback-${qi}" aria-live="polite" hidden>${escapeHtml(item.explain)}</p>
            `;
            list.appendChild(q);
        });

        panel.appendChild(section);
        enhanceExternalLinks(section);
        setupQuizSection(section, panelId, questions.length);
        section.querySelector('.quiz-reset').addEventListener('click', function () {
            section.querySelectorAll('.quiz-q').forEach(question => {
                question.classList.remove('answered', 'is-correct');
                question.querySelectorAll('.quiz-opt').forEach(option => {
                    option.classList.remove('correct', 'incorrect', 'locked');
                    option.removeAttribute('aria-disabled');
                    option.removeAttribute('aria-describedby');
                    option.querySelector('.quiz-option-result')?.remove();
                });
                question.querySelector('.quiz-explain').hidden = true;
            });
            updateQuizScore(section, panelId, questions.length);
            section.querySelector('.quiz-opt').focus();
        });
    });

    renderAllCertificateSections();
}

function setupQuizSection(section, panelId, total) {
    section.querySelectorAll('.quiz-q').forEach(q => {
        const answer = parseInt(q.dataset.answer, 10);
        const opts = q.querySelectorAll('.quiz-opt');
        opts.forEach(opt => {
            opt.addEventListener('click', () => {
                if (q.classList.contains('answered')) return;
                q.classList.add('answered');
                const chosen = parseInt(opt.dataset.i, 10);
                opts[answer].classList.add('correct');
                if (chosen === answer) {
                    q.classList.add('is-correct');
                } else {
                    opt.classList.add('incorrect');
                }
                opts.forEach(option => {
                    option.classList.add('locked');
                    option.setAttribute('aria-disabled', 'true');
                });
                opts[answer].insertAdjacentHTML('beforeend', '<span class="quiz-option-result">Correct answer</span>');
                if (chosen !== answer) opt.insertAdjacentHTML('beforeend', '<span class="quiz-option-result">Your answer</span>');
                const ex = q.querySelector('.quiz-explain');
                if (ex) {
                    ex.hidden = false;
                    opt.setAttribute('aria-describedby', ex.id);
                }
                updateQuizScore(section, panelId, total);
            });
        });
    });
}

function updateQuizScore(section, panelId, total) {
    const correct = section.querySelectorAll('.quiz-q.is-correct').length;
    const answered = section.querySelectorAll('.quiz-q.answered').length;
    const best = answered === total ? Math.max(getQuizBest(panelId), correct) : getQuizBest(panelId);
    if (answered === total) setQuizBest(panelId, best);
    const scoreEl = section.querySelector('.quiz-score');
    if (scoreEl) {
        scoreEl.textContent = answered === total
            ? `Score: ${correct}/${total} · Best: ${best}/${total}`
            : `Answered: ${answered}/${total} · Best: ${best}/${total}`;
    }

    renderCertificateSection(panelId);
}

function getQuizBest(panelId) {
    const raw = window.CloudvanaStorage.get('cloudvana-quiz-' + panelId);
    if (raw === null) return 0;
    const value = Number(raw);
    if (Number.isInteger(value) && value >= 0 && value <= (quizData[panelId] || []).length) return value;
    window.CloudvanaStorage.reportInvalidData(new TypeError('Invalid quiz score for ' + panelId));
    return 0;
}

function setQuizBest(panelId, value) {
    window.CloudvanaStorage.set('cloudvana-quiz-' + panelId, String(value));
}

document.addEventListener('DOMContentLoaded', function() {
    loadModules();
    loadQuizzes();
    setupModuleDialog();
});

// Load all modules into their respective grids
function loadModules() {
    loadModulesIntoGrid('ml-models-grid', courseData.mlModels);
    loadModulesIntoGrid('data-grid', courseData.data);
    loadModulesIntoGrid('advanced-ml-grid', courseData.advancedML);
    loadModulesIntoGrid('realworld-ml-grid', courseData.realWorldML);
    loadModulesIntoGrid('ai-fundamentals-grid', courseData.aiFundamentals);
    loadModulesIntoGrid('azure-ai-transformation-leader-grid', courseData.azureAiTransformationLeader);
    loadModulesIntoGrid('ai-apps-agents-grid', courseData.aiAppsAgents);
    loadModulesIntoGrid('azure-mlops-grid', courseData.azureMlOps);
    loadModulesIntoGrid('azure-ai-clouddev-grid', courseData.azureAiCloudDev);
    loadModulesIntoGrid('aws-ai-practitioner-grid', courseData.awsAiPractitioner);
    loadModulesIntoGrid('aws-ml-engineer-grid', courseData.awsMlEngineer);
    loadModulesIntoGrid('aws-genai-developer-grid', courseData.awsGenAiDeveloper);
    loadModulesIntoGrid('gcp-genai-leader-grid', courseData.gcpGenAiLeader);
    loadModulesIntoGrid('gcp-mle-grid', courseData.gcpMlEngineer);
    loadModulesIntoGrid('salesforce-agentforce-grid', courseData.salesforceAgentforce);
    loadModulesIntoGrid('sap-genai-developer-grid', courseData.sapGenAiDeveloper);
    // AI Engineering: Zero to Hero (defined in js/ai-courses.js, Courses page only)
    loadModulesIntoGrid('deep-learning-grid', courseData.deepLearning);
    loadModulesIntoGrid('generative-ai-grid', courseData.generativeAI);
    loadModulesIntoGrid('ai-agents-grid', courseData.aiAgents);
    loadModulesIntoGrid('agent-frameworks-grid', courseData.agentFrameworks);
    loadModulesIntoGrid('dsa-foundations-grid', courseData.dsaFoundations);
    loadModulesIntoGrid('dsa-interview-grid', courseData.dsaInterview);
    loadModulesIntoGrid('system-design-foundations-grid', courseData.systemDesignFoundations);
    loadModulesIntoGrid('system-design-advanced-grid', courseData.systemDesignAdvanced);
    loadModulesIntoGrid('os-fundamentals-grid', courseData.osFundamentals);
    loadModulesIntoGrid('os-processes-memory-grid', courseData.osProcessesMemory);
    loadModulesIntoGrid('networking-fundamentals-grid', courseData.networkingFundamentals);
    loadModulesIntoGrid('networking-protocols-grid', courseData.networkingProtocols);
    loadModulesIntoGrid('java-programming-grid', courseData.javaProgramming);
    loadModulesIntoGrid('python-programming-grid', courseData.pythonProgramming);
    loadModulesIntoGrid('csharp-programming-grid', courseData.csharpProgramming);
    loadModulesIntoGrid('cpp-programming-grid', courseData.cppProgramming);
}

// Load modules into a specific grid
function loadModulesIntoGrid(gridId, modules) {
    const grid = document.getElementById(gridId);
    if (!grid || !modules) return; // grid or data not present on this page

    // Keep existing progress keys while disambiguating repeated legacy labels.
    const seenNumbers = new Set();
    modules.forEach((module, index) => {
        const duplicate = seenNumbers.has(module.number);
        const suffix = duplicate ? '::' + module.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') : '';
        const moduleId = `${gridId}#${module.number}${suffix}`;
        seenNumbers.add(module.number);
        const displayModule = duplicate ? { ...module, number: module.number.replace(/\d+$/, String(index + 1)) } : module;
        const card = createModuleCard(displayModule, gridId, moduleId);
        grid.appendChild(card);
    });
}

// ---- Module read/progress tracking (non-blocking, saved on this device) ----
function getReadSet() {
    return window.CloudvanaStorage.readObject('cloudvana-read');
}

function isModuleRead(id) {
    const value = getReadSet()[id];
    if (value === undefined) return false;
    if (value === 1) return true;
    window.CloudvanaStorage.reportInvalidData(new TypeError('Invalid module progress for ' + id));
    return false;
}

function setModuleRead(id, value) {
    const set = getReadSet();
    if (value) {
        set[id] = 1;
    } else {
        delete set[id];
    }
    window.CloudvanaStorage.set('cloudvana-read', JSON.stringify(set));
}

function toggleModuleRead(id, btn) {
    const now = !isModuleRead(id);
    setModuleRead(id, now);
    document.querySelectorAll('.module-card').forEach(card => {
        if (card.dataset.moduleId === id) {
            card.classList.toggle('read', now);
            card.querySelector('.module-status').textContent = now ? 'Completed' : 'Open module';
            card.querySelector('.module-open .sr-only').textContent = ', ' + card.dataset.moduleNumber + (now ? ', completed' : '');
        }
    });
    if (btn) {
        btn.classList.toggle('is-read', now);
        btn.setAttribute('aria-pressed', String(now));
        btn.innerHTML = now
            ? '<i class="fas fa-circle-check" aria-hidden="true"></i> Completed'
            : '<i class="far fa-circle" aria-hidden="true"></i> Mark as complete';
    }

    refreshCertificateSectionForModule(id);
}

const DEEP_DIVE_GRIDS = [
    'deep-learning-grid',
    'generative-ai-grid',
    'ai-agents-grid',
    'agent-frameworks-grid',
    'dsa-foundations-grid',
    'dsa-interview-grid',
    'system-design-foundations-grid',
    'system-design-advanced-grid',
    'os-fundamentals-grid',
    'os-processes-memory-grid',
    'networking-fundamentals-grid',
    'networking-protocols-grid',
    'java-programming-grid',
    'python-programming-grid',
    'csharp-programming-grid',
    'cpp-programming-grid'
];

function getAutoDeepDiveSections(module, gridId) {
    if (DEEP_DIVE_GRIDS.indexOf(gridId) === -1) return [];

    // Keep programming-language courses focused on authored curriculum content only.
    if (/-programming-grid$/.test(String(gridId || ''))) {
        return [];
    }

    // Rich modules already include a full structured lesson flow,
    // so avoid appending generic checklist/lab/pitfall sections.
    if (Array.isArray(module.detailedContent) && module.detailedContent.length >= 12) {
        return [];
    }

    const topics = Array.isArray(module.topics) ? module.topics : [];
    const topicLines = topics.length
        ? topics.map(function (t) { return '- ' + t; }).join('\n')
        : '- Core concepts\n- Practical implementation\n- Validation and review';

    const isDsa = gridId.indexOf('dsa-') === 0;
    const isSystemDesign = gridId.indexOf('system-design-') === 0;
    const isOs = gridId.indexOf('os-') === 0;
    const isNetworking = gridId.indexOf('networking-') === 0;
    const isProgrammingLanguage = /^(java|python|csharp|cpp)-/.test(gridId);
    const isAiEngineering = !isDsa && !isSystemDesign && !isOs && !isNetworking && !isProgrammingLanguage;

    let practiceTitle;
    let practiceContent;
    let pitfallsTitle;
    let pitfallsContent;

    if (isSystemDesign) {
        practiceTitle = 'Guided Design Exercise';
        practiceContent = `Run this module as a 90-minute design round:

1. 10 min: clarify functional and non-functional requirements.
2. 20 min: capacity estimation and key assumptions.
3. 25 min: high-level architecture and data flow.
4. 20 min: bottlenecks, failure modes, and mitigations.
5. 15 min: trade-offs, rollout strategy, and observability.

Deliverable:
• one high-level architecture diagram
• one API contract sketch
• one risk-and-mitigation table`;
        pitfallsTitle = 'Common Design Pitfalls to Avoid';
        pitfallsContent = `Watch for these design pitfalls:
• skipping requirements and assumptions
• over-designing before sizing and constraints
• ignoring reliability and failure scenarios
• weak trade-off discussion (cost, latency, consistency)

Reviewer signal:
balanced architecture, explicit assumptions, and operational readiness.`;
    } else if (isDsa) {
        practiceTitle = 'Guided Practice Sprint';
        practiceContent = `Run this module as a 75-minute coding sprint:

1. 15 min: concept refresh from notes.
2. 25 min: solve two easy/medium problems with dry run.
3. 20 min: implement one problem end-to-end without hints.
4. 15 min: complexity review + edge-case testing.

Checkpoint:
• explain the optimal approach in plain language
• state time/space complexity confidently
• identify one common failure pattern`;
        pitfallsTitle = 'Common Interview Mistakes to Avoid';
        pitfallsContent = `Watch for these interview mistakes:
• jumping to code before clarifying constraints
• missing edge cases (empty input, duplicates, bounds)
• weak complexity explanation
• not discussing trade-offs between approaches

Interviewer signal:
clear reasoning + structured communication + correct complexity analysis.`;
    } else if (isOs) {
        practiceTitle = 'Hands-On OS Lab';
        practiceContent = `Turn this module into a 60-minute hands-on lab on your own machine:

1. 15 min: observe the concept live (e.g., list running processes, check memory usage, watch CPU load).
2. 20 min: change one variable (start/stop a heavy process, run parallel tasks) and note the effect.
3. 15 min: map what you observed back to the OS concept in this module.
4. 10 min: write a 5-line summary of how the OS managed the resource.

Checkpoint:
• connect the theory to something you can actually see on a computer
• explain what the OS did behind the scenes`;
        pitfallsTitle = 'Common Misconceptions to Clear Up';
        pitfallsContent = `Clear up these frequent OS misunderstandings:
• confusing a program (on disk) with a process (running)
• confusing storage (disk) with memory (RAM)
• assuming more threads always means more speed
• thinking virtual memory is "fake" memory rather than an address mapping

Understanding signal:
you can explain how the OS shares limited hardware safely and efficiently.`;
    } else if (isNetworking) {
        practiceTitle = 'Hands-On Networking Lab';
        practiceContent = `Turn this module into a 60-minute practical lab:

1. 15 min: inspect the concept live (e.g., run a DNS lookup, check your IP, open browser dev-tools network tab).
2. 20 min: trace a real request and identify each layer/step involved.
3. 15 min: break something safely (wrong URL, blocked port) and read the error/status.
4. 10 min: summarize how the data traveled from your device to the server and back.

Checkpoint:
• connect the theory to a real request you can observe
• name the protocols involved at each step`;
        pitfallsTitle = 'Common Misconceptions to Clear Up';
        pitfallsContent = `Clear up these frequent networking misunderstandings:
• confusing IP (which machine) with port (which app)
• thinking DNS and the web server are the same thing
• assuming HTTPS only "hides" data (it also verifies identity + integrity)
• treating TCP and UDP as interchangeable

Understanding signal:
you can trace a request end-to-end and justify protocol choices.`;
    } else if (isProgrammingLanguage) {
        practiceTitle = 'Hands-On Coding Lab';
        practiceContent = `Turn this module into a 75-minute coding lab:

1. 15 min: restate the concept in your own words.
2. 20 min: type the example code from scratch without copy-paste.
3. 20 min: modify the example to support one new requirement.
4. 20 min: explain the design, edge cases, and one improvement you would make.

Checkpoint:
• you can write the code from memory
• you can explain why the language feature exists
• you can adapt it to a new use case`;
        pitfallsTitle = 'Professional Coding Pitfalls to Avoid';
        pitfallsContent = `Avoid these common language-learning mistakes:
• memorizing syntax without understanding the runtime model
• copying examples without rewriting them yourself
• ignoring error handling and edge cases
• learning isolated features without building complete programs

Pro signal:
you can design, code, debug, and explain a small solution independently.`;
    } else {
        // AI Engineering (deep learning, generative AI, agents, frameworks)
        practiceTitle = 'Hands-On AI Build Sprint';
        practiceContent = `Run this module as a 75-minute build sprint:

1. 15 min: restate the concept and where it fits in an AI system.
2. 25 min: build or run a minimal example (model call, pipeline step, or agent action).
3. 20 min: change one parameter or input and observe how behavior changes.
4. 15 min: evaluate output quality and note one improvement.

Checkpoint:
• explain the concept in plain language
• show a working minimal example
• describe one quality or safety consideration`;
        pitfallsTitle = 'Production Pitfalls & Quality Checks';
        pitfallsContent = `Before moving on, verify:
• can you explain this concept to a beginner in 2 minutes?
• can you build a minimal working example from memory?
• can you identify failure modes (hallucination, cost, latency, safety)?

Senior-engineer signal:
you can justify design and model choices, not just repeat definitions.`;
    }

    return [
        {
            title: 'Mastery Checklist',
            content: `By the end of this module, you should be able to:\n\n${topicLines}\n\nSelf-check:\n• Can I explain each topic in simple language?\n• Can I apply it to a new problem?\n• Can I justify my approach and trade-offs?`
        },
        {
            title: practiceTitle,
            content: practiceContent
        },
        {
            title: pitfallsTitle,
            content: pitfallsContent
        }
    ];
}

function getPhaseTwoExpansionSections(module, gridId) {
    const key = gridId + '#' + module.number;
    const expansions = {
        'deep-learning-grid#Module 1': [
            {
                title: 'Deep-Dive Example: Churn Prediction Network',
                content: `Scenario: you are predicting customer churn from tabular features (usage, tenure, payment method, support tickets).

What to design:
• input layer matching feature count
• two hidden layers with ReLU
• binary output with sigmoid

Engineering checks:
• normalize numeric features
• one-hot encode categorical values
• track precision/recall (not accuracy alone) for imbalanced classes.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Build a 2-layer network and print output shape for a sample batch.

Medium:
Train on a small dataset and compare learning rates (1e-2 vs 1e-3).

Hard:
Diagnose overfitting by plotting train/validation loss and propose two fixes.

Submission standard:
• code runs end-to-end
• complexity/shape reasoning is documented
• at least one failure case is explained.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why do hidden layers usually use ReLU?
Answer: It introduces non-linearity and trains efficiently with reduced vanishing-gradient issues.

Q2: What does backpropagation compute?
Answer: Gradients of loss with respect to model parameters.

Q3: Why split train/validation/test?
Answer: To train, tune, and evaluate generalization honestly.

Q4: Which metric matters more for imbalanced churn data?
Answer: Precision/recall (and F1), not raw accuracy.

Q5: One sign of overfitting?
Answer: Training loss drops while validation loss rises.`
            }
        ],
        'deep-learning-grid#Module 2': [
            {
                title: 'Deep-Dive Example: Regularization Strategy Playbook',
                content: `Given a model that overfits after 5 epochs, create a stabilization plan:
• add dropout (0.2-0.4)
• apply weight decay
• introduce early stopping
• increase data via augmentation where applicable

Decision rule:
change one major factor at a time and compare validation metrics.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Train baseline model and report train/validation loss.

Medium:
Add dropout + weight decay and compare metrics in a table.

Hard:
Implement early stopping and explain why the selected epoch is optimal.

Review rubric:
• clear experiment setup
• fair comparison
• justified conclusions.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is underfitting?
Answer: The model is too simple and fails on both train and validation.

Q2: Purpose of weight decay?
Answer: Penalize large weights to improve generalization.

Q3: Why is validation set necessary?
Answer: Hyperparameter tuning without leaking into final test evaluation.

Q4: Is higher model complexity always better?
Answer: No, it can overfit and hurt unseen-data performance.

Q5: What does early stopping prevent?
Answer: Over-training after validation performance plateaus or worsens.`
            }
        ],
        'generative-ai-grid#Module 2': [
            {
                title: 'Deep-Dive Example: Reliable LLM API Design',
                content: `Design a robust endpoint for "summarize support ticket":
• strict system prompt
• temperature 0.2 for deterministic output
• max token limits
• JSON schema response
• timeout/retry + fallback model

Production objective:
stable, parseable output under varying ticket quality.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Create a prompt that returns bullet summary in <= 5 lines.

Medium:
Return strict JSON with keys: summary, priority, action_items.

Hard:
Add validation and retry when model returns malformed JSON.

Success criteria:
• consistent format
• error handling
• measurable output quality.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What does temperature control?
Answer: Output randomness/creativity.

Q2: Why prefer structured output in apps?
Answer: Reliable parsing and downstream automation.

Q3: What is the context window?
Answer: Maximum token budget for input + output.

Q4: Should model output be trusted blindly?
Answer: No, validate as untrusted input.

Q5: Why keep a fallback model?
Answer: Improve reliability during provider/model failures.`
            }
        ],
        'generative-ai-grid#Module 5': [
            {
                title: 'Deep-Dive Example: RAG for Company Knowledge Base',
                content: `Build a RAG assistant for internal policy docs.

Design choices:
• chunk size and overlap strategy
• embedding model selection
• vector + keyword hybrid retrieval
• top-k tuning with grounding instructions

Quality goal:
answers must cite relevant source passages.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Index 10 documents and retrieve top-3 context chunks.

Medium:
Implement grounding prompt that references retrieved context only.

Hard:
Add retrieval evaluation set and tune chunking + top-k for better groundedness.

Deliverables:
• retrieval quality notes
• grounded response examples
• failure case analysis.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Main benefit of RAG over fine-tuning for changing knowledge?
Answer: Faster updates without retraining model weights.

Q2: Why use hybrid search?
Answer: Combines semantic and lexical relevance.

Q3: What reduces hallucinations in RAG?
Answer: Strong retrieval quality + strict grounding prompt.

Q4: What is chunk overlap for?
Answer: Preserve context continuity across splits.

Q5: Should you evaluate retrieval separately from generation?
Answer: Yes, both layers affect final answer quality.`
            }
        ],
        'dsa-foundations-grid#DSA · Module 1': [
            {
                title: 'Deep-Dive Example: Converting Brute Force to Optimal',
                content: `Problem theme: find two numbers matching a target condition.

Path:
• brute force O(n^2)
• identify repeated lookup
• switch to hash map O(n)

Interview value:
showing optimization thought process is often scored higher than speed alone.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
For three simple loops, derive time complexity.

Medium:
Refactor one O(n^2) solution to O(n) and explain why.

Hard:
Given two solutions with same time complexity, compare space and practical trade-offs.

Evaluation:
• correct complexity
• clear trade-off narrative
• edge-case awareness.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why start with brute force?
Answer: It ensures correctness and reveals optimization bottlenecks.

Q2: O(n log n) is usually better than O(n^2) for large n?
Answer: Yes, due to slower growth rate.

Q3: What is space complexity?
Answer: Extra memory used by algorithm relative to input size.

Q4: Why test edge cases early?
Answer: Prevent hidden bugs before optimization.

Q5: Does fastest code always win interviews?
Answer: No, structured reasoning and correctness matter equally.`
            }
        ],
        'dsa-foundations-grid#DSA · Module 2': [
            {
                title: 'Deep-Dive Example: Data Structure Selection Matrix',
                content: `Given requirements (fast lookup, ordered access, top-k, dedupe), map to structures:
• lookup/dedupe -> hash set/map
• ordered retrieval -> heap/tree
• sequential processing -> array/list

Create your own selection matrix and reuse it during interviews.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Pick data structure for 8 scenarios and justify in one line each.

Medium:
Solve a frequency-count problem using both array and hash map, compare complexity.

Hard:
Implement top-k frequent elements with heap and explain why it scales.

Rubric:
• right structure choice
• complexity correctness
• concise justification.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Best structure for O(1) average lookup?
Answer: Hash map/set.

Q2: LIFO behavior belongs to?
Answer: Stack.

Q3: FIFO behavior belongs to?
Answer: Queue.

Q4: Best for dynamic top-k stream?
Answer: Heap.

Q5: Why avoid one-structure-for-all problems?
Answer: Operation patterns differ; wrong choice harms performance.`
            }
        ],
        'system-design-foundations-grid#System Design · Module 1': [
            {
                title: 'Deep-Dive Example: Requirement-to-Architecture Flow',
                content: `Use this design flow for any system:
1. functional requirements
2. non-functional targets (latency, availability, consistency)
3. rough capacity estimate
4. high-level components
5. risk and trade-off review

This structure keeps designs practical and interview-ready.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Write functional and non-functional requirements for a note-taking app.

Medium:
Estimate peak RPS and daily storage for that app.

Hard:
Propose high-level architecture and defend two major trade-offs.

Expected output:
clear assumptions + measurable targets + justifiable design choices.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What are non-functional requirements?
Answer: Quality attributes like latency, availability, scalability, security.

Q2: Why do rough estimates matter?
Answer: They drive architecture and capacity choices.

Q3: Can we optimize latency, consistency, and cost all at once?
Answer: Usually no; trade-offs are required.

Q4: Why define assumptions explicitly?
Answer: They bound scope and make decisions reviewable.

Q5: What is a common beginner mistake?
Answer: Jumping to components before requirement clarity.`
            }
        ],
        'system-design-foundations-grid#System Design · Module 2': [
            {
                title: 'Deep-Dive Example: Choosing the Right Building Block',
                content: `Decision examples:
• SQL for strong relational constraints
• NoSQL for high write scale and flexible schema
• Cache for read-heavy hot paths
• Queue for async decoupling
• CDN for global static/delivery acceleration

Strong designs combine these based on workload characteristics.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Map five product scenarios to SQL/NoSQL/cache/queue choices.

Medium:
Design cache strategy for product catalog with stale-data constraints.

Hard:
Design async notification pipeline with queue retries and dead-letter handling.

Assessment:
• component fit
• failure handling
• operational clarity.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why use cache-aside?
Answer: Reduce DB load while keeping application control over cache fill.

Q2: When is queue preferred over sync call?
Answer: For slow/retry-prone workflows and traffic smoothing.

Q3: SQL vs NoSQL choice starts from?
Answer: Access patterns and consistency requirements.

Q4: Why use CDN?
Answer: Lower latency and origin offload through edge caching.

Q5: Common cache pitfall?
Answer: Poor invalidation strategy causing stale or inconsistent reads.`
            }
        ],
        'dsa-interview-grid#DSA · Module 5': [
            {
                title: 'Deep-Dive Example: DP State Design Clinic',
                content: `Use this DP template on any problem:
1. define state precisely
2. define transition relation
3. define base case
4. choose memoization/tabulation
5. optimize space if possible

Case: coin change / climbing stairs / knapsack style variants are solved by this same framework.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Write state and transition for Fibonacci and climbing stairs.

Medium:
Solve house robber using top-down memoization and bottom-up tabulation.

Hard:
Optimize 2D DP to 1D where valid and justify correctness.

Scoring:
• clean state definition
• correct transitions
• complexity explanation.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: When should DP be considered?
Answer: Overlapping subproblems + optimal substructure.

Q2: Top-down vs bottom-up difference?
Answer: Recursion with memo vs iterative table fill.

Q3: Why is state definition critical?
Answer: It determines correctness and transition logic.

Q4: Can all DP be space optimized?
Answer: Not always; only when transitions depend on limited prior states.

Q5: Common DP interview mistake?
Answer: Memorizing formulas without deriving state/transition.`
            }
        ],
        'dsa-interview-grid#DSA · Module 6': [
            {
                title: 'Deep-Dive Example: Graph + Greedy Decision Tree',
                content: `Selection rules:
• dependency ordering -> topological sort
• shortest path weighted -> Dijkstra
• component connectivity -> Union-Find
• local-choice optimization -> greedy (with proof)

Goal: pick the right algorithm quickly under interview pressure.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Run BFS/DFS manually on sample graph and list visit order.

Medium:
Solve course schedule with topological sort and cycle detection.

Hard:
Implement Kruskal MST with Union-Find and discuss complexity.

Evaluation focus:
• algorithm fit
• correctness proof intuition
• implementation stability.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Topological sort requires what graph type?
Answer: Directed acyclic graph (DAG).

Q2: Why use Union-Find in Kruskal?
Answer: Efficient cycle detection during edge selection.

Q3: BFS gives shortest path in which graph type?
Answer: Unweighted graphs.

Q4: What is greedy risk?
Answer: Local optimum may not imply global optimum without proof.

Q5: Core Dijkstra assumption?
Answer: Non-negative edge weights.`
            }
        ],
        'dsa-interview-grid#DSA · Module 7': [
            {
                title: 'Deep-Dive Example: 35-Minute Interview Execution Plan',
                content: `Suggested time split:
• 4 min clarify and restate
• 6 min brute force + constraints
• 8 min optimized approach and edge cases
• 12 min coding with narration
• 5 min testing + complexity summary

This structure improves both clarity and interviewer confidence.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Record yourself explaining one easy problem in 2 minutes.

Medium:
Do full dry run before coding and catch one edge case bug.

Hard:
Mock interview with interruption handling and trade-off discussion.

Review criteria:
• communication flow
• composure under correction
• clarity in final complexity summary.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why restate the problem first?
Answer: Align understanding and reduce wrong assumptions.

Q2: Should you hide brute force approach?
Answer: No, it shows baseline thinking before optimization.

Q3: Why think aloud while coding?
Answer: Exposes reasoning and helps interviewer follow trade-offs.

Q4: What closes an answer strongly?
Answer: Correctness check + edge cases + complexity summary.

Q5: Is perfect code enough without communication?
Answer: Usually no; communication is a scoring dimension.`
            }
        ],
        'system-design-advanced-grid#System Design · Module 5': [
            {
                title: 'Deep-Dive Example: URL Shortener at 100M Redirects/Day',
                content: `Design upgrades at scale:
• key generation strategy (base62, collision handling)
• hot-link cache with TTL
• redirect latency optimization
• analytics ingestion via async pipeline
• abuse detection and throttling

Target:
sub-50ms redirect for cache hits.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Design API contract for create and resolve short URL.

Medium:
Estimate storage and QPS for 10M links/day.

Hard:
Design anti-abuse and click-analytics architecture.

Evaluation:
• realistic assumptions
• low-latency design
• operational safeguards.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why is cache critical in URL shortener?
Answer: Redirect path is read-heavy and latency-sensitive.

Q2: Key collision handling can be ignored?
Answer: No, must be handled deterministically.

Q3: Why async analytics?
Answer: Keep redirect hot path fast.

Q4: Common abuse vector?
Answer: Spam/phishing links and brute-force key probing.

Q5: What metric matters most for user experience?
Answer: Redirect latency and failure rate.`
            }
        ],
        'system-design-advanced-grid#System Design · Module 6': [
            {
                title: 'Deep-Dive Example: Chat Reliability Model',
                content: `Build for reliability:
• persistent connection gateway
• message broker backbone
• durable message store
• delivery acknowledgment + retry
• per-user/device cursor tracking

Design explicitly for offline users and reconnection sync.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Sketch data model for users, conversations, and messages.

Medium:
Design message flow for online and offline recipients.

Hard:
Handle ordering guarantees in multi-device environment.

Scoring:
• reliability model clarity
• ordering assumptions
• failure recovery completeness.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why use message broker in chat?
Answer: Decouple producers/consumers and absorb bursts.

Q2: Exactly-once delivery is always required?
Answer: Not always; many systems use at-least-once + dedupe.

Q3: Why store per-device cursor?
Answer: Correct sync/read state across devices.

Q4: Core real-time channel tech?
Answer: WebSockets (commonly).

Q5: Offline support needs what minimum component?
Answer: Durable message persistence.`
            }
        ],
        'system-design-advanced-grid#System Design · Module 7': [
            {
                title: 'Deep-Dive Example: Feed Service Trade-offs',
                content: `Design decisions:
• fan-out on write for low read latency
• fan-out on read for lower write amplification
• hybrid for celebrity/high-follower accounts
• ranking service decoupled from write path

Make consistency and freshness assumptions explicit.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Draw feed read path components and cache layers.

Medium:
Compare fan-out on write vs read under traffic spikes.

Hard:
Design hybrid strategy with celebrity fallback path.

Expected output:
• clear write/read trade-off discussion
• cache invalidation strategy
• ranking latency budget.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why is hybrid fan-out common?
Answer: Balances write amplification and read latency.

Q2: Feed ranking should run in request hot path always?
Answer: Prefer partial precompute + lightweight online ranking.

Q3: Key cache risk in feeds?
Answer: Stale timelines due to invalidation lag.

Q4: Why separate write and read stores sometimes?
Answer: Different access patterns and scaling requirements.

Q5: Main product metric for feed quality?
Answer: Engagement relevance with acceptable latency.`
            }
        ],
        'system-design-advanced-grid#System Design · Module 8': [
            {
                title: 'Deep-Dive Example: Video Pipeline End-to-End',
                content: `Pipeline blueprint:
• chunked upload to object storage
• async transcoding workers
• ABR manifest generation
• CDN edge delivery
• playback telemetry stream

Business goal:
high completion rate with low buffering.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Define upload and playback API contracts.

Medium:
Design transcoding queue strategy with retries.

Hard:
Optimize global delivery for startup latency and cost.

Review criteria:
• throughput and storage assumptions
• failure handling
• QoE instrumentation quality.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why adaptive bitrate streaming?
Answer: Adjust quality to network conditions and reduce buffering.

Q2: Why decouple upload from transcoding?
Answer: Async processing improves reliability and scalability.

Q3: What does CDN primarily improve?
Answer: Latency and origin offload.

Q4: One key playback QoE metric?
Answer: Rebuffer ratio.

Q5: Why store multiple encoded variants?
Answer: Serve diverse bandwidth/device capabilities.`
            }
        ],
        'agent-frameworks-grid#Module 2': [
            {
                title: 'Deep-Dive Example: LangGraph State Machine Design',
                content: `Model agent workflow as explicit graph nodes:
• planner node
• tool execution node
• reflection/validation node
• final response node

Benefits:
• deterministic control flow
• easier retries and human-in-the-loop insertion
• better observability than opaque loops.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Create a two-node graph (retrieve -> answer).

Medium:
Add conditional edge for tool retry or fallback.

Hard:
Add checkpoint persistence and human approval branch.

Evaluation:
• state schema clarity
• edge-condition correctness
• failure-handling design.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Why use graph-based orchestration?
Answer: Explicit, testable control flow for multi-step agent logic.

Q2: What does checkpointing enable?
Answer: Resume, auditability, and safer long-running workflows.

Q3: When add human-in-the-loop node?
Answer: High-risk actions or low-confidence outcomes.

Q4: LangGraph best suited for?
Answer: Stateful, multi-step agent systems.

Q5: Common orchestration mistake?
Answer: Missing failure branches and fallback edges.`
            }
        ],
        'os-processes-memory-grid#OS · Module 5': [
            {
                title: 'Deep-Dive Example: Choosing a Scheduler for a Workload',
                content: `Scenario: pick a scheduling approach for three very different systems.

• Interactive phone UI -> Round Robin (responsiveness first)
• Batch data-processing server -> SJF/throughput focus
• Mixed real-time + background -> priority + multilevel queues

Key lesson:
there is no single best scheduler - it depends on which metric (responsiveness, throughput, fairness) matters most for that system.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Given arrival and burst times for 3 processes, compute average waiting time under FCFS.

Medium:
Compare Round Robin (time quantum = 2) vs SJF for the same processes and explain the difference.

Hard:
Design a scheduling policy for a system with both latency-sensitive and background jobs, and justify how you prevent starvation.

Evaluation:
• correct metric calculation
• clear trade-off reasoning
• starvation awareness.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Which algorithm is best for responsiveness?
Answer: Round Robin (small time slices).

Q2: Which minimizes average waiting time in theory?
Answer: Shortest Job First (SJF).

Q3: What problem can pure priority scheduling cause?
Answer: Starvation of low-priority processes.

Q4: What makes an OS preemptive?
Answer: The ability to forcibly pause a running process (via timer interrupt).

Q5: Why do context switches have a cost?
Answer: Saving/restoring state consumes CPU time (overhead).`
            }
        ],
        'os-processes-memory-grid#OS · Module 6': [
            {
                title: 'Deep-Dive Example: Tracing a Memory Access',
                content: `Follow what happens when a program reads a variable.

1. Program uses a virtual address.
2. The MMU checks the TLB (fast cache) for a mapping.
3. On a miss, it walks the page table to find the physical frame.
4. If the page is not in RAM -> page fault -> load from disk.
5. The physical address is accessed and value returned.

Insight:
virtual memory gives isolation and simplicity, but page faults and thrashing are the performance risks to watch.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Explain in your own words why two processes can both use address 0x1000 safely.

Medium:
Given a small page table, translate three virtual addresses to physical addresses.

Hard:
A system is thrashing under load. Propose three fixes and explain the trade-offs of each.

Rubric:
• correct translation logic
• clear understanding of isolation
• practical thrashing remedies.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is a page fault?
Answer: Access to a page not currently in RAM, requiring a load from disk.

Q2: What does the page table map?
Answer: Virtual pages to physical frames.

Q3: What is the TLB for?
Answer: Caching recent address translations to speed up memory access.

Q4: What is thrashing?
Answer: Excessive swapping where the system does little real work.

Q5: One key benefit of virtual memory?
Answer: Process isolation (and the illusion of large contiguous memory).`
            }
        ],
        'os-processes-memory-grid#OS · Module 7': [
            {
                title: 'Deep-Dive Example: Fixing a Race Condition',
                content: `Scenario: two threads increment a shared counter and updates are lost.

Root cause:
read-modify-write is not atomic, so interleaving causes lost updates.

Fix:
protect the critical section with a mutex so only one thread updates at a time.

Caution:
locking too broadly kills performance; locking inconsistently can cause deadlock.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Identify the critical section in a shared-counter example.

Medium:
Rewrite the counter update using a mutex and explain why it is now safe.

Hard:
Given two threads acquiring two locks in different orders, show how deadlock occurs and fix it.

Assessment:
• correct critical-section identification
• proper lock usage
• deadlock prevention strategy.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is a race condition?
Answer: A bug where the outcome depends on unpredictable thread timing on shared data.

Q2: What does a mutex guarantee?
Answer: Only one thread is in the critical section at a time.

Q3: Difference between binary and counting semaphore?
Answer: Binary allows 1 holder; counting allows N concurrent holders.

Q4: Name one of the four deadlock conditions.
Answer: Circular wait (also: mutual exclusion, hold and wait, no preemption).

Q5: A simple way to prevent deadlock?
Answer: Always acquire locks in a consistent global order.`
            }
        ],
        'networking-protocols-grid#Networking · Module 3': [
            {
                title: 'Deep-Dive Example: Picking TCP or UDP',
                content: `Decide the transport protocol for real features.

• File download / web page -> TCP (every byte must arrive, in order)
• Live video call -> UDP (low latency; a dropped frame beats a long delay)
• Online multiplayer game -> UDP (fast updates, app handles loss)
• Bank transaction API -> TCP (correctness is non-negotiable)

Principle:
reliability vs latency is the core trade-off.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
List two applications for TCP and two for UDP with one-line reasons.

Medium:
Explain the TCP 3-way handshake and why it exists.

Hard:
Design how a video-call app could use UDP but still recover gracefully from packet loss.

Evaluation:
• correct protocol choice
• handshake understanding
• practical loss-handling design.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Which protocol guarantees ordered, reliable delivery?
Answer: TCP.

Q2: Which protocol has no handshake and lower latency?
Answer: UDP.

Q3: What are the 3 steps of the TCP handshake?
Answer: SYN, SYN-ACK, ACK.

Q4: Why is DNS often over UDP?
Answer: Small, fast queries where speed matters more than guaranteed delivery.

Q5: Best choice for a live voice call?
Answer: UDP.`
            }
        ],
        'networking-protocols-grid#Networking · Module 4': [
            {
                title: 'Deep-Dive Example: Debugging a Slow/Broken Domain',
                content: `Scenario: a newly launched site is unreachable for some users.

Checklist:
1. Does the domain have a correct A/AAAA record?
2. Is the TTL causing stale cached answers after a recent change?
3. Are the authoritative name servers correct?
4. Is it a DNS issue or a server issue (isolate with direct IP access)?

Insight:
many "website down" issues are actually DNS propagation or misconfiguration problems.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Match record types (A, CNAME, MX, TXT) to their purpose.

Medium:
Explain the full resolution path from browser to authoritative server.

Hard:
You changed your server IP but users still hit the old one. Explain why and how TTL planning avoids this.

Rubric:
• correct record knowledge
• accurate resolution flow
• TTL/propagation understanding.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What does DNS translate?
Answer: Domain names into IP addresses.

Q2: Which record maps a name to an IPv4 address?
Answer: A record.

Q3: What does TTL control?
Answer: How long a DNS record can be cached.

Q4: Why can DNS changes take time to appear?
Answer: Cached records persist until TTL expires (propagation).

Q5: Which record type defines mail servers?
Answer: MX record.`
            }
        ],
        'networking-protocols-grid#Networking · Module 5': [
            {
                title: 'Deep-Dive Example: The Full Journey of a Page Load',
                content: `Trace visiting https://example.com end to end.

1. DNS resolves example.com to an IP.
2. TCP connection is established (3-way handshake).
3. TLS handshake sets up encryption (HTTPS).
4. Browser sends an HTTP GET request.
5. Server process (managed by its OS) builds a response.
6. Response returns with a status code (e.g., 200).
7. Browser renders HTML, CSS, and JS.

This single action combines DNS, TCP/IP, TLS, HTTP, and OS process handling.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Match status codes (200, 301, 404, 500) to their meaning.

Medium:
Explain the difference between HTTP and HTTPS and what TLS adds.

Hard:
Given a page that loads slowly, list where in the request lifecycle the delay could occur and how to diagnose each.

Assessment:
• correct status-code knowledge
• clear TLS explanation
• structured performance reasoning.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What transport protocol does HTTP usually run on?
Answer: TCP.

Q2: What does HTTPS add over HTTP?
Answer: TLS encryption, integrity, and server authentication.

Q3: What does a 404 status mean?
Answer: Resource not found (client-side error).

Q4: What does a 5xx status indicate?
Answer: A server-side error.

Q5: What is the first network step when visiting a domain?
Answer: A DNS lookup to resolve the IP address.`
            }
        ],
        'os-fundamentals-grid#OS · Module 1': [
            {
                title: 'Deep-Dive Example: What Happens When You Open an App',
                content: `Trace clicking an application icon.

1. The OS locates the program file on disk.
2. It loads the program into memory and creates a process.
3. The scheduler gives the process CPU time.
4. The app requests services (files, network, screen) via system calls.
5. The kernel safely performs those privileged operations.

Insight:
the OS is the trusted middle layer - your app never touches hardware directly.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
List three things the OS manages and give one everyday example of each.

Medium:
Explain the difference between kernel space and user space with a real scenario.

Hard:
Pick a common action (saving a file) and describe which system calls and OS responsibilities are involved.

Evaluation:
• clear grasp of OS role
• correct kernel/user distinction
• accurate mapping of actions to OS services.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is the main job of an operating system?
Answer: Manage hardware resources and act as a safe layer between apps and hardware.

Q2: What runs in kernel space?
Answer: The trusted OS core (scheduler, memory manager, drivers) with full hardware access.

Q3: What is a system call?
Answer: A request from a user program for a privileged OS service.

Q4: Why separate kernel and user space?
Answer: Protection - a buggy app cannot directly crash or corrupt the system.

Q5: Name two OS responsibilities.
Answer: Process management and memory management (also files, devices, security).`
            }
        ],
        'os-fundamentals-grid#OS · Module 2': [
            {
                title: 'Deep-Dive Example: One Program, Many Processes',
                content: `Open the same browser twice - you now have two independent processes from one program.

Why they do not interfere:
• each process gets its own memory space
• each has its own state saved in a PCB
• the OS schedules them separately

Insight:
"program" is the passive file; "process" is the active, running instance.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
In your own words, explain program vs process using a real example.

Medium:
Describe the fetch-decode-execute cycle and why it repeats.

Hard:
Explain how timer interrupts let a single CPU appear to run many programs at once.

Rubric:
• correct program/process distinction
• accurate CPU cycle description
• clear link between interrupts and multitasking.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: Difference between a program and a process?
Answer: A program is a passive file; a process is a running instance with its own state.

Q2: What are the three steps of the CPU cycle?
Answer: Fetch, decode, execute.

Q3: What is an interrupt?
Answer: A signal that makes the CPU pause and handle an urgent event.

Q4: How do interrupts help multitasking?
Answer: Timer interrupts let the OS switch between processes.

Q5: Can one program create multiple processes?
Answer: Yes, running it multiple times creates independent processes.`
            }
        ],
        'os-fundamentals-grid#OS · Module 3': [
            {
                title: 'Deep-Dive Example: Following a File From Name to Disk',
                content: `You open /home/user/report.txt.

1. The OS resolves the path through the directory tree.
2. Metadata (inode) tells it which disk blocks hold the file.
3. The OS reads those blocks into RAM.
4. Permissions are checked to confirm you are allowed access.

Insight:
the file system is a clean human-friendly view over messy physical storage.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Explain the difference between an absolute path and a relative path with examples.

Medium:
Describe how file permissions (read/write/execute) protect a shared system.

Hard:
Explain the difference between storage and memory and why the OS constantly moves data between them.

Assessment:
• correct path understanding
• clear permission model
• accurate disk-vs-RAM reasoning.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is a directory?
Answer: A structure that organizes files into a hierarchy (folder).

Q2: What does an inode-style metadata record track?
Answer: Where a file's data blocks live on disk (plus attributes).

Q3: What are the three basic Unix permission types?
Answer: Read, write, execute.

Q4: Key difference between RAM and disk?
Answer: RAM is fast and temporary; disk is slower and permanent.

Q5: Why do we need file permissions?
Answer: To control access and protect data between users/programs.`
            }
        ],
        'os-processes-memory-grid#OS · Module 4': [
            {
                title: 'Deep-Dive Example: Why Your App Stays Responsive',
                content: `A photo app downloads images while the UI still scrolls smoothly.

How:
• the UI runs on one thread
• the download runs on another thread
• both share the process memory but run "concurrently"

Insight:
threads let one application do several things at once without freezing.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
List the process states and what triggers moving between Ready and Running.

Medium:
Explain the difference between a thread and a process, with one benefit and one risk of threads.

Hard:
Describe what happens during a context switch and why too many switches hurt performance.

Evaluation:
• correct state model
• clear thread vs process contrast
• accurate context-switch explanation.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What does a Process Control Block store?
Answer: Process ID, state, saved registers, memory info, and open files.

Q2: How do threads differ from processes?
Answer: Threads share the process's memory and are lighter to create/switch.

Q3: What is a context switch?
Answer: Saving one process/thread's state and loading another's to run it.

Q4: Why is IPC needed?
Answer: Processes have separate memory and need explicit ways to communicate.

Q5: Name one IPC method.
Answer: Pipes (also message queues, shared memory, sockets).`
            }
        ],
        'networking-fundamentals-grid#Networking · Module 1': [
            {
                title: 'Deep-Dive Example: A Message Through the Layers',
                content: `Sending a chat message travels down the layers and back up.

Sending side:
• Application: your text
• Transport: adds port + reliability info
• Internet: adds source/destination IP
• Link: puts bits on the wire

Receiving side unwraps each layer in reverse.

Insight:
layering lets each part (Wi-Fi, IP, HTTP) evolve independently.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
List the four TCP/IP layers and one protocol at each.

Medium:
Explain encapsulation using the "nested envelopes" analogy.

Hard:
Explain the client-server model and how IP + port together deliver data to the right app.

Rubric:
• correct layer knowledge
• clear encapsulation explanation
• accurate IP-vs-port distinction.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What is a network?
Answer: Two or more devices connected to exchange data.

Q2: Name the four TCP/IP layers.
Answer: Application, Transport, Internet, Link.

Q3: What is encapsulation?
Answer: Each layer wrapping data with its own header as it goes down the stack.

Q4: What does a port number identify?
Answer: Which service/app on a machine.

Q5: What does the IP identify?
Answer: Which machine on the network.`
            }
        ],
        'networking-fundamentals-grid#Networking · Module 2': [
            {
                title: 'Deep-Dive Example: How Your Devices Share One Public IP',
                content: `At home, your phone, laptop, and TV all browse the internet using one public IP.

How:
• each device gets a private IP from the router
• the router uses NAT to map them to its single public IP
• responses are routed back to the right device

Insight:
private IPs organize the local network; NAT bridges them to the public internet.`
            },
            {
                title: 'Graded Exercises (Easy -> Medium -> Hard)',
                content: `Easy:
Give an example of a private IP and a public IP and state the difference.

Medium:
Explain in simple terms what routing does across the internet.

Hard:
Explain how NAT lets many devices share one public IP and why IPv6 reduces the need for it.

Assessment:
• correct public/private distinction
• clear routing explanation
• accurate NAT understanding.`
            },
            {
                title: 'Mini Quiz (with Answers)',
                content: `Q1: What does an IP address identify?
Answer: A device on a network.

Q2: Difference between IPv4 and IPv6?
Answer: IPv6 has a far larger address space than IPv4.

Q3: Are private IPs directly reachable from the internet?
Answer: No, they are used inside local networks.

Q4: What does routing do?
Answer: Moves packets across routers toward the destination network.

Q5: What does NAT enable?
Answer: Many private devices to share one public IP.`
            }
        ]
    };

    return expansions[key] || [];
}

function moduleLessonsLabel(module, gridId) {
    if (Array.isArray(module.detailedContent) && module.detailedContent.length) {
        const n = module.detailedContent.length + getAutoDeepDiveSections(module, gridId).length + getPhaseTwoExpansionSections(module, gridId).length;
        return n + (n === 1 ? ' lesson' : ' lessons');
    }
    return module.lessons;
}

function getModuleDifficulty(gridId, moduleNumberLabel) {
    if (!/-programming-grid$/.test(gridId)) {
        return '';
    }
    const number = parseInt(String(moduleNumberLabel || '').replace(/[^0-9]/g, ''), 10);
    if (!number || Number.isNaN(number)) {
        return '';
    }
    if (number <= 2) {
        return 'Beginner';
    }
    if (number <= 4) {
        return 'Intermediate';
    }
    return 'Advanced';
}

// Create a module card element
function createModuleCard(module, gridId, moduleId = `${gridId}#${module.number}`) {
    const card = document.createElement('article');
    const read = isModuleRead(moduleId);
    card.className = `module-card ${module.isNew ? 'new' : ''} ${module.isLocked ? 'locked' : ''} ${read ? 'read' : ''}`;
    card.dataset.moduleId = moduleId;
    card.dataset.moduleNumber = module.number;

    const lockIcon = module.isLocked ? '<div class="lock-icon">🔒</div>' : '';
    const hasCode = Array.isArray(module.detailedContent) && module.detailedContent.some(s => s && s.code);
    const durationMeta = module.duration
        ? `<span class="meta-item"><i class="fas fa-clock" aria-hidden="true"></i> Est. study time: ${module.duration}</span>`
        : '';
    const codeMeta = hasCode
        ? '<span class="meta-item meta-code"><i class="fas fa-code"></i> Code examples</span>'
        : '';
    const difficulty = getModuleDifficulty(gridId, module.number);
    const difficultyClass = difficulty ? ' module-difficulty-' + difficulty.toLowerCase() : '';
    const difficultyTag = difficulty
        ? `<span class="module-difficulty${difficultyClass}">${difficulty}</span>`
        : '';
    const projectMeta = /project|capstone/i.test(String(module.title || ''))
        ? '<span class="meta-item meta-project"><i class="fas fa-diagram-project"></i> Mini project</span>'
        : '';

    card.innerHTML = `
        ${lockIcon}
        <span class="read-check" aria-hidden="true"><i class="fas fa-circle-check"></i></span>
        <div class="module-number">${module.number}</div>
        ${difficultyTag}
        <h4 class="module-title"><button type="button" class="module-open">${module.title}<span class="sr-only">, ${module.number}${read ? ', completed' : ''}</span></button></h4>
        <p class="module-description">${module.description}</p>
        <div class="module-meta">
            <span class="meta-item"><i class="fas fa-book-open"></i> ${moduleLessonsLabel(module, gridId)}</span>
            ${durationMeta}
            ${codeMeta}
            ${projectMeta}
        </div>
        <span class="module-status">${read ? 'Completed' : 'Open module'}</span>
    `;

    card.querySelector('.module-open').addEventListener('click', () => openModuleModal(module, moduleId));
    enhanceExternalLinks(card);
    return card;
}

function extractMermaidDefinition(diagramText) {
    if (!diagramText) return '';

    const raw = String(diagramText).trim();
    if (!raw) return '';

    const fenced = raw.match(/```mermaid\s*([\s\S]*?)```/i);
    if (fenced && fenced[1]) {
        return fenced[1].trim();
    }

    const startsLikeMermaid = /^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|mindmap|timeline|gitGraph|pie)\b/i.test(raw);
    if (startsLikeMermaid) {
        const lines = raw
            .split('\n')
            .map(function (line) { return line.trimEnd(); });

        const stopPattern = /^(Practice|Worked Example|Common Mistakes|Best Practices|Recap|Mini Exercise|Key Takeaways|Expected Output|Code Example)\s*:/i;

        const diagramLines = [];
        for (let i = 0; i < lines.length; i += 1) {
            const line = lines[i].trim();
            if (!line) {
                diagramLines.push('');
                continue;
            }
            if (i > 0 && stopPattern.test(line)) {
                break;
            }
            diagramLines.push(lines[i]);
        }

        let cleaned = diagramLines.join('\n').trim();
        // Normalize node labels so Mermaid doesn't choke on punctuation-heavy labels.
        cleaned = cleaned.replace(/\b([A-Za-z][A-Za-z0-9_]*)\[([^\]]+)\]/g, function (_, nodeId, label) {
            const trimmed = String(label).trim();
            const alreadyQuoted = trimmed.startsWith('"') && trimmed.endsWith('"');
            if (alreadyQuoted) {
                return nodeId + '[' + trimmed + ']';
            }

            const safe = trimmed.replace(/"/g, '\\"');
            return nodeId + '["' + safe + '"]';
        });

        return cleaned;
    }

    return '';
}

function renderVisualDiagram(diagramText) {
    const mermaidDefinition = extractMermaidDefinition(diagramText);
    if (mermaidDefinition) {
        const encoded = encodeURIComponent(mermaidDefinition);
        return `
            <div class="learning-diagram learning-diagram-mermaid" data-mermaid-source="${encoded}">
                <span class="content-label">Visual:</span>
                <p class="diagram-status" role="status"></p>
                <div class="mermaid-host" role="region" aria-label="Scrollable diagram definition" tabindex="0">
                    <pre class="mermaid-fallback"><code>${escapeHtml(mermaidDefinition)}</code></pre>
                </div>
                <details class="diagram-description" hidden><summary>Read diagram definition</summary><pre class="mermaid-fallback">${escapeHtml(mermaidDefinition)}</pre></details>
            </div>
        `;
    }

    return `<div class="learning-diagram"><span class="content-label">Visual:</span> ${escapeHtml(String(diagramText || ''))}</div>`;
}

let mermaidLoaderPromise = null;
let mermaidSequence = 0;

function ensureMermaidLoaded() {
    if (window.mermaid) {
        return Promise.resolve(window.mermaid);
    }

    if (mermaidLoaderPromise) {
        return mermaidLoaderPromise;
    }

    mermaidLoaderPromise = new Promise(function (resolve, reject) {
        const script = document.createElement('script');
        let settled = false;
        script.src = 'https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js';
        script.async = true;
        const timeout = setTimeout(() => finish(new Error('Diagram library loading timed out')), 15000);
        function finish(error) {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);
            if (error) {
                script.remove();
                mermaidLoaderPromise = null;
                reject(error);
            } else {
                resolve(window.mermaid);
            }
        }
        script.onload = function () {
            finish(window.mermaid && typeof window.mermaid.render === 'function' ? null : new Error('The diagram library did not initialize'));
        };
        script.onerror = function () { finish(new Error('Failed to load the diagram library')); };
        document.head.appendChild(script);
    });

    return mermaidLoaderPromise;
}

function renderMermaidDiagrams(container) {
    if (!container) return;

    const nodes = Array.from(container.querySelectorAll('.learning-diagram-mermaid[data-mermaid-source]:not([data-mermaid-rendered])'))
        .filter(node => !node.closest('.content-body[hidden]'));
    if (!nodes.length) return;
    nodes.forEach(node => {
        node.dataset.mermaidRendered = 'loading';
        node.querySelector('.diagram-status').textContent = 'Loading diagram...';
    });

    function showFallback(node, error) {
        console.warn('A learning diagram could not be rendered:', error);
        node.dataset.mermaidRendered = 'error';
        node.querySelector('.diagram-status').textContent = 'Visual diagram unavailable. The text definition is shown below.';
    }

    ensureMermaidLoaded().then(function (mermaid) {
        mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'strict',
            suppressErrorRendering: true,
            theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'default'
        });

        nodes.forEach(function (node) {
            if (!node.isConnected) return;
            const host = node.querySelector('.mermaid-host');
            const encoded = node.getAttribute('data-mermaid-source') || '';
            const source = decodeURIComponent(encoded);
            if (!host || !source) return;

            const renderId = 'cloudvana-diagram-' + (++mermaidSequence);
            mermaid.render(renderId, source).then(function (result) {
                if (!node.isConnected) return;
                host.innerHTML = result.svg;
                host.setAttribute('aria-label', 'Scrollable learning diagram');
                const svg = host.querySelector('svg');
                svg.setAttribute('role', 'img');
                svg.setAttribute('aria-label', 'Learning flow. A text definition is available below.');
                node.querySelector('.diagram-description').hidden = false;
                node.querySelector('.diagram-status').hidden = true;
                node.setAttribute('data-mermaid-rendered', 'true');
            }).catch(function (error) {
                if (node.isConnected) showFallback(node, error);
            });
        });
    }).catch(function (error) {
        nodes.filter(node => node.isConnected).forEach(node => showFallback(node, error));
    });
}

function formatSectionContent(rawContent, sectionTitle) {
    if (!rawContent) return '';

    const lines = String(rawContent).split('\n');
    const labeledLinePattern = /^(Learning Objective|Learning Objectives|Estimated Reading Time|Estimated Completion Time|Difficulty|Theory Content|Core Theory|Real World Analogy|Visual Diagram(?: \(Markdown\))?|Diagram \(Mermaid\)|Code Example|Expected Output|Common Mistakes|Best Practices|Mini Exercise|Worked Example|Practice|Key Takeaways|Recap|Skills Gained|Prerequisites|Module Number|Module Name|Module Description|Module Quiz|Interview Preparation|Module Summary|Final Assessment|Project Deliverables|Suggested Project Options|Mini Challenge|Course Completion Path|Next Module Bridge|Next Lesson Connection):\s*(.*)$/;

    const labelAliases = {
        'Core Theory': 'Theory Content',
        'Diagram (Mermaid)': 'Visual Diagram',
        'Worked Example': 'Mini Exercise',
        'Practice': 'Mini Exercise',
        'Recap': 'Key Takeaways'
    };

    const normalizeLabel = function (label) {
        return labelAliases[label] || label;
    };

    const fieldMap = {};
    const genericLines = [];
    let currentLabel = null;

    lines.forEach(function (line) {
        const text = line.trim();
        if (!text) {
            if (currentLabel && fieldMap[currentLabel]) {
                fieldMap[currentLabel] += '\n';
            } else {
                genericLines.push('');
            }
            return;
        }

        const labeled = text.match(labeledLinePattern);
        if (labeled) {
            currentLabel = normalizeLabel(labeled[1]);
            fieldMap[currentLabel] = fieldMap[currentLabel] ? fieldMap[currentLabel] + '\n' + labeled[2] : labeled[2];
            return;
        }

        if (currentLabel) {
            fieldMap[currentLabel] += '\n' + text;
            return;
        }

        genericLines.push(text);
    });

    function renderValue(value) {
        if (!value) return '';
        const valueLines = String(value).split('\n').map(function (v) { return v.trim(); }).filter(Boolean);
        const hasBullets = valueLines.some(function (v) { return v.startsWith('• ') || v.startsWith('- '); });
        if (hasBullets) {
            return '<ul class="content-list">' + valueLines.map(function (v) {
                if (v.startsWith('• ') || v.startsWith('- ')) return '<li>' + v.substring(2) + '</li>';
                return '<li>' + v + '</li>';
            }).join('') + '</ul>';
        }
        return '<p class="content-paragraph">' + valueLines.join('<br>') + '</p>';
    }

    const normalizedTitle = String(sectionTitle || '').trim().toLowerCase();

    const isModuleBlueprint = /module blueprint/i.test(String(sectionTitle || '')) ||
        !!(fieldMap['Module Number'] && fieldMap['Module Name'] && fieldMap['Difficulty']);

    if (isModuleBlueprint) {
        return `
            <div class="blueprint-card">
                <div class="blueprint-head">
                    <h4>Module At A Glance</h4>
                </div>
                <div class="blueprint-meta-grid">
                    ${fieldMap['Module Number'] ? `<div class="blueprint-meta-item"><span class="content-label">Module</span><span>${fieldMap['Module Number']}</span></div>` : ''}
                    ${fieldMap['Module Name'] ? `<div class="blueprint-meta-item"><span class="content-label">Name</span><span>${fieldMap['Module Name']}</span></div>` : ''}
                    ${fieldMap['Difficulty'] ? `<div class="blueprint-meta-item"><span class="content-label">Difficulty</span><span>${fieldMap['Difficulty']}</span></div>` : ''}
                    ${fieldMap['Estimated Reading Time'] ? `<div class="blueprint-meta-item"><span class="content-label">Reading Time</span><span>${fieldMap['Estimated Reading Time']}</span></div>` : ''}
                    ${fieldMap['Estimated Completion Time'] ? `<div class="blueprint-meta-item"><span class="content-label">Completion Time</span><span>${fieldMap['Estimated Completion Time']}</span></div>` : ''}
                    ${fieldMap['Prerequisites'] ? `<div class="blueprint-meta-item"><span class="content-label">Prerequisites</span><span>${fieldMap['Prerequisites']}</span></div>` : ''}
                </div>
                <div class="blueprint-panels">
                    ${fieldMap['Learning Objectives'] || fieldMap['Learning Objective']
                        ? `<div class="blueprint-panel"><h5>Learning Objectives</h5>${renderValue(fieldMap['Learning Objectives'] || fieldMap['Learning Objective'])}</div>`
                        : ''}
                    ${fieldMap['Skills Gained']
                        ? `<div class="blueprint-panel"><h5>Skills Gained</h5>${renderValue(fieldMap['Skills Gained'])}</div>`
                        : ''}
                </div>
            </div>
        `;
    }

    const isLessonLike = !!(fieldMap['Learning Objective'] || fieldMap['Learning Objectives']) &&
        !!(fieldMap['Theory Content'] || fieldMap['Mini Exercise'] || fieldMap['Key Takeaways']);

    if (isLessonLike) {
        const objective = fieldMap['Learning Objective'] || fieldMap['Learning Objectives'] || '';
        const readingTime = fieldMap['Estimated Reading Time'] || '';
        const difficulty = fieldMap['Difficulty'] || '';
        const theory = fieldMap['Theory Content'] || '';
        const analogy = fieldMap['Real World Analogy'] || '';
        const diagram = fieldMap['Visual Diagram (Markdown)'] || fieldMap['Visual Diagram'] || '';
        const output = fieldMap['Expected Output'] || '';
        const mistakes = fieldMap['Common Mistakes'] || '';
        const bestPractices = fieldMap['Best Practices'] || '';
        const miniExercise = fieldMap['Mini Exercise'] || '';
        const takeaways = fieldMap['Key Takeaways'] || '';
        const hasTryContent = !!(output || miniExercise);
        const hasRememberContent = !!(mistakes || bestPractices || takeaways);

        return `
            <div class="learning-flow">
                <div class="learning-meta">
                    ${readingTime ? `<span class="learning-chip"><i class="fas fa-clock"></i> ${readingTime}</span>` : ''}
                    ${difficulty ? `<span class="learning-chip"><i class="fas fa-signal"></i> ${difficulty}</span>` : ''}
                    ${sectionTitle ? `<span class="learning-chip"><i class="fas fa-graduation-cap"></i> Lesson Flow</span>` : ''}
                </div>
                <div class="learning-block learning-block-understand">
                    <h4>Understand</h4>
                    ${objective ? `<p class="content-row"><span class="content-label">Goal:</span> ${objective}</p>` : ''}
                    ${theory ? `<p class="content-row"><span class="content-label">Concept:</span> ${theory}</p>` : ''}
                    ${analogy ? `<p class="content-row"><span class="content-label">Analogy:</span> ${analogy}</p>` : ''}
                    ${diagram ? renderVisualDiagram(diagram) : ''}
                </div>
                <div class="learning-grid">
                    ${hasTryContent ? `<div class="learning-block learning-block-try">
                        <h4>Try</h4>
                        ${output ? `<p class="content-row"><span class="content-label">Expected Output:</span> ${output}</p>` : ''}
                        ${miniExercise ? `<p class="content-row"><span class="content-label">Mini Exercise:</span> ${miniExercise}</p>` : ''}
                    </div>` : ''}
                    ${hasRememberContent ? `<div class="learning-block learning-block-remember">
                        <h4>Remember</h4>
                        ${mistakes ? `<p class="content-row"><span class="content-label">Mistake Alert:</span> ${mistakes}</p>` : ''}
                        ${bestPractices ? `<p class="content-row"><span class="content-label">Best Practice:</span> ${bestPractices}</p>` : ''}
                        ${takeaways ? `<p class="content-row"><span class="content-label">Takeaway:</span> ${takeaways}</p>` : ''}
                    </div>` : ''}
                </div>
            </div>
        `;
    }

    const isMiniProject = normalizedTitle.indexOf('mini project') === 0;
    if (isMiniProject) {
        const lines = String(rawContent).split('\n').map(function (l) { return l.trim(); });
        const groups = [];
        let intro = '';
        let current = null;

        lines.forEach(function (line) {
            if (!line) return;

            if (/^[^:]+:\s*(.+)$/.test(line) && !/^[•-]\s+/.test(line)) {
                const singleLine = line.match(/^([^:]+):\s*(.+)$/);
                if (singleLine) {
                    if (!intro) {
                        intro = singleLine[2].trim();
                    } else {
                        current = { title: singleLine[1].trim(), items: [singleLine[2].trim()] };
                        groups.push(current);
                    }
                }
                return;
            }

            const heading = line.match(/^([^:]+):\s*$/);
            if (heading) {
                current = { title: heading[1].trim(), items: [] };
                groups.push(current);
                return;
            }

            if (/^[•-]\s+/.test(line)) {
                const item = line.replace(/^[•-]\s+/, '');
                if (!current) {
                    current = { title: 'Plan', items: [] };
                    groups.push(current);
                }
                current.items.push(item);
                return;
            }

            if (!intro) {
                intro = line;
            } else if (current) {
                current.items.push(line);
            }
        });

        const projectTitle = String(sectionTitle || '').replace(/^mini project:\s*/i, '').trim() || 'Mini Project';

        return `
            <div class="section-focus section-focus-project">
                <div class="section-focus-head">
                    <h4>${projectTitle}</h4>
                    <span class="section-focus-tag">Build Brief</span>
                </div>
                ${intro ? `<p class="section-focus-intro">${intro}</p>` : ''}
                <div class="mini-project-grid">
                    ${groups.map(function (group) {
                        return `
                            <article class="mini-project-card">
                                <h5>${group.title}</h5>
                                <ul>
                                    ${group.items.map(function (item) { return `<li>${item}</li>`; }).join('')}
                                </ul>
                            </article>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    const isSuggestedProjects = normalizedTitle === 'suggested project options';
    if (isSuggestedProjects) {
        const blocks = String(rawContent)
            .split(/\n\s*\n/)
            .map(function (b) { return b.trim(); })
            .filter(Boolean);

        const intro = blocks.shift() || 'Choose one project and execute it deeply.';
        const cards = blocks.map(function (block) {
            const rows = block.split('\n').map(function (r) { return r.trim(); }).filter(Boolean);
            const title = (rows[0] || '').replace(/^[-•]\s*/, '');
            const points = rows.slice(1).map(function (p) {
                return p.replace(/^[•-]\s*/, '');
            });
            return { title: title, points: points };
        }).filter(function (card) { return card.title; });

        return `
            <div class="section-focus section-focus-projects">
                <div class="section-focus-head">
                    <h4>Pick Your Capstone</h4>
                    <span class="section-focus-tag">Choose One</span>
                </div>
                <p class="section-focus-intro">${intro}</p>
                <div class="project-options-grid">
                    ${cards.map(function (card) {
                        return `
                            <article class="project-option-card">
                                <h5>${card.title}</h5>
                                <ul>
                                    ${card.points.map(function (point) { return `<li>${point}</li>`; }).join('')}
                                </ul>
                            </article>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    const isProjectDeliverables = normalizedTitle === 'project deliverables';
    if (isProjectDeliverables) {
        const lines = String(rawContent).split('\n').map(function (l) { return l.trim(); });
        const intro = (lines.find(function (l) { return l && !/^[•-]/.test(l) && !/:$/.test(l) && l.indexOf(':') === -1; }) || '').trim();
        const groups = [];
        let current = null;

        lines.forEach(function (line) {
            if (!line || line === intro) return;
            const headingMatch = line.match(/^([^:]+):\s*$/);
            if (headingMatch) {
                current = { title: headingMatch[1].trim(), items: [] };
                groups.push(current);
                return;
            }
            if (/^[•-]\s+/.test(line) && current) {
                current.items.push(line.replace(/^[•-]\s+/, ''));
                return;
            }
            if (current) {
                current.items.push(line);
            }
        });

        return `
            <div class="section-focus section-focus-deliverables">
                <div class="section-focus-head">
                    <h4>Delivery Checklist</h4>
                    <span class="section-focus-tag">Must Have</span>
                </div>
                ${intro ? `<p class="section-focus-intro">${intro}</p>` : ''}
                <div class="deliverables-grid">
                    ${groups.map(function (group) {
                        return `
                            <article class="deliverable-card">
                                <h5>${group.title}</h5>
                                <ul>
                                    ${group.items.map(function (item) { return `<li>${item}</li>`; }).join('')}
                                </ul>
                            </article>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    const isFinalAssessment = normalizedTitle === 'final assessment';
    if (isFinalAssessment) {
        const rows = String(rawContent).split('\n').map(function (r) { return r.trim(); }).filter(Boolean);
        const keyInfo = [];
        let formatItems = [];
        let passItems = [];
        let bucket = null;

        rows.forEach(function (row) {
            if (/^Assessment Format:/i.test(row)) {
                bucket = 'format';
                return;
            }
            if (/^Pass Criteria:/i.test(row)) {
                bucket = 'pass';
                return;
            }
            if (/^[•-]\s+/.test(row)) {
                if (bucket === 'format') {
                    formatItems.push(row.replace(/^[•-]\s+/, ''));
                } else if (bucket === 'pass') {
                    passItems.push(row.replace(/^[•-]\s+/, ''));
                }
                return;
            }
            bucket = null;
            keyInfo.push(row);
        });

        const purpose = keyInfo.find(function (line) { return /^Purpose:/i.test(line); }) || '';
        const why = keyInfo.find(function (line) { return /^Why it exists:/i.test(line); }) || '';

        return `
            <div class="section-focus section-focus-assessment">
                <div class="section-focus-head">
                    <h4>Final Assessment Guide</h4>
                    <span class="section-focus-tag">Readiness Gate</span>
                </div>
                ${purpose ? `<p class="content-row"><span class="content-label">Purpose:</span> ${purpose.replace(/^Purpose:\s*/i, '')}</p>` : ''}
                ${why ? `<p class="content-row"><span class="content-label">Why This Tab:</span> ${why.replace(/^Why it exists:\s*/i, '')}</p>` : ''}
                <div class="assessment-grid">
                    <article class="assessment-card">
                        <h5>What You Will Be Evaluated On</h5>
                        <ul>
                            ${formatItems.map(function (item) { return `<li>${item}</li>`; }).join('')}
                        </ul>
                    </article>
                    <article class="assessment-card">
                        <h5>How You Pass</h5>
                        <ul>
                            ${passItems.map(function (item) { return `<li>${item}</li>`; }).join('')}
                        </ul>
                    </article>
                </div>
            </div>
        `;
    }

    const isMiniChallenge = normalizedTitle === 'mini challenge';
    if (isMiniChallenge) {
        return `
            <div class="section-focus section-focus-challenge">
                <div class="section-focus-head">
                    <h4>Hands-On Challenge</h4>
                    <span class="section-focus-tag">Do This Now</span>
                </div>
                <p class="section-focus-intro">Apply the module concepts in a practical task before moving forward.</p>
                ${renderValue(rawContent)}
            </div>
        `;
    }

    const isModuleQuiz = normalizedTitle === 'module quiz' || normalizedTitle === 'mini assessment';
    if (isModuleQuiz) {
        const quizLines = String(rawContent)
            .split('\n')
            .map(function (l) { return l.trim(); })
            .filter(Boolean);

        const quizItems = quizLines.map(function (line) {
            // Remove leading numbering like "1) " to avoid duplicate numbering in ordered list.
            const cleaned = line.replace(/^\d+\)\s*/, '');
            const parts = cleaned.split(':');
            if (parts.length < 2) {
                return { question: cleaned, options: [] };
            }

            const question = parts[0].trim();
            const optionsPart = parts.slice(1).join(':').trim();
            const options = [];
            const optionRegex = /([A-D])\)\s*([^A-D]+?)(?=\s+[A-D]\)|$)/g;
            let match;
            while ((match = optionRegex.exec(optionsPart)) !== null) {
                options.push({ label: match[1], text: match[2].trim() });
            }

            if (!options.length) {
                return { question: cleaned, options: [] };
            }
            return { question: question, options: options };
        });

        return `
            <div class="section-focus section-focus-quiz">
                <div class="section-focus-head">
                    <h4>Knowledge Check</h4>
                    <span class="section-focus-tag">Self-Test</span>
                </div>
                <p class="section-focus-intro">Use this quiz to verify understanding before interviews and projects.</p>
                <ol class="quiz-list quiz-list-cards">
                    ${quizItems.map(function (item) {
                        return `
                            <li class="quiz-item">
                                <p class="quiz-question">${item.question}</p>
                                ${item.options.length
                                    ? `<div class="quiz-options">${item.options.map(function (opt) {
                                        return `<span class="quiz-option"><strong>${opt.label})</strong> ${opt.text}</span>`;
                                    }).join('')}</div>`
                                    : ''}
                            </li>
                        `;
                    }).join('')}
                </ol>
            </div>
        `;
    }

    const isInterviewPrep = normalizedTitle === 'interview preparation';
    if (isInterviewPrep) {
        return `
            <div class="section-focus section-focus-interview">
                <div class="section-focus-head">
                    <h4>Interview Readiness</h4>
                    <span class="section-focus-tag">Job Focus</span>
                </div>
                <p class="section-focus-intro">Practice these prompts out loud as if you are in a real interview round.</p>
                ${renderValue(rawContent)}
            </div>
        `;
    }

    const isModuleSummary = normalizedTitle === 'module summary';
    if (isModuleSummary) {
        return `
            <div class="section-focus section-focus-summary">
                <div class="section-focus-head">
                    <h4>What You Can Do Now</h4>
                    <span class="section-focus-tag">Checkpoint</span>
                </div>
                ${renderValue(rawContent)}
            </div>
        `;
    }

    const isNextBridge = normalizedTitle === 'next module bridge' || normalizedTitle === 'next lesson connection';
    if (isNextBridge) {
        return `
            <div class="section-focus section-focus-next">
                <div class="section-focus-head">
                    <h4>What Comes Next</h4>
                    <span class="section-focus-tag">Transition</span>
                </div>
                ${renderValue(rawContent)}
            </div>
        `;
    }

    let html = '';
    let listOpen = false;

    function closeList() {
        if (listOpen) {
            html += '</ul>';
            listOpen = false;
        }
    }

    lines.forEach(function (line) {
        const text = line.trim();

        if (!text) {
            closeList();
            html += '<div class="content-gap"></div>';
            return;
        }

        if (text.startsWith('• ') || text.startsWith('- ')) {
            if (!listOpen) {
                html += '<ul class="content-list">';
                listOpen = true;
            }
            html += `<li>${text.substring(2)}</li>`;
            return;
        }

        const labeled = text.match(labeledLinePattern);
        if (labeled) {
            closeList();
            html += `<p class="content-row"><span class="content-label">${labeled[1]}:</span> ${labeled[2]}</p>`;
            return;
        }

        closeList();
        html += `<p class="content-paragraph">${text}</p>`;
    });

    closeList();
    return html;
}

// Open module details in modal
function openModuleModal(module, moduleId) {
    const modal = document.getElementById('moduleModal');
    const modalBody = document.getElementById('modal-body');
    const read = moduleId ? isModuleRead(moduleId) : false;
    const readBtnHtml = moduleId ? `
                <button class="btn read-toggle ${read ? 'is-read' : ''}" aria-pressed="${read}" onclick="toggleModuleRead('${moduleId}', this)">
                    ${read ? '<i class="fas fa-circle-check" aria-hidden="true"></i> Completed' : '<i class="far fa-circle" aria-hidden="true"></i> Mark as complete'}
                </button>` : '';
    // Check if module is locked
    if (module.isLocked) {
        modalBody.innerHTML = `
            <div class="modal-header" style="text-align: center;">
                <div class="lock-icon-large">🔒</div>
                <h2 class="modal-title" id="moduleModalTitle">Module Locked</h2>
                <p class="modal-description">
                    This module is currently locked. Complete the previous modules to unlock this content.
                </p>
            </div>
            <div style="margin-top: 2rem; text-align: center;">
                <button class="btn btn-secondary" onclick="closeModal()">
                    Close
                </button>
            </div>
        `;
        showModuleDialog(modal);
        return;
    }
    
    // Check if module has detailed content
    if (module.detailedContent && module.detailedContent.length > 0) {
        const gridId = moduleId ? moduleId.split('#')[0] : '';
        const allSections = module.detailedContent
            .concat(getAutoDeepDiveSections(module, gridId))
            .concat(getPhaseTwoExpansionSections(module, gridId));
        // Show detailed content with expandable sections
        const contentSections = allSections.map((section, index) => `
            <div class="content-section">
                <h3 class="content-heading">
                    <button class="content-header" type="button" aria-expanded="false" aria-controls="content-${index}" onclick="toggleContent(${index})">
                    <span class="content-title">
                        <span class="content-number">${index + 1}</span>
                        ${section.title}
                    </span>
                    <span class="expand-icon" id="icon-${index}" aria-hidden="true">+</span>
                    </button>
                </h3>
                <div class="content-body" id="content-${index}" hidden>
                    <div class="content-text">${formatSectionContent(section.content, section.title)}</div>
                    ${section.code ? `
                        <div class="code-section">
                            <div class="code-header">
                                <span>Code example</span>
                                <button class="copy-btn" onclick="copyCode(${index}, event)" aria-live="polite">Copy code</button>
                            </div>
                            <pre tabindex="0" aria-label="Code example: ${escapeHtml(section.title)}"><code id="code-${index}">${escapeHtml(section.code)}</code></pre>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
        modalBody.innerHTML = `
            <div class="modal-header">
                <div class="module-number">${module.number}</div>
                <h2 class="modal-title" id="moduleModalTitle">${module.title}</h2>
                <p class="modal-description">${module.detailedDescription}</p>
                <div class="module-meta" style="justify-content: center; margin-top: 1rem;">
                    <span class="meta-item"><i class="fas fa-book-open"></i> ${moduleLessonsLabel(module, moduleId ? moduleId.split('#')[0] : '')}</span>
                </div>
            </div>
            <div class="detailed-content">
                <div class="content-toolbar">
                    <button class="btn btn-secondary btn-sm" onclick="expandAllContentSections()">Expand All</button>
                    <button class="btn btn-secondary btn-sm" onclick="collapseAllContentSections()">Collapse All</button>
                </div>
                ${contentSections}
            </div>
            <div class="modal-actions">
                ${readBtnHtml}
                <button class="btn btn-primary" onclick="closeModal()">Close</button>
            </div>
        `;
    } else {
        // Original simple view for modules without detailed content
        modalBody.innerHTML = `
            <div class="modal-header">
                <div class="module-number">${module.number}</div>
                <h2 class="modal-title" id="moduleModalTitle">${module.title}</h2>
                <p class="modal-description">${module.detailedDescription}</p>
            </div>
            <div class="modal-stats">
                <div class="module-meta">
                    <span class="meta-item"><i class="fas fa-book-open"></i> ${moduleLessonsLabel(module, moduleId ? moduleId.split('#')[0] : '')}</span>
                </div>
            </div>
            <div class="topics-section">
                <h3>What You'll Learn</h3>
                <ul class="topics-list">
                    ${module.topics.map(topic => `<li>${topic}</li>`).join('')}
                </ul>
            </div>
            <div class="modal-actions">
                ${readBtnHtml}
                <button class="btn btn-primary" onclick="closeModal()">Close</button>
            </div>
        `;
    }
    
    showModuleDialog(modal);

    if (!modalBody.querySelector('.modal-back-to-top')) {
        modalBody.insertAdjacentHTML('beforeend', `
            <button class="modal-back-to-top" onclick="scrollModalToTop()" aria-label="Back to top">
                <i class="fas fa-arrow-up"></i>
                <span>Top</span>
            </button>
        `);
    }

    setupModalBackToTop();

    // Auto-open the first section for better discoverability.
    const firstSection = document.getElementById('content-0');
    if (firstSection && firstSection.hidden) {
        toggleContent(0);
    }

    renderMermaidDiagrams(modalBody);
    enhanceExternalLinks(modalBody);
}

let modalPreviousOverflow = '';

function showModuleDialog(modal) {
    modalPreviousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    if (!modal.open) modal.showModal();
    modal.querySelector('.modal-content').scrollTop = 0;
}

function setupModuleDialog() {
    const modal = document.getElementById('moduleModal');
    if (!modal) return;
    modal.addEventListener('close', () => {
        document.body.style.overflow = modalPreviousOverflow;
    });
    modal.addEventListener('click', event => {
        if (event.target !== modal) return;
        const bounds = modal.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeModal();
    });
}

function closeModal() {
    const modal = document.getElementById('moduleModal');
    if (modal?.open) modal.close();
}

function setupModalBackToTop() {
    const modal = document.getElementById('moduleModal');
    const modalContent = modal ? modal.querySelector('.modal-content') : null;
    const backToTopButton = document.querySelector('.modal-back-to-top');
    if (!modalContent || !backToTopButton) return;

    const updateVisibility = function () {
        const shouldShow = modalContent.scrollTop > 320;
        backToTopButton.classList.toggle('visible', shouldShow);
    };

    modalContent.onscroll = updateVisibility;
    updateVisibility();
}

function scrollModalToTop() {
    const modal = document.getElementById('moduleModal');
    const modalContent = modal ? modal.querySelector('.modal-content') : null;
    if (!modalContent) return;

    modalContent.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'instant' : 'smooth' });
}

// Toggle content section
function toggleContent(index) {
    const content = document.getElementById(`content-${index}`);
    const icon = document.getElementById(`icon-${index}`);
    
    const button = document.querySelector('[aria-controls="content-' + index + '"]');
    content.hidden = !content.hidden;
    button.setAttribute('aria-expanded', String(!content.hidden));
    icon.textContent = content.hidden ? '+' : '-';
    if (!content.hidden) {
        renderMermaidDiagrams(content);
    }
}

function expandAllContentSections() {
    const sections = document.querySelectorAll('.content-body[id^="content-"]');
    sections.forEach(function (section) {
        if (section.hidden) {
            const index = section.id.replace('content-', '');
            toggleContent(index);
        }
    });
}

function collapseAllContentSections() {
    const sections = document.querySelectorAll('.content-body[id^="content-"]');
    sections.forEach(function (section) {
        if (!section.hidden) {
            const index = section.id.replace('content-', '');
            toggleContent(index);
        }
    });
}

// Copy code to clipboard
function copyCode(index, event) {
    const codeElement = document.getElementById(`code-${index}`);
    const text = codeElement.textContent;
    
    const copyBtn = event.currentTarget;
    if (!navigator.clipboard?.writeText) {
        window.alert('Clipboard access is unavailable. Select the code and copy it manually.');
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        // Show success message
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        alert('Failed to copy code');
    });
}

// Escape HTML for code display
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Toggle Accordion
function toggleAccordion(header) {
    console.log('Toggle accordion clicked');
    const accordionItem = header.closest('.accordion-item');
    
    if (!accordionItem) {
        console.error('Accordion item not found');
        return;
    }
    
    console.log('Accordion item found:', accordionItem);
    console.log('Current active status:', accordionItem.classList.contains('active'));
    
    const allItems = document.querySelectorAll('.accordion-item');
    console.log('Total accordion items:', allItems.length);
    
    // Close all other accordions
    allItems.forEach(item => {
        if (item !== accordionItem) {
            item.classList.remove('active');
        }
    });
    
    // Toggle current accordion
    accordionItem.classList.toggle('active');
    console.log('New active status:', accordionItem.classList.contains('active'));
}
