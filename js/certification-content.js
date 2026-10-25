/* global courseData */
Object.assign(courseData, {
    aiFundamentals: [
        {
            number: "AI-901 · Module 1",
            title: "AI Concepts & Workloads",
            description: "Understand core artificial intelligence workloads and the considerations for building AI solutions on Azure.",
            duration: "45 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is Artificial Intelligence?",
                "Common AI workloads",
                "Machine learning vs. deep learning vs. generative AI",
                "Azure AI services overview",
                "Microsoft Foundry introduction"
            ],
            detailedDescription: "This module introduces the fundamental concepts of artificial intelligence and the common AI workloads you can build on Azure. You'll learn how to identify the right AI workload for a scenario, understand the Azure AI service landscape, and get an introduction to Microsoft Foundry — the unified platform for building AI solutions.",
            detailedContent: [
                {
                    title: "What is Artificial Intelligence?",
                    content: `Artificial Intelligence (AI) is software that imitates human capabilities and behaviors such as learning, reasoning, perception, and decision-making.

<strong>How AI differs from traditional software:</strong>
• Traditional code follows explicit rules written by a developer
• AI learns patterns from data and improves with more examples

<strong>Everyday examples:</strong>
• Recommendations on streaming and shopping sites
• Voice assistants that understand speech
• Fraud detection on card transactions
• Chatbots that answer questions in natural language

<strong>Why it matters for AI-901:</strong> The exam expects you to recognize AI scenarios and map them to the right Azure capability.`
                },
                {
                    title: "Common AI Workloads",
                    content: `Azure organizes AI capabilities into recognizable <strong>workloads</strong>. Being able to match a scenario to a workload is a core exam skill.

<strong>• Machine Learning</strong>
Predict values or categories from data (e.g., forecast sales).

<strong>• Computer Vision</strong>
Interpret images and video — classification, object detection, OCR, face analysis.

<strong>• Natural Language Processing (NLP)</strong>
Understand written and spoken language — sentiment, entities, translation.

<strong>• Generative AI</strong>
Create new content — text, code, images — from prompts.

<strong>• Document Intelligence & Knowledge Mining</strong>
Extract structured data from documents and search large content sets.`,
                    code: `# Map a scenario to the right Azure AI workload
scenarios = {
    "Predict next month's revenue":        "Machine Learning",
    "Read totals from scanned receipts":    "Document Intelligence",
    "Detect defects in product photos":     "Computer Vision",
    "Summarize customer support tickets":   "Natural Language Processing",
    "Draft a product description":          "Generative AI",
}

for scenario, workload in scenarios.items():
    print(f"{scenario:38} -> {workload}")`
                },
                {
                    title: "Machine Learning vs. Deep Learning vs. Generative AI",
                    content: `These related terms are often confused. The exam expects you to tell them apart.

<strong>Machine Learning (ML):</strong>
The broad field of algorithms that learn from data. Includes simple models like linear regression.

<strong>Deep Learning:</strong>
A subset of ML that uses multi-layer neural networks. Powers computer vision and speech.

<strong>Generative AI:</strong>
A recent branch built on large deep-learning models (LLMs) that generate new content rather than only classifying or predicting.

<strong>Relationship:</strong> Generative AI ⊂ Deep Learning ⊂ Machine Learning ⊂ Artificial Intelligence.`
                },
                {
                    title: "Azure AI Services Overview",
                    content: `<strong>Azure AI services</strong> are prebuilt, cloud-based APIs that add AI to apps without training your own models.

<strong>Key services:</strong>
• <strong>Azure AI Vision</strong> — image analysis and OCR
• <strong>Azure AI Language</strong> — text analytics and Q&A
• <strong>Azure AI Speech</strong> — speech-to-text and text-to-speech
• <strong>Azure AI Document Intelligence</strong> — data from forms
• <strong>Azure AI Search</strong> — knowledge mining
• <strong>Azure OpenAI</strong> — generative AI models

<strong>Provisioning:</strong> Create a single-service resource for one capability, or a multi-service resource to share one key and endpoint across several.`,
                    code: `# Every Azure AI service call needs an endpoint + credential
from azure.core.credentials import AzureKeyCredential

endpoint = "https://<your-resource>.cognitiveservices.azure.com/"
credential = AzureKeyCredential("<your-key>")

# The same endpoint/credential pattern works across
# Vision, Language, Speech, and Document Intelligence SDKs.
print("Ready to call Azure AI services at", endpoint)`
                },
                {
                    title: "Getting Started with Microsoft Foundry",
                    content: `<strong>Microsoft Foundry</strong> (formerly Azure AI Foundry) is the unified platform for building, evaluating, and deploying AI applications and agents on Azure.

<strong>Key building blocks:</strong>
• <strong>Hubs & Projects:</strong> Organize resources, data, and access
• <strong>Model catalog:</strong> Browse and deploy foundation models (GPT, Llama, Phi, etc.)
• <strong>Playground:</strong> Experiment with prompts before writing code
• <strong>Prompt flow:</strong> Orchestrate multi-step AI workflows
• <strong>Evaluations:</strong> Measure quality, safety, and groundedness

<strong>Why it matters for AI-901:</strong> Modern Azure AI solutions are increasingly built and managed through Foundry, so understanding the platform is now core to the fundamentals exam.`,
                    code: `# Connect to a Microsoft Foundry project (Python)
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project = AIProjectClient(
    endpoint="https://<your-project>.services.ai.azure.com/api/projects/<name>",
    credential=DefaultAzureCredential(),
)

# List models available in the project's catalog
for deployment in project.deployments.list():
    print(deployment.name, "->", deployment.model_name)`
                }
            ]
        },
        {
            number: "AI-901 · Module 2",
            title: "Machine Learning on Azure",
            description: "Explore the fundamental principles of machine learning and how to train models with Azure Machine Learning.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Supervised vs. unsupervised learning",
                "Regression, classification & clustering",
                "Features and labels",
                "Training and evaluation",
                "Automated machine learning"
            ],
            detailedDescription: "Machine learning is the foundation of modern AI. This module covers the core principles you need for the exam: the difference between supervised and unsupervised learning, common model types, how features and labels drive training, and how Azure Machine Learning's automated ML makes model building accessible.",
            detailedContent: [
                {
                    title: "Supervised vs. Unsupervised Learning",
                    content: `Machine learning approaches are grouped by whether the training data has known answers (labels).

<strong>Supervised learning:</strong>
• Trained on labeled examples (input + correct output)
• Learns to predict the label for new inputs
• Examples: spam detection, price prediction

<strong>Unsupervised learning:</strong>
• Trained on unlabeled data
• Discovers structure or groupings on its own
• Examples: customer segmentation, anomaly detection

<strong>Exam tip:</strong> "We have historical outcomes to learn from" → supervised. "Find natural groups" → unsupervised.`
                },
                {
                    title: "Regression, Classification & Clustering",
                    content: `The three most common model types you must recognize:

<strong>Regression (supervised):</strong>
Predicts a continuous number — house price, temperature, demand.

<strong>Classification (supervised):</strong>
Predicts a category — spam/not spam, disease/no disease. Binary (two classes) or multiclass.

<strong>Clustering (unsupervised):</strong>
Groups similar items without predefined labels — segmenting customers by behavior.

<strong>Choosing:</strong> Ask "am I predicting a number, a category, or finding groups?"`,
                    code: `# One library, three model types
from sklearn.linear_model import LinearRegression       # regression
from sklearn.tree import DecisionTreeClassifier          # classification
from sklearn.cluster import KMeans                        # clustering

model_types = {
    "Predict price":        LinearRegression(),
    "Predict spam/not":     DecisionTreeClassifier(),
    "Group customers":      KMeans(n_clusters=3),
}
for task, model in model_types.items():
    print(f"{task:18} -> {type(model).__name__}")`
                },
                {
                    title: "Features and Labels",
                    content: `Understanding features and labels is essential to how models learn.

<strong>Features (X):</strong>
The input variables — the columns you use to make a prediction (e.g., size, location, age).

<strong>Label (y):</strong>
The value you want to predict (e.g., price). Only supervised learning has labels.

<strong>Good features matter:</strong>
• Relevant to the outcome
• Clean and consistent
• Free of leakage (no information from the future)

<strong>Feature engineering</strong> — creating better inputs — often improves accuracy more than changing the algorithm.`
                },
                {
                    title: "Training and Evaluation",
                    content: `Models are trained on one portion of data and evaluated on another to check they generalize.

<strong>The workflow:</strong>
1. Split data into training and validation/test sets
2. Train the model on the training set
3. Predict on the held-out set
4. Compare predictions to known labels

<strong>Common metrics:</strong>
• Classification: accuracy, precision, recall
• Regression: RMSE, R²

<strong>Overfitting:</strong> A model that memorizes training data but fails on new data — caught by evaluating on unseen data.`,
                    code: `# Train, then evaluate on unseen data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = [[600], [800], [1000], [1200], [1400]]   # feature: size
y = [150, 200, 250, 300, 350]                 # label: price (k)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print("R2 score:", r2_score(y_test, model.predict(X_test)))`
                },
                {
                    title: "Automated Machine Learning (AutoML)",
                    content: `<strong>Automated ML</strong> in Azure Machine Learning automatically tries many algorithms and preprocessing steps to find the best model for your data — no deep ML expertise required.

<strong>What AutoML handles for you:</strong>
• Feature scaling and normalization
• Algorithm selection
• Hyperparameter tuning
• Model ranking by your chosen metric

<strong>Exam tip:</strong> AutoML supports classification, regression, and time-series forecasting, and it produces an explainability report so you can understand which features mattered most.`,
                    code: `# Configure an AutoML classification job (Azure ML SDK v2)
from azure.ai.ml import automl

classification_job = automl.classification(
    compute="cpu-cluster",
    training_data=my_training_data,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
)
classification_job.set_limits(timeout_minutes=60, max_trials=20)
# submit with ml_client.jobs.create_or_update(classification_job)`
                }
            ]
        },
        {
            number: "AI-901 · Module 3",
            title: "Computer Vision",
            description: "Discover how Azure AI Vision analyzes images and video, from classification to optical character recognition.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Image classification & object detection",
                "Optical Character Recognition (OCR)",
                "Face detection and analysis",
                "Azure AI Vision service"
            ],
            detailedDescription: "Computer vision lets applications interpret the visual world. This module covers the common vision workloads — classification, object detection, OCR, and face analysis — and how the Azure AI Vision service exposes them through simple APIs.",
            detailedContent: [
                {
                    title: "Image Classification & Object Detection",
                    content: `Two foundational computer vision tasks:

<strong>Image classification:</strong>
Assigns one or more labels to an <em>entire</em> image (e.g., "cat", "invoice").

<strong>Object detection:</strong>
Locates <em>multiple</em> objects and returns a label plus a bounding box for each.

<strong>Prebuilt vs. custom:</strong>
• <strong>Azure AI Vision</strong> offers prebuilt tagging and detection
• <strong>Custom Vision</strong> lets you train on your own labeled images when you need domain-specific classes

<strong>Exam tip:</strong> "What is in the image?" → classification. "Where are the items?" → object detection.`
                },
                {
                    title: "Optical Character Recognition (OCR)",
                    content: `<strong>OCR</strong> extracts printed and handwritten text from images and documents.

<strong>Capabilities:</strong>
• Read text from photos, scans, and screenshots
• Return text with location (bounding boxes)
• Support many languages and mixed handwriting/print

<strong>Common uses:</strong>
• Digitizing paper forms
• Reading license plates or signage
• Extracting text for search and translation

The <strong>Read API</strong> in Azure AI Vision is optimized for dense documents and returns text organized into pages, lines, and words.`,
                    code: `# Read text from an image with Azure AI Vision (Python)
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    endpoint="https://<resource>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<key>"),
)

result = client.analyze_from_url(
    image_url="https://aka.ms/azai/vision/example.jpg",
    visual_features=[VisualFeatures.READ],
)
for line in result.read.blocks[0].lines:
    print("OCR:", line.text)`
                },
                {
                    title: "Face Detection and Analysis",
                    content: `The <strong>Face</strong> capability detects human faces and can analyze attributes.

<strong>What it provides:</strong>
• Face location (bounding box) and landmarks
• Attributes such as head pose, blur, and occlusion
• Face comparison and verification (with approval)

<strong>Responsible AI:</strong>
Face recognition is a <strong>Limited Access</strong> feature. Sensitive attributes (like emotion inference) have been retired, and use requires registration to prevent misuse.

<strong>Exam tip:</strong> Know that facial recognition raises privacy considerations and is gated behind responsible-AI controls.`
                },
                {
                    title: "The Azure AI Vision Service",
                    content: `<strong>Azure AI Vision</strong> brings these capabilities together behind a single resource and SDK.

<strong>Features in one service:</strong>
• Caption and dense captions
• Tags and object detection
• OCR (Read)
• Smart crops
• People detection

<strong>Getting results:</strong>
Send an image (URL or bytes), request the visual features you need, and receive structured JSON with confidence scores.`,
                    code: `# Request multiple visual features at once
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.analyze_from_url(
    image_url="https://aka.ms/azai/vision/example.jpg",
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
)
print("Caption:", result.caption.text)
print("Tags:", [t.name for t in result.tags.list])`
                }
            ]
        },
        {
            number: "AI-901 · Module 4",
            title: "Natural Language Processing",
            description: "Learn how Azure AI Language and Speech understand text and spoken language.",
            duration: "45 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Key phrase & entity extraction",
                "Sentiment analysis",
                "Language detection & translation",
                "Speech-to-text and text-to-speech",
                "Conversational language understanding"
            ],
            detailedDescription: "Natural Language Processing enables applications to understand and generate human language. This module covers the text analysis features of Azure AI Language, the capabilities of Azure AI Speech, and how conversational language understanding powers chatbots and voice assistants.",
            detailedContent: [
                {
                    title: "Key Phrase & Entity Extraction",
                    content: `<strong>Azure AI Language</strong> turns unstructured text into structured insight.

<strong>Key phrase extraction:</strong>
Pulls out the main talking points from text — useful for tagging and summarizing.

<strong>Named entity recognition (NER):</strong>
Identifies entities like people, organizations, locations, dates, and quantities.

<strong>PII detection:</strong>
A specialized form of NER that finds and can redact personal data (emails, phone numbers).

<strong>Exam tip:</strong> These are prebuilt capabilities — no training required.`,
                    code: `# Extract key phrases and entities (Python)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

docs = ["Microsoft released Foundry in Seattle last year."]
print("Key phrases:", client.extract_key_phrases(docs)[0].key_phrases)
for entity in client.recognize_entities(docs)[0].entities:
    print(entity.text, "->", entity.category)`
                },
                {
                    title: "Sentiment Analysis",
                    content: `<strong>Sentiment analysis</strong> classifies text as positive, neutral, or negative, with confidence scores.

<strong>How it works:</strong>
• Returns an overall document sentiment
• Provides per-sentence sentiment
• Supports <strong>opinion mining</strong> to link sentiment to specific targets (e.g., "the <em>staff</em> was great")

<strong>Common uses:</strong>
• Analyzing product reviews
• Monitoring social media
• Prioritizing support tickets`,
                    code: `# Sentiment analysis with Azure AI Language (Python)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

documents = ["The course was fantastic and easy to follow!"]
result = client.analyze_sentiment(documents=documents)[0]
print("Sentiment:", result.sentiment)
print("Scores:", result.confidence_scores)`
                },
                {
                    title: "Language Detection & Translation",
                    content: `<strong>Language detection</strong> identifies the language of input text and returns an ISO code with a confidence score — handy for routing multilingual content.

<strong>Azure AI Translator</strong> provides:
• Text translation across 100+ languages
• Language auto-detection
• Transliteration (script conversion)
• Custom translation models for domain terms

<strong>Exam tip:</strong> Translation and detection are separate from the core Language service (Translator is its own resource), but both fall under NLP workloads.`,
                    code: `# Detect the language of input text (Python)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.detect_language(["Bonjour tout le monde"])[0]
print(result.primary_language.name, result.primary_language.iso6391_name)`
                },
                {
                    title: "Speech-to-Text and Text-to-Speech",
                    content: `<strong>Azure AI Speech</strong> adds voice capabilities to applications.

<strong>Speech-to-text (STT):</strong>
Transcribes spoken audio into text in real time or from files.

<strong>Text-to-speech (TTS):</strong>
Generates natural-sounding speech from text, with many neural voices and styles.

<strong>Also included:</strong>
• Speech translation
• Speaker recognition
• Custom neural voice (gated)

<strong>Common uses:</strong> voice assistants, captioning, IVR systems, accessibility.`,
                    code: `# Synthesize speech from text (Azure AI Speech SDK)
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription="<key>", region="<region>")
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

synth = speechsdk.SpeechSynthesizer(speech_config=speech_config)
synth.speak_text_async("Welcome to the Azure AI course!").get()`
                },
                {
                    title: "Conversational Language Understanding",
                    content: `<strong>Conversational Language Understanding (CLU)</strong> extracts <strong>intents</strong> (what the user wants) and <strong>entities</strong> (key details) from utterances — the brain behind chatbots and voice assistants.

<strong>Building a CLU model:</strong>
1. Define intents (e.g., BookFlight, CheckWeather)
2. Label entities (e.g., destination, date)
3. Provide example utterances and train
4. Deploy and query the model

<strong>Exam tip:</strong> CLU is a <em>custom</em> capability — you train it on your own intents, unlike prebuilt sentiment or NER.`,
                    code: `# Predict intent & entities with CLU (Python)
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential

client = ConversationAnalysisClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.analyze_conversation(task={
    "kind": "Conversation",
    "analysisInput": {"conversationItem": {
        "id": "1", "participantId": "user",
        "text": "Book a flight to Seattle on Friday"}},
    "parameters": {"projectName": "travel", "deploymentName": "prod"},
})
print("Intent:", result["result"]["prediction"]["topIntent"])`
                }
            ]
        },
        {
            number: "AI-901 · Module 5",
            title: "Generative AI & Microsoft Foundry",
            description: "Understand generative AI, large language models, and how to build solutions with Azure OpenAI and Microsoft Foundry.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is generative AI?",
                "Large language models & tokens",
                "Prompt engineering basics",
                "Azure OpenAI in Foundry",
                "Retrieval Augmented Generation (RAG)",
                "AI agents overview"
            ],
            detailedDescription: "Generative AI is transforming how applications are built. This module explains how large language models work, the basics of prompt engineering, how to ground models on your own data with RAG, and how Microsoft Foundry brings it all together — a growing focus area of the updated AI-901 exam.",
            detailedContent: [
                {
                    title: "What is Generative AI?",
                    content: `<strong>Generative AI</strong> creates new content — text, code, and images — from natural-language prompts, rather than only classifying or predicting.

<strong>What it can do:</strong>
• Draft and summarize text
• Answer questions conversationally
• Generate and explain code
• Create and edit images

<strong>Limitations to know:</strong>
• Can <strong>hallucinate</strong> (produce confident but wrong answers)
• Has a knowledge cutoff unless grounded with fresh data
• Reflects biases present in training data

<strong>Exam tip:</strong> Responsible use and grounding are recurring themes.`
                },
                {
                    title: "Large Language Models & Tokens",
                    content: `<strong>Large language models (LLMs)</strong> are deep neural networks trained on vast text to predict the next <strong>token</strong>.

<strong>Key ideas:</strong>
• <strong>Tokens:</strong> Text is broken into word pieces; models bill and limit by tokens
• <strong>Context window:</strong> The max tokens a model can consider at once
• <strong>Parameters:</strong> The learned weights; larger models often capture more nuance

<strong>Model families in Azure:</strong> GPT, Phi, Llama, and more are available in the Foundry model catalog for different cost/quality trade-offs.`,
                    code: `# Estimate tokens before calling a model
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
text = "Generative AI creates new content from prompts."
tokens = encoding.encode(text)

print("Token count:", len(tokens))
print("First tokens:", tokens[:5])`
                },
                {
                    title: "Prompt Engineering Basics",
                    content: `<strong>Prompt engineering</strong> is crafting inputs that steer the model toward useful outputs.

<strong>Techniques:</strong>
• <strong>System message:</strong> Set the model's role and rules
• <strong>Clear instructions:</strong> Be specific about format and length
• <strong>Few-shot examples:</strong> Show sample input/output pairs
• <strong>Temperature:</strong> Lower for factual, higher for creative

<strong>Good prompt pattern:</strong> role + task + context + constraints + output format.`,
                    code: `# A well-structured prompt with a system message
messages = [
    {"role": "system",
     "content": "You are a concise ML tutor. Answer in one sentence."},
    {"role": "user",
     "content": "What is overfitting?"},
]
# Lower temperature => more deterministic, factual answers
params = {"model": "gpt-4o", "messages": messages, "temperature": 0.2}
print(params)`
                },
                {
                    title: "Azure OpenAI in Foundry",
                    content: `<strong>Azure OpenAI</strong> provides OpenAI models (GPT-4o, GPT-4, embeddings, DALL·E) with Azure security, compliance, and regional hosting.

<strong>Working in Foundry:</strong>
• Deploy a model to get a named deployment + endpoint
• Test in the Playground
• Call from code with the OpenAI SDK
• Add content filters and monitoring

<strong>Exam tip:</strong> You call your <em>deployment name</em>, not the raw model name, and access is secured with keys or Microsoft Entra ID.`,
                    code: `# Chat completion with Azure OpenAI in Foundry (Python)
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<key>",
    api_version="2024-10-21",
)

response = client.chat.completions.create(
    model="gpt-4o",   # your deployment name
    messages=[{"role": "user", "content": "Explain overfitting briefly."}],
    temperature=0.3,
)
print(response.choices[0].message.content)`
                },
                {
                    title: "Retrieval Augmented Generation (RAG)",
                    content: `<strong>Retrieval Augmented Generation (RAG)</strong> combines an LLM with a search over your own content so responses are grounded in trusted data.

<strong>The RAG pattern:</strong>
1. User asks a question
2. Retrieve relevant documents from <strong>Azure AI Search</strong>
3. Add those documents to the prompt as context
4. The model answers using the supplied context

<strong>Benefits:</strong> Up-to-date answers, source citations, and reduced hallucinations — the foundation of enterprise generative AI on Azure.`,
                    code: `# RAG: retrieve context, then ground the model
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search = SearchClient(
    endpoint="https://<search>.search.windows.net",
    index_name="course-content",
    credential=AzureKeyCredential("<key>"),
)

hits = search.search(search_text="what is gradient descent", top=3)
context = "\\n".join(doc["content"] for doc in hits)

prompt = f"Answer using ONLY this context:\\n{context}"
# pass prompt to the chat completion call from the previous lesson`
                },
                {
                    title: "AI Agents Overview",
                    content: `<strong>AI agents</strong> extend generative AI from answering questions to <em>taking actions</em> toward a goal.

<strong>What makes an agent:</strong>
• A model with <strong>instructions</strong> (its role)
• <strong>Tools</strong> it can call (functions, search, code)
• <strong>Memory</strong> of the conversation
• A loop that plans, acts, and observes results

<strong>On Azure:</strong> The Microsoft Foundry Agent Service manages threads, tool calls, and state so you can build agents without wiring all the plumbing yourself.

<strong>Exam tip:</strong> Agents are an emerging focus — know that they combine LLMs, tools, and orchestration.`
                }
            ]
        },
        {
            number: "AI-901 · Module 6",
            title: "Responsible AI",
            description: "Learn Microsoft's six responsible AI principles and how to apply them to Azure AI solutions.",
            duration: "35 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "The six responsible AI principles",
                "Fairness and inclusiveness",
                "Transparency and accountability",
                "Content safety and guardrails"
            ],
            detailedDescription: "Responsible AI ensures solutions are fair, safe, and trustworthy. This module covers Microsoft's six guiding principles and the practical tools — like Azure AI Content Safety and the Responsible AI dashboard — that help you build responsible solutions.",
            detailedContent: [
                {
                    title: "The Six Responsible AI Principles",
                    content: `Microsoft's responsible AI framework rests on six principles:

1. <strong>Fairness:</strong> Treat all people equitably; avoid bias
2. <strong>Reliability & Safety:</strong> Perform consistently and handle failure safely
3. <strong>Privacy & Security:</strong> Protect data and respect consent
4. <strong>Inclusiveness:</strong> Empower and engage people of all abilities
5. <strong>Transparency:</strong> Make systems understandable
6. <strong>Accountability:</strong> People remain responsible for AI systems

<strong>Exam tip:</strong> Expect scenario questions asking which principle applies.`
                },
                {
                    title: "Fairness and Inclusiveness",
                    content: `<strong>Fairness</strong> means an AI system treats all groups equitably and does not amplify bias present in data.

<strong>Sources of unfairness:</strong>
• Unrepresentative training data
• Proxy features correlated with sensitive attributes
• Feedback loops that reinforce bias

<strong>Inclusiveness</strong> ensures solutions work for people of all abilities and backgrounds — for example, captions for the hearing impaired and voice input for those who can't type.

<strong>Tool:</strong> The <strong>Fairlearn</strong> integration in the Responsible AI dashboard measures performance across groups.`
                },
                {
                    title: "Transparency and Accountability",
                    content: `<strong>Transparency</strong> means people can understand how a system works and its limitations.

<strong>Practices:</strong>
• Explain what data is used and why
• Provide model explanations (feature importance)
• Communicate confidence and known limits

<strong>Accountability</strong> means humans — not the AI — remain responsible for outcomes.

<strong>Practices:</strong>
• Clear ownership and governance
• Human oversight for high-impact decisions
• Auditing and monitoring in production`,
                    code: `# Explain predictions to support transparency (Azure ML)
from azureml.interpret import ExplanationClient

client = ExplanationClient.from_run(run)
explanation = client.download_model_explanation()

# Which features drove the model's decisions?
for feature, importance in explanation.get_feature_importance_dict().items():
    print(f"{feature:20} {importance:.3f}")`
                },
                {
                    title: "Content Safety and Guardrails",
                    content: `<strong>Azure AI Content Safety</strong> detects and filters harmful content across four categories: hate, sexual, violence, and self-harm — with severity levels.

<strong>Guardrails for generative AI:</strong>
• Input and output content filters
• Prompt-shield against jailbreak attempts
• Groundedness detection for hallucinations
• Blocklists for domain-specific terms

<strong>Exam tip:</strong> Content Safety applies to both user input and model output, and is a key part of building responsible generative AI apps.`,
                    code: `# Screen text with Azure AI Content Safety (Python)
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential

client = ContentSafetyClient(
    endpoint="https://<resource>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<key>"),
)

result = client.analyze_text(AnalyzeTextOptions(text="You are welcome here!"))
for category in result.categories_analysis:
    print(category.category, "severity:", category.severity)`
                }
            ]
        },
        {
            number: "AI-901 · Module 7",
            title: "Document Intelligence, Knowledge Mining & Service Selection",
            description: "Round out the exam breadth: extract data from documents, mine knowledge with Azure AI Search, apply Content Safety, and choose and provision the right service.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Document Intelligence",
                "Knowledge mining with Azure AI Search",
                "Azure AI Content Safety",
                "Single-service vs multi-service resources",
                "Authentication: keys vs Entra ID",
                "Choosing the right Azure AI service"
            ],
            detailedDescription: "AI-901 tests the full breadth of Azure AI workloads. This module covers the remaining high-value services - Document Intelligence, knowledge mining with Azure AI Search, and Content Safety - plus how to provision and secure Azure AI resources and pick the right service for a scenario.",
            detailedContent: [
                {
                    title: "Document Intelligence",
                    content: `<strong>Azure AI Document Intelligence</strong> extracts text, key-value pairs, tables, and structure from documents.

<strong>Model options:</strong>
• <strong>Prebuilt models:</strong> invoices, receipts, IDs, business cards, tax forms
• <strong>Layout model:</strong> text, tables, and structure from any document
• <strong>Custom models:</strong> train on your own forms

<strong>Exam tip:</strong> Document Intelligence is the answer when a scenario needs structured data extracted from forms or documents - not general OCR alone.`,
                    code: `# Extract fields from an invoice (Azure AI Document Intelligence)
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

poller = client.begin_analyze_document(
    "prebuilt-invoice",
    {"urlSource": "https://aka.ms/azai/invoice-sample.pdf"})
for doc in poller.result().documents:
    total = doc.fields.get("InvoiceTotal")
    if total:
        print("Invoice total:", total.get("content"))`
                },
                {
                    title: "Knowledge Mining with Azure AI Search",
                    content: `<strong>Knowledge mining</strong> makes large content sets searchable and is the backbone of enterprise RAG.

<strong>Azure AI Search core objects:</strong>
• <strong>Data source:</strong> Blob, SQL, Cosmos DB
• <strong>Index:</strong> the searchable schema (fields, vectors)
• <strong>Indexer:</strong> pulls and populates the index
• <strong>Skillset:</strong> AI enrichment (OCR, entities, embeddings) during indexing

<strong>Exam tip:</strong> Azure AI Search is the knowledge-mining and retrieval service - and the vector store behind grounding for generative AI.`
                },
                {
                    title: "Azure AI Content Safety",
                    content: `<strong>Azure AI Content Safety</strong> detects and filters harmful content across four categories: hate, sexual, violence, and self-harm, each with severity levels.

<strong>Capabilities:</strong>
• Analyze text and images for harmful content
• Prompt shields against jailbreak attempts
• Groundedness detection for hallucinations
• Custom blocklists

<strong>Exam tip:</strong> Content Safety applies to both user input and model output and is central to building responsible generative AI apps.`
                },
                {
                    title: "Single-Service vs Multi-Service Resources",
                    content: `Provisioning is a common exam topic.

<strong>Single-service resource:</strong>
• One capability (for example, just Vision)
• Its own key and endpoint, granular billing

<strong>Multi-service (Azure AI services / Foundry) resource:</strong>
• Many capabilities behind one key and endpoint
• Simpler management for solutions that combine services

<strong>Exam tip:</strong> Prefer a multi-service resource when a solution uses several AI capabilities together.`
                },
                {
                    title: "Authentication: Keys vs Entra ID",
                    content: `Clients authenticate to Azure AI services in two main ways.

<strong>API keys:</strong>
• Simple; pass the key with each request
• Good for development; rotate regularly; never commit to source

<strong>Microsoft Entra ID (recommended for production):</strong>
• Managed identities - no secrets in code
• Role-based access control (least privilege)
• Central auditing

<strong>Exam tip:</strong> For production, prefer Entra ID with managed identities over keys.`
                },
                {
                    title: "Choosing the Right Azure AI Service",
                    content: `The exam frequently asks you to map a scenario to a service.

<strong>Quick map:</strong>
• Analyze images / OCR -> Azure AI Vision
• Extract data from forms -> Document Intelligence
• Sentiment, entities, language -> Azure AI Language
• Speech-to-text / text-to-speech -> Azure AI Speech
• Translate text -> Azure AI Translator
• Search over content -> Azure AI Search
• Generate content -> Azure OpenAI in Foundry
• Filter harmful content -> Content Safety

<strong>Rule:</strong> pick the prebuilt service that matches the workload before considering custom models.`,
                    code: `# Map scenarios to the right Azure AI service
services = {
    "Read text from scanned receipts": "Document Intelligence",
    "Detect objects in photos":        "Azure AI Vision",
    "Summarize support tickets":       "Azure AI Language",
    "Transcribe a meeting recording":  "Azure AI Speech",
    "Answer from company documents":   "Azure AI Search + Azure OpenAI",
}
for scenario, service in services.items():
    print(f"{scenario:34} -> {service}")`
                }
            ]
        },
        {
            number: "AI-901 · Module 8",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Azure AI Studio/Foundry labs plus a domain-mapped study plan and exam-day strategy for AI-901.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Language Studio sentiment",
                "Lab: Vision Studio image analysis",
                "Lab: Foundry playground & Azure OpenAI",
                "Lab: Content Safety",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AI-901 concepts into practice. This module provides four guided labs you can run in a free Azure account, then maps your study to the official exam domains and gives a practical exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Language Studio Sentiment",
                    content: `<strong>Goal:</strong> experience prebuilt NLP with no code.

<strong>Steps:</strong>
1. In the Azure portal, create an Azure AI Language resource.
2. Open Language Studio and pick "Analyze sentiment and opinions".
3. Paste a few product reviews and run the analysis.
4. Note document- and sentence-level sentiment and confidence scores.

<strong>Takeaway:</strong> prebuilt language features work instantly - no training required.`,
                    code: `# The same analysis in code (Azure AI Language)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))
result = client.analyze_sentiment(["The course was fantastic!"])[0]
print(result.sentiment, result.confidence_scores)`
                },
                {
                    title: "Lab: Vision Studio Image Analysis",
                    content: `<strong>Goal:</strong> see computer vision capabilities firsthand.

<strong>Steps:</strong>
1. Create an Azure AI Vision resource.
2. Open Vision Studio and try "Add captions to images".
3. Upload an image and review the caption, tags, and objects.
4. Try the "Extract text (OCR)" feature on a photo of text.

<strong>Takeaway:</strong> one service provides captions, tags, objects, and OCR through simple APIs.`
                },
                {
                    title: "Lab: Foundry Playground & Azure OpenAI",
                    content: `<strong>Goal:</strong> experience generative AI and grounding.

<strong>Steps:</strong>
1. In Microsoft Foundry, deploy a GPT model.
2. Open the Chat playground and set a system message.
3. Ask a question, then add your own data to ground the answer.
4. Compare answers with and without grounding.

<strong>Takeaway:</strong> Foundry is where you build, test, and deploy generative AI; grounding on your data reduces hallucinations.`,
                    code: `# Chat completion with Azure OpenAI in Foundry
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<key>", api_version="2024-10-21")
resp = client.chat.completions.create(
    model="gpt-4o",  # your deployment name
    messages=[{"role": "user", "content": "Explain Azure AI in one line."}])
print(resp.choices[0].message.content)`
                },
                {
                    title: "Lab: Content Safety",
                    content: `<strong>Goal:</strong> understand responsible AI guardrails in practice.

<strong>Steps:</strong>
1. Create an Azure AI Content Safety resource.
2. Open Content Safety Studio and try "Moderate text content".
3. Test benign and borderline text and review category severities.
4. Explore prompt shields for jailbreak detection.

<strong>Takeaway:</strong> Content Safety screens both input and output - a core part of responsible generative AI.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official AI-901 (Azure AI Fundamentals) domains.

<strong>Describe AI workloads & considerations:</strong> Modules 1 and 6 (workloads, responsible AI).
<strong>ML principles on Azure:</strong> Module 2 (supervised/unsupervised, training, AutoML).
<strong>Computer vision:</strong> Module 3 (classification, detection, OCR, faces).
<strong>NLP:</strong> Module 4 plus Module 7 (language, speech, translation, Document Intelligence, knowledge mining).
<strong>Generative AI:</strong> Module 5 plus Module 8 labs (LLMs, prompts, RAG, Foundry).

<strong>Recommended plan:</strong>
• Week 1: Modules 1-3 + Vision/Language labs
• Week 2: Modules 4-5 + 7 + Foundry lab
• Week 3: Modules 6 + 8 + Microsoft Learn practice assessment
• Week 4: Review + full practice test

<strong>Pair with:</strong> the free Microsoft Learn AI-900/AI-901 learning paths and official practice assessment.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AI-901 is conceptual and scenario-based - no coding required.

<strong>During the exam:</strong>
• Match each scenario to the correct workload (vision, NLP, generative, ML)
• Pick the prebuilt service that fits before considering custom models
• For "combine several capabilities," choose a multi-service/Foundry resource
• For production auth, prefer Entra ID and managed identities
• When responsible AI is mentioned, map to the correct principle or Content Safety

<strong>Mindset:</strong> you are identifying the right Azure capability for a business need, not writing code.`
                }
            ]
        }
    ],

    // ==========================================================
    // AI-103: Azure AI Apps and Agents Developer Associate
    // ==========================================================
    aiAppsAgents: [
        {
            number: "AI-103 · Module 1",
            title: "Plan & Manage an Azure AI Solution",
            description: "Select, provision, and secure Azure AI services and Microsoft Foundry resources for production solutions.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Choosing Azure AI services",
                "Provisioning resources & endpoints",
                "Authentication and keys",
                "Managing cost and scale",
                "Monitoring and diagnostics"
            ],
            detailedDescription: "Before building AI features you must plan the right services and manage them securely. This module covers selecting Azure AI services, provisioning Foundry resources, authenticating clients, and monitoring solutions in production — the first skill area of the AI-103 exam.",
            detailedContent: [
                {
                    title: "Choosing Azure AI Services",
                    content: `Selecting the right service is the first design decision.

<strong>Single-service vs. multi-service:</strong>
• <strong>Single-service</strong> resource — one capability, its own key and endpoint, granular billing
• <strong>Multi-service / Foundry</strong> resource — many capabilities behind one key, simpler management

<strong>Selection factors:</strong>
• Required capabilities (vision, language, generative)
• Data residency and region
• Pricing tier and expected volume
• Network isolation needs

<strong>Exam tip:</strong> Prefer a Foundry (multi-service) resource when a solution combines several AI capabilities.`
                },
                {
                    title: "Provisioning Resources & Endpoints",
                    content: `Every Azure AI resource exposes an <strong>endpoint</strong> (the URL your SDK calls) and lives in a resource group and region.

<strong>Ways to provision:</strong>
• Azure portal — quick, interactive
• Azure CLI / PowerShell — scriptable
• Bicep / ARM / Terraform — repeatable infrastructure as code

<strong>Foundry projects</strong> add a project-level endpoint that ties together models, connections, and deployments.`,
                    code: `# Provision a Foundry (AI Services) resource with Azure CLI
# az login first, then:
az cognitiveservices account create \\
  --name my-ai-resource \\
  --resource-group my-rg \\
  --kind AIServices \\
  --sku S0 \\
  --location eastus \\
  --yes

# Retrieve the endpoint and keys
az cognitiveservices account show \\
  --name my-ai-resource --resource-group my-rg \\
  --query properties.endpoint`
                },
                {
                    title: "Authentication and Keys",
                    content: `Clients authenticate to Azure AI services in two main ways.

<strong>API keys:</strong>
• Simple; pass the key with each request
• Good for development and quick tests
• Rotate regularly; never commit to source control

<strong>Microsoft Entra ID (recommended for production):</strong>
• Use managed identities — no secrets in code
• Role-based access control (RBAC) with least privilege
• Central credential management and auditing

<strong>Exam tip:</strong> For production, prefer Entra ID + managed identity over keys.`,
                    code: `# Two ways to authenticate the same client
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Development: key-based
key_cred = AzureKeyCredential("<key>")

# Production: keyless via managed identity / Entra ID
entra_cred = DefaultAzureCredential()

print("Use entra_cred in production to avoid storing secrets.")`
                },
                {
                    title: "Managing Cost and Scale",
                    content: `Production AI solutions must stay within budget while handling load.

<strong>Cost levers:</strong>
• Choose the right pricing tier (Free F0 for dev, Standard for prod)
• Right-size model deployments and throughput units (TPM/PTU)
• Cache and batch requests where possible

<strong>Scale & reliability:</strong>
• Handle HTTP 429 (throttling) with exponential backoff and retries
• Use regional deployments and quotas
• Monitor token usage for generative workloads`,
                    code: `# Robust calls: retry on throttling (HTTP 429)
import time
from openai import AzureOpenAI, RateLimitError

def call_with_retry(client, **kwargs):
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = 2 ** attempt           # exponential backoff
            print(f"Throttled, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Exceeded retry attempts")`
                },
                {
                    title: "Monitoring and Diagnostics",
                    content: `Once live, a solution must be observable.

<strong>What to monitor:</strong>
• Request volume, latency, and error rates
• Throttling (429s) and quota consumption
• Token usage and cost for generative models
• Content-filter and safety events

<strong>How:</strong>
• Enable <strong>diagnostic settings</strong> to send logs to Log Analytics
• Build dashboards and <strong>alerts</strong> in Azure Monitor
• Trace end-to-end with Application Insights

<strong>Exam tip:</strong> Diagnostic settings + Azure Monitor are the standard answer for observability.`,
                    code: `# Route resource logs to Log Analytics (Azure CLI)
az monitor diagnostic-settings create \\
  --name ai-diagnostics \\
  --resource <ai-resource-id> \\
  --workspace <log-analytics-workspace-id> \\
  --logs '[{"category":"RequestResponse","enabled":true}]' \\
  --metrics '[{"category":"AllMetrics","enabled":true}]'`
                }
            ]
        },
        {
            number: "AI-103 · Module 2",
            title: "Generative AI & Agentic Solutions",
            description: "Build generative AI apps and autonomous agents using Azure OpenAI and the Foundry Agent Service.",
            duration: "60 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prompt engineering & system messages",
                "Retrieval Augmented Generation (RAG)",
                "Function / tool calling",
                "Building agents with Foundry Agent Service",
                "Multi-agent orchestration",
                "Evaluations and safety"
            ],
            detailedDescription: "The heart of AI-103 is building generative and agentic solutions. This module covers grounding models with RAG, extending them with tool calling, and creating autonomous agents with the Microsoft Foundry Agent Service that can reason and act on behalf of users.",
            detailedContent: [
                {
                    title: "Prompt Engineering & System Messages",
                    content: `Prompt design is the primary way developers control generative behavior.

<strong>The message roles:</strong>
• <strong>system</strong> — sets persona, rules, and output format
• <strong>user</strong> — the request
• <strong>assistant</strong> — prior model responses (context)

<strong>Techniques:</strong>
• Give explicit, unambiguous instructions
• Provide few-shot examples for tricky formats
• Constrain output (JSON, length, tone)
• Lower temperature for deterministic results`,
                    code: `# System message + few-shot pattern
messages = [
    {"role": "system",
     "content": "Classify tickets as Billing, Technical, or Other. Reply with one word."},
    {"role": "user", "content": "My payment failed twice."},
    {"role": "assistant", "content": "Billing"},
    {"role": "user", "content": "The app crashes on launch."},
]
# Model continues the pattern -> "Technical"
print(messages[-1])`
                },
                {
                    title: "Retrieval Augmented Generation (RAG)",
                    content: `<strong>RAG</strong> grounds a model on your own data so answers are accurate and current.

<strong>The pattern:</strong>
1. Convert the question to an embedding
2. Retrieve top matches from <strong>Azure AI Search</strong> (vector or hybrid)
3. Insert retrieved passages into the prompt as context
4. Ask the model to answer using only that context

<strong>Benefits:</strong> Fresh answers, citations, and fewer hallucinations. This is the dominant enterprise pattern on Azure.`,
                    code: `# Grounding a model with retrieved context
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search = SearchClient(
    "https://<search>.search.windows.net", "docs",
    AzureKeyCredential("<key>"))

hits = search.search(search_text="how to deploy an endpoint", top=3)
context = "\\n".join(d["content"] for d in hits)

grounded_prompt = [
    {"role": "system", "content": "Answer ONLY from the context."},
    {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: How do I deploy?"},
]
print(grounded_prompt)`
                },
                {
                    title: "Function / Tool Calling",
                    content: `<strong>Tool (function) calling</strong> lets a model invoke your code to fetch data or take actions, then use the results in its response.

<strong>How it works:</strong>
1. Describe available functions to the model
2. The model decides when to call one and returns the arguments
3. Your app runs the function and returns the result
4. The model produces a grounded final answer

This is the foundation of <strong>agents</strong> — models that plan, call tools, and act toward a goal.`,
                    code: `# Tool calling with Azure OpenAI (Python)
tools = [{
    "type": "function",
    "function": {
        "name": "get_enrollment",
        "description": "Get a student's enrolled course count",
        "parameters": {
            "type": "object",
            "properties": {"student_id": {"type": "string"}},
            "required": ["student_id"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "How many courses is S-42 in?"}],
    tools=tools,
)
# Inspect response.choices[0].message.tool_calls to run the function`
                },
                {
                    title: "Building Agents with Foundry Agent Service",
                    content: `The <strong>Microsoft Foundry Agent Service</strong> lets you build agents that combine an LLM with instructions, tools, and knowledge — with conversation state handled for you.

<strong>Core concepts:</strong>
• <strong>Agent:</strong> Model + instructions + tools
• <strong>Thread:</strong> A persistent conversation
• <strong>Run:</strong> The agent processing the thread and invoking tools
• <strong>Tools:</strong> Code interpreter, file search (RAG), or custom functions

The service manages threads, tool invocation, and run state so you focus on behavior, not plumbing.`,
                    code: `# Create and run an agent with Foundry Agent Service (Python)
agent = project.agents.create_agent(
    model="gpt-4o",
    name="course-advisor",
    instructions="You help students choose Azure AI courses.",
    tools=[{"type": "file_search"}],
)

thread = project.agents.create_thread()
project.agents.create_message(
    thread_id=thread.id, role="user",
    content="Which cert should I take after AI-901?")

run = project.agents.create_and_process_run(
    thread_id=thread.id, agent_id=agent.id)
print("Run status:", run.status)`
                },
                {
                    title: "Multi-Agent Orchestration",
                    content: `Complex problems are often solved by <strong>multiple specialized agents</strong> working together.

<strong>Common patterns:</strong>
• <strong>Orchestrator–worker:</strong> A lead agent delegates subtasks to specialists
• <strong>Sequential:</strong> Agents form a pipeline, each refining the output
• <strong>Group chat:</strong> Agents collaborate and critique in a shared thread

<strong>On Azure:</strong> Frameworks like Semantic Kernel and the Foundry Agent Service coordinate connected agents, tool sharing, and hand-offs.

<strong>Exam tip:</strong> Know that decomposing work across agents improves reliability for multi-step goals.`,
                    code: `# Connect a specialist agent as a tool for a lead agent (concept)
lead = project.agents.create_agent(
    model="gpt-4o", name="planner",
    instructions="Break the goal into steps and delegate to specialists.")

researcher = project.agents.create_agent(
    model="gpt-4o", name="researcher",
    instructions="Answer factual questions using file_search.",
    tools=[{"type": "file_search"}])

# The planner calls the researcher as a connected tool/agent
print("Lead:", lead.name, "| Specialist:", researcher.name)`
                },
                {
                    title: "Evaluations and Safety",
                    content: `Generative solutions must be measured and guarded before and after release.

<strong>Evaluations:</strong>
• Quality metrics: relevance, coherence, fluency
• Grounding metrics: groundedness, retrieval accuracy
• Risk metrics: hate, violence, self-harm, jailbreak

<strong>Safety guardrails:</strong>
• Content filters on input and output
• Prompt shields against jailbreaks
• Groundedness detection for hallucinations

<strong>Exam tip:</strong> Use the Azure AI Evaluation SDK to score responses and the content-safety filters to enforce guardrails.`,
                    code: `# Evaluate relevance of generated answers (Azure AI Evaluation)
from azure.ai.evaluation import RelevanceEvaluator

relevance = RelevanceEvaluator(model_config)
score = relevance(
    query="How do I deploy a model?",
    response="Create an online endpoint, then a deployment, then route traffic.",
)
print("Relevance score:", score)`
                }
            ]
        },
        {
            number: "AI-103 · Module 3",
            title: "Computer Vision Solutions",
            description: "Implement image analysis, custom classification, and spatial analysis with Azure AI Vision.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Image analysis with Azure AI Vision",
                "Custom image classification & object detection",
                "Optical Character Recognition (OCR)",
                "Face detection and recognition",
                "Video analysis"
            ],
            detailedDescription: "This module covers building production computer vision features: analyzing images with prebuilt models, training custom classifiers and detectors, reading text with OCR, and analyzing faces — all through the Azure AI Vision SDKs.",
            detailedContent: [
                {
                    title: "Image Analysis with Azure AI Vision",
                    content: `<strong>Azure AI Vision</strong> provides prebuilt image analysis with no training required.

<strong>Features you can request:</strong>
• Caption and dense captions
• Tags and objects
• People and smart crops
• Read (OCR)

You send an image (URL or bytes), specify the visual features, and receive structured JSON with confidence scores.`,
                    code: `# Request multiple visual features at once (Python)
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.analyze_from_url(
    image_url="https://aka.ms/azai/vision/example.jpg",
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.OBJECTS])
print("Caption:", result.caption.text)`
                },
                {
                    title: "Custom Image Classification & Object Detection",
                    content: `When prebuilt models aren't enough, train a <strong>custom model</strong> on your own images.

<strong>Two custom tasks:</strong>
• <strong>Image classification:</strong> Predict one or more labels per image
• <strong>Object detection:</strong> Predict labels plus bounding boxes

<strong>Workflow:</strong>
1. Create a Custom Vision / Foundry project
2. Upload and tag training images
3. Train and evaluate (precision, recall, mAP)
4. Publish an endpoint and call it from your app

<strong>Tip:</strong> Balanced classes and at least 15-30 images per tag improve accuracy.`,
                    code: `# Predict with a published Custom Vision model (Python)
from azure.cognitiveservices.vision.customvision.prediction \\
    import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

credentials = ApiKeyCredentials(in_headers={"Prediction-key": "<key>"})
predictor = CustomVisionPredictionClient("<endpoint>", credentials)

with open("sample.jpg", "rb") as image:
    results = predictor.classify_image("<project-id>", "<model-name>", image.read())

for prediction in results.predictions:
    print(f"{prediction.tag_name}: {prediction.probability:.1%}")`
                },
                {
                    title: "Optical Character Recognition (OCR)",
                    content: `The <strong>Read</strong> capability extracts printed and handwritten text from images and documents.

<strong>Results structure:</strong>
• Pages → lines → words
• Bounding boxes for each element
• Confidence scores

<strong>Use cases:</strong> digitizing forms, reading signage, extracting text for search and translation. For dense, structured documents, pair OCR with Document Intelligence.`,
                    code: `# Read text from an image (Python)
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.analyze_from_url(
    image_url="https://aka.ms/azai/vision/example.jpg",
    visual_features=[VisualFeatures.READ])
for line in result.read.blocks[0].lines:
    print("OCR:", line.text)`
                },
                {
                    title: "Face Detection and Recognition",
                    content: `The <strong>Face</strong> service detects faces and, with approval, supports verification and identification.

<strong>Capabilities:</strong>
• Detect faces with bounding boxes and landmarks
• Attributes: head pose, blur, occlusion, glasses
• Verify (1:1) and identify (1:N) against a person group

<strong>Responsible AI:</strong> Face recognition is a <strong>Limited Access</strong> feature requiring registration. Emotion inference and some attributes were retired to protect privacy.`,
                    code: `# Detect faces in an image (Python)
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from azure.core.credentials import AzureKeyCredential

client = FaceClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

faces = client.detect_from_url(
    url="https://aka.ms/azai/vision/people.jpg",
    detection_model=FaceDetectionModel.DETECTION03,
    recognition_model=FaceRecognitionModel.RECOGNITION04,
    return_face_id=False)
print("Faces found:", len(faces))`
                },
                {
                    title: "Video Analysis",
                    content: `Beyond still images, Azure analyzes <strong>video</strong> for insights and real-time scenarios.

<strong>Options:</strong>
• <strong>Azure AI Video Indexer</strong> — extract transcripts, faces, objects, topics, and scenes from recorded video
• <strong>Spatial analysis</strong> — detect people's presence and movement in live camera feeds (e.g., occupancy, distancing)

<strong>Use cases:</strong> media search, content moderation, retail analytics, and safety monitoring.

<strong>Exam tip:</strong> Video Indexer = insights from recorded media; spatial analysis = real-time people analytics.`
                }
            ]
        },
        {
            number: "AI-103 · Module 4",
            title: "Text Analysis Solutions",
            description: "Implement NLP features such as entity recognition, sentiment, translation, and conversational language understanding.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Named entity recognition & PII detection",
                "Sentiment and opinion mining",
                "Text translation",
                "Custom question answering",
                "Conversational language understanding (CLU)"
            ],
            detailedDescription: "Text analysis solutions turn unstructured language into structured insight. This module covers entity and PII detection, sentiment mining, translation, and building custom conversational language understanding models for intent and entity extraction.",
            detailedContent: [
                {
                    title: "Named Entity Recognition & PII Detection",
                    content: `<strong>Named entity recognition (NER)</strong> identifies entities such as people, organizations, locations, dates, and quantities in text.

<strong>PII detection</strong> is a specialized form that finds — and can redact — personal data like emails, phone numbers, and IDs.

<strong>Uses:</strong>
• Tag and route documents
• Redact sensitive data before storage
• Populate structured records from free text

Both are prebuilt capabilities of Azure AI Language.`,
                    code: `# Detect and redact PII (Python)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

docs = ["Call me at 555-123-4567 or email sam@contoso.com"]
result = client.recognize_pii_entities(docs)[0]
print("Redacted:", result.redacted_text)`
                },
                {
                    title: "Sentiment and Opinion Mining",
                    content: `<strong>Sentiment analysis</strong> classifies text as positive, neutral, or negative with confidence scores, at both document and sentence level.

<strong>Opinion mining</strong> goes deeper, linking sentiment to specific <em>targets</em> and <em>assessments</em> — e.g., "the <strong>staff</strong> (target) was <strong>friendly</strong> (positive assessment)."

<strong>Uses:</strong> review analysis, brand monitoring, support prioritization.`,
                    code: `# Sentiment with opinion mining (Python)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

docs = ["The room was clean but the service was slow."]
result = client.analyze_sentiment(docs, show_opinion_mining=True)[0]
print("Overall:", result.sentiment)
for sentence in result.sentences:
    for opinion in sentence.mined_opinions:
        print(opinion.target.text, "->", opinion.target.sentiment)`
                },
                {
                    title: "Text Translation",
                    content: `<strong>Azure AI Translator</strong> provides real-time machine translation across 100+ languages.

<strong>Capabilities:</strong>
• Translate text between languages
• Auto-detect the source language
• Transliterate between scripts
• <strong>Custom Translator</strong> for domain-specific terminology

<strong>Exam tip:</strong> Translator is its own resource, separate from the core Language service.`,
                    code: `# Translate text with Azure AI Translator (REST via requests)
import requests, uuid

endpoint = "https://api.cognitive.microsofttranslator.com/translate"
params = {"api-version": "3.0", "to": ["es", "fr"]}
headers = {
    "Ocp-Apim-Subscription-Key": "<key>",
    "Ocp-Apim-Subscription-Region": "<region>",
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}
body = [{"text": "Welcome to the Azure AI course"}]
resp = requests.post(endpoint, params=params, headers=headers, json=body)
print(resp.json())`
                },
                {
                    title: "Custom Question Answering",
                    content: `<strong>Custom question answering</strong> builds a knowledge base from your FAQs, documents, and URLs, then answers user questions conversationally.

<strong>Workflow:</strong>
1. Import sources (FAQ pages, PDFs, manuals)
2. Auto-generate question/answer pairs
3. Add alternate phrasings and follow-up prompts
4. Test, then deploy as an endpoint

<strong>Uses:</strong> support bots, help desks, and self-service portals — often combined with CLU for richer bots.`,
                    code: `# Query a custom question answering project (Python)
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

client = QuestionAnsweringClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

output = client.get_answers(
    question="How do I reset my password?",
    project_name="support-kb", deployment_name="production")
for answer in output.answers:
    print(round(answer.confidence, 2), answer.answer)`
                },
                {
                    title: "Conversational Language Understanding (CLU)",
                    content: `<strong>Conversational Language Understanding (CLU)</strong> extracts <strong>intents</strong> (what the user wants) and <strong>entities</strong> (key details) from utterances — the brain behind chatbots and voice assistants.

<strong>Building a CLU model:</strong>
1. Define intents (e.g., BookFlight, CheckStatus)
2. Label entities (e.g., destination, date)
3. Provide example utterances
4. Train, test, and deploy the model

At runtime you send an utterance and receive the top intent plus extracted entities, which your app uses to route the request.`,
                    code: `# Predict intent & entities with CLU (Python)
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential

client = ConversationAnalysisClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

result = client.analyze_conversation(task={
    "kind": "Conversation",
    "analysisInput": {"conversationItem": {
        "id": "1", "participantId": "user",
        "text": "Book a flight to Seattle on Friday"}},
    "parameters": {"projectName": "travel", "deploymentName": "prod"},
})
prediction = result["result"]["prediction"]
print("Intent:", prediction["topIntent"])`
                }
            ]
        },
        {
            number: "AI-103 · Module 5",
            title: "Information Extraction Solutions",
            description: "Extract structured data from documents and build knowledge mining pipelines with Document Intelligence and Azure AI Search.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prebuilt document models",
                "Custom extraction models",
                "Knowledge mining with Azure AI Search",
                "Skillsets and enrichment pipelines",
                "Vector search for RAG"
            ],
            detailedDescription: "Information extraction unlocks the data trapped in documents and content stores. This module covers Azure AI Document Intelligence for structured extraction and Azure AI Search for knowledge mining and vector-based retrieval that powers RAG applications.",
            detailedContent: [
                {
                    title: "Prebuilt Document Models",
                    content: `<strong>Azure AI Document Intelligence</strong> ships with prebuilt models for common document types.

<strong>Available prebuilt models:</strong>
• Invoices, receipts, and tax forms
• Identity documents and business cards
• The general <strong>layout</strong> model (text, tables, structure)

Each returns structured JSON with fields, values, and confidence scores — no training required.`,
                    code: `# Extract fields from an invoice (Python)
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

poller = client.begin_analyze_document(
    "prebuilt-invoice",
    {"urlSource": "https://aka.ms/azai/invoice-sample.pdf"})
result = poller.result()

for doc in result.documents:
    total = doc.fields.get("InvoiceTotal")
    if total:
        print("Invoice total:", total.get("content"))`
                },
                {
                    title: "Custom Extraction Models",
                    content: `When your forms are unique, train a <strong>custom model</strong>.

<strong>Custom model types:</strong>
• <strong>Custom template:</strong> Fixed layouts (fast, few samples)
• <strong>Custom neural:</strong> Varied layouts (more robust)
• <strong>Composed models:</strong> Combine several custom models behind one endpoint

<strong>Workflow:</strong> Label ~5+ sample documents, train, evaluate accuracy, then analyze new documents against your model ID.`,
                    code: `# Analyze a document with a trained custom model (Python)
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))

poller = client.begin_analyze_document(
    "my-custom-model-id",
    {"urlSource": "https://<storage>/form.pdf"})
for doc in poller.result().documents:
    for name, field in doc.fields.items():
        print(name, "=", field.get("content"))`
                },
                {
                    title: "Knowledge Mining with Azure AI Search",
                    content: `<strong>Azure AI Search</strong> makes large content sets searchable and powers RAG.

<strong>Core objects:</strong>
• <strong>Data source:</strong> Blob storage, SQL, Cosmos DB
• <strong>Index:</strong> The searchable schema (fields, analyzers, vectors)
• <strong>Indexer:</strong> Pulls data and populates the index
• <strong>Skillset:</strong> AI enrichment during indexing

Together these turn raw documents into rich, queryable knowledge.`,
                    code: `# Simple keyword query over an index (Python)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

client = SearchClient(
    "https://<search>.search.windows.net", "docs",
    AzureKeyCredential("<key>"))

for r in client.search(search_text="deploy online endpoint", top=3):
    print(r["title"], "-", r["@search.score"])`
                },
                {
                    title: "Skillsets and Enrichment Pipelines",
                    content: `A <strong>skillset</strong> applies AI during indexing to enrich raw content.

<strong>Built-in skills:</strong>
• OCR and text extraction
• Entity recognition and key phrases
• Language detection and translation
• Embedding generation (vectorization)

<strong>Custom skills:</strong> Call your own Azure Function for bespoke logic. Enriched fields are mapped into the index so they become searchable.`,
                    code: `# A skillset entry that generates embeddings (JSON concept)
skill = {
    "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
    "resourceUri": "https://<resource>.openai.azure.com",
    "deploymentId": "text-embedding-3-large",
    "inputs": [{"name": "text", "source": "/document/content"}],
    "outputs": [{"name": "embedding", "targetName": "contentVector"}],
}
print(skill["@odata.type"])`
                },
                {
                    title: "Vector Search for RAG",
                    content: `<strong>Vector search</strong> retrieves content by semantic meaning, not just keywords — the retrieval step behind RAG.

<strong>How it works:</strong>
• Content is embedded into vectors at indexing time
• A query is embedded and compared by nearest-neighbor
• <strong>Hybrid search</strong> combines vectors + keywords + semantic ranking for best results

<strong>Exam tip:</strong> Vector/hybrid search in Azure AI Search is the recommended grounding source for generative apps.`,
                    code: `# Vector search over an index for RAG grounding (Python)
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential

client = SearchClient(
    "https://<search>.search.windows.net", "docs",
    AzureKeyCredential("<key>"))

results = client.search(
    search_text=None,
    vector_queries=[VectorizableTextQuery(
        text="how do agents call tools", k_nearest_neighbors=3,
        fields="contentVector")])

for r in results:
    print(r["title"], "-", r["@search.score"])`
                }
            ]
        },
        {
            number: "AI-103 · Module 6",
            title: "Speech, Translation & Content Understanding",
            description: "Round out AI-103 breadth: build speech, translation, and audio/video content-understanding solutions with Azure AI.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Speech-to-text & text-to-speech",
                "Speech translation",
                "Azure AI Translator (custom)",
                "Audio/video content understanding",
                "Choosing the right modality service"
            ],
            detailedDescription: "AI-103 requires implementing speech, translation, and content-understanding features. This module covers Azure AI Speech, Translator with custom models, and analyzing audio and video content in production solutions.",
            detailedContent: [
                {
                    title: "Speech-to-Text & Text-to-Speech",
                    content: `<strong>Azure AI Speech</strong> adds voice to applications.

<strong>Speech-to-text:</strong> real-time and batch transcription, custom speech models for domain terms.
<strong>Text-to-speech:</strong> natural neural voices, SSML control, and custom neural voice (gated).

<strong>Exam tip:</strong> use batch transcription for large volumes of recorded audio and real-time for live scenarios.`,
                    code: `# Synthesize speech from text (Azure AI Speech SDK)
import azure.cognitiveservices.speech as speechsdk

cfg = speechsdk.SpeechConfig(subscription="<key>", region="<region>")
cfg.speech_synthesis_voice_name = "en-US-JennyNeural"
synth = speechsdk.SpeechSynthesizer(speech_config=cfg)
synth.speak_text_async("Welcome to the AI-103 course!").get()`
                },
                {
                    title: "Speech Translation",
                    content: `<strong>Speech translation</strong> converts spoken input in one language to text or speech in another.

<strong>Use cases:</strong>
• Live multilingual meetings and captions
• Voice assistants for global users

<strong>Exam tip:</strong> speech translation combines recognition and translation in one pipeline - distinct from text-only Translator.`
                },
                {
                    title: "Azure AI Translator (Custom)",
                    content: `<strong>Azure AI Translator</strong> provides text translation across 100+ languages.

<strong>Capabilities:</strong>
• Auto language detection and transliteration
• <strong>Custom Translator</strong> for domain-specific terminology
• Document translation preserving format

<strong>Exam tip:</strong> use Custom Translator when generic translation misses industry or brand terms.`
                },
                {
                    title: "Audio/Video Content Understanding",
                    content: `Beyond transcription, Azure extracts insights from media.

<strong>Options:</strong>
• <strong>Azure AI Video Indexer</strong> - transcripts, faces, topics, and scenes from recorded video
• Content understanding pipelines that combine speech, vision, and language

<strong>Exam tip:</strong> Video Indexer is the answer for extracting rich insights from recorded media at scale.`
                },
                {
                    title: "Choosing the Right Modality Service",
                    content: `Match the media task to the right service.

<strong>Quick map:</strong>
• Transcribe audio -> Azure AI Speech (speech-to-text)
• Generate speech -> Azure AI Speech (text-to-speech)
• Translate text -> Azure AI Translator
• Translate speech -> Speech translation
• Insights from video -> Azure AI Video Indexer

<strong>Rule:</strong> choose the prebuilt modality service that matches the input and required output.`
                }
            ]
        },
        {
            number: "AI-103 · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Foundry/SDK labs plus a domain-mapped study plan and exam-day strategy for AI-103.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: RAG with Azure AI Search",
                "Lab: Build an agent (Foundry Agent Service)",
                "Lab: Document Intelligence extraction",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AI-103 skills into practice with guided labs using Microsoft Foundry and the Azure AI SDKs, then map your study to the official exam skills and apply an exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: RAG with Azure AI Search",
                    content: `<strong>Goal:</strong> ground a model on your own data.

<strong>Steps:</strong>
1. Create an Azure AI Search index and load a few documents.
2. Add vector fields and an embedding skill.
3. Retrieve top matches for a query and inject them into a prompt.
4. Compare grounded vs ungrounded answers.

<strong>Takeaway:</strong> vector/hybrid search in Azure AI Search is the retrieval engine for enterprise RAG.`,
                    code: `# Retrieve grounding context from Azure AI Search
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

client = SearchClient("https://<search>.search.windows.net", "docs",
                      AzureKeyCredential("<key>"))
ctx = "\\n".join(d["content"] for d in client.search("deploy a model", top=3))
print(ctx[:80], "...")`
                },
                {
                    title: "Lab: Build an Agent (Foundry Agent Service)",
                    content: `<strong>Goal:</strong> create an agent that uses tools.

<strong>Steps:</strong>
1. In Microsoft Foundry, create an agent with instructions.
2. Add a tool (file search / function calling).
3. Create a thread, post a message, and run the agent.
4. Inspect how it plans and calls tools.

<strong>Takeaway:</strong> the Foundry Agent Service manages threads, tool calls, and state so you focus on behavior.`
                },
                {
                    title: "Lab: Document Intelligence Extraction",
                    content: `<strong>Goal:</strong> extract structured data from documents.

<strong>Steps:</strong>
1. Use a prebuilt invoice or layout model.
2. Analyze a sample document.
3. Read fields, tables, and confidence scores.
4. Consider when a custom model is needed.

<strong>Takeaway:</strong> Document Intelligence turns unstructured documents into structured data for downstream apps.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official AI-103 skills.

<strong>Plan & manage Azure AI:</strong> Module 1 (services, provisioning, auth, monitoring, cost).
<strong>Generative & agentic solutions:</strong> Module 2 + Module 7 labs (RAG, tool calling, agents, evaluation).
<strong>Computer vision solutions:</strong> Module 3.
<strong>NLP / text solutions:</strong> Module 4 + Module 6 (language, speech, translation).
<strong>Information extraction & knowledge mining:</strong> Module 5 + Module 7 (Document Intelligence, Azure AI Search).

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-3 + RAG lab
• Week 3: Modules 4-6 + agent and Document Intelligence labs
• Week 4: Module 7 + Microsoft Learn practice assessment

<strong>Pair with:</strong> the Microsoft Learn AI-102/AI-103 learning paths and hands-on Azure practice.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AI-103 is a developer exam - expect SDK, config, and scenario questions.

<strong>During the exam:</strong>
• Match the scenario to the right Azure AI service and SDK
• For grounding, think Azure AI Search (vector/hybrid) + RAG
• For agents, think Foundry Agent Service (threads, tools, runs)
• For production auth, prefer Entra ID and managed identities
• Watch for monitoring, throttling (429 + retries), and cost cues

<strong>Mindset:</strong> you are the AI developer building, securing, and operating a solution.`
                }
            ]
        }
    ],

    // ==========================================================
    // AB-731: Microsoft Certified AI Transformation Leader
    // ==========================================================
    azureAiTransformationLeader: [
        {
            number: "AB-731 · Module 1",
            title: "AI Transformation Strategy & Vision",
            description: "Define enterprise AI vision, align stakeholders, and prioritize initiatives tied to business outcomes.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Business outcomes-first AI strategy",
                "Current-state and readiness assessment",
                "Prioritization frameworks",
                "Executive communication"
            ],
            detailedDescription: "AB-731 starts with strategic clarity. This module helps leaders identify where AI creates measurable value, assess organizational readiness, and build a transformation roadmap that aligns business, technology, and operating constraints.",
            detailedContent: [
                {
                    title: "Start with Business Outcomes",
                    content: `Successful AI programs begin with measurable business outcomes, not tools.

<strong>Focus areas:</strong>
• Revenue growth (new products, improved conversion)
• Cost efficiency (automation, optimization)
• Risk reduction (compliance, fraud, reliability)
• Customer experience (faster service, personalization)

<strong>Leadership principle:</strong> Choose a small number of high-value outcomes and use them to guide all AI investment decisions.`
                },
                {
                    title: "Assess Readiness Across People, Process, and Platform",
                    content: `Readiness determines transformation speed.

<strong>People:</strong> Skills, sponsorship, and change capacity
<strong>Process:</strong> Governance, decision rights, and delivery model
<strong>Platform:</strong> Data quality, security, integration, and tooling

<strong>Exam tip:</strong> AB-731 emphasizes that weak governance or poor data quality can block AI value even when model quality is high.`
                },
                {
                    title: "Prioritize an AI Portfolio",
                    content: `Use an explicit framework to choose what to do first.

<strong>Common rubric:</strong>
• Business value (impact)
• Feasibility (data + technical effort)
• Time-to-value (delivery speed)
• Risk and compliance complexity

Prioritize quick wins that build confidence while funding longer-term strategic capabilities.`,
                    code: `# Example prioritization scorecard (concept)
initiatives = [
    {"name": "Contact center copilot", "value": 5, "feasibility": 4, "speed": 4, "risk": 2},
    {"name": "Contract intelligence", "value": 4, "feasibility": 3, "speed": 3, "risk": 3},
    {"name": "Predictive maintenance", "value": 5, "feasibility": 2, "speed": 2, "risk": 3},
]

for item in initiatives:
    item["score"] = item["value"] + item["feasibility"] + item["speed"] - item["risk"]

for row in sorted(initiatives, key=lambda x: x["score"], reverse=True):
    print(row["name"], "->", row["score"])`
                },
                {
                    title: "Communicate Strategy to Executives",
                    content: `Executive alignment requires a clear narrative:

1. Why change now?
2. What measurable outcomes are targeted?
3. Which capabilities must be built?
4. What risks are controlled and how?
5. What milestones prove progress?

A concise transformation story improves funding decisions and cross-functional alignment.`
                }
            ]
        },
        {
            number: "AB-731 · Module 2",
            title: "Responsible AI, Governance, and Risk",
            description: "Establish responsible AI controls, policy guardrails, and governance structures for safe scale.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Responsible AI principles",
                "Model and data governance",
                "Risk management and controls",
                "Policy, compliance, and auditability"
            ],
            detailedDescription: "AI transformation fails without trust. This module covers governance operating models, risk controls, and responsible AI practices that make enterprise deployment safe, compliant, and auditable.",
            detailedContent: [
                {
                    title: "Design Governance for Scale",
                    content: `Governance should accelerate safe delivery, not slow everything down.

<strong>Key elements:</strong>
• Clear decision rights (who approves what)
• Tiered controls by use-case risk
• Required documentation and review gates
• Escalation paths for high-impact issues`
                },
                {
                    title: "Build a Practical Risk Framework",
                    content: `AI risk spans technical, legal, operational, and reputational domains.

<strong>Typical risk categories:</strong>
• Fairness and bias
• Privacy and data leakage
• Hallucinations and reliability
• Security and prompt injection
• Regulatory non-compliance

Controls should map directly to these risk categories with clear ownership.`
                },
                {
                    title: "Implement Guardrails and Monitoring",
                    content: `Guardrails are only effective when continuously monitored.

<strong>Guardrail examples:</strong>
• Input/output safety filters
• Human-in-the-loop approvals for high-risk actions
• Grounding and citation requirements
• Logging and audit trails for sensitive workflows`,
                    code: `# Governance control checklist (concept)
controls = {
    "safety_filters": True,
    "pii_redaction": True,
    "human_approval_for_high_risk": True,
    "audit_logging": True,
    "fallback_path": True,
}

for name, enabled in controls.items():
    print(f"{name:30} {'OK' if enabled else 'MISSING'}")`
                },
                {
                    title: "Operationalize Responsible AI",
                    content: `Responsible AI must be embedded into delivery lifecycles.

<strong>Operating pattern:</strong>
• Assess risk before deployment
• Validate against policy and quality thresholds
• Monitor outcomes after release
• Trigger incident and remediation workflows when thresholds are exceeded`
                }
            ]
        },
        {
            number: "AB-731 · Module 3",
            title: "Operating Model, Architecture, and Delivery",
            description: "Choose an enterprise AI operating model and delivery approach that balances speed, control, and reuse.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Centralized vs federated AI teams",
                "Reference architecture and platform services",
                "Use-case factory model",
                "Value realization cadence"
            ],
            detailedDescription: "This module helps leaders design how AI initiatives are delivered across the enterprise: team topology, platform standards, reuse patterns, and release governance for reliable scale.",
            detailedContent: [
                {
                    title: "Select the Right Operating Model",
                    content: `There is no one-size-fits-all AI organization.

<strong>Common models:</strong>
• <strong>Centralized:</strong> Strong standards, slower domain responsiveness
• <strong>Federated:</strong> Domain speed, requires strong platform governance
• <strong>Hybrid:</strong> Shared platform + domain delivery squads

Most enterprises converge on hybrid as transformation matures.`
                },
                {
                    title: "Standardize Through a Reference Architecture",
                    content: `A reference architecture improves consistency, security, and speed.

<strong>Include standards for:</strong>
• Identity and access
• Data access and grounding patterns
• Prompt/model lifecycle and evaluation
• Observability, incident management, and rollback`
                },
                {
                    title: "Run a Use-Case Factory",
                    content: `A use-case factory repeatedly takes ideas from intake to production.

<strong>Factory stages:</strong>
1. Intake and qualification
2. Discovery and feasibility
3. MVP build and evaluation
4. Production hardening
5. Scale and optimization`,
                    code: `# Throughput dashboard example (concept)
pipeline = {
    "intake": 27,
    "discovery": 14,
    "mvp": 8,
    "production": 3,
}

for stage, count in pipeline.items():
    print(f"{stage:12}: {count}")`
                },
                {
                    title: "Measure Value Realization",
                    content: `Delivery velocity is not enough; leaders must show realized value.

<strong>Track both:</strong>
• Leading indicators (adoption, cycle time, quality)
• Lagging indicators (cost savings, revenue, risk reduction)

Use quarterly business reviews to recalibrate the portfolio.`
                }
            ]
        },
        {
            number: "AB-731 · Module 4",
            title: "Adoption, Change Management, and AI Culture",
            description: "Drive adoption with capability building, communication, and durable change-management practices.",
            duration: "40 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Stakeholder mapping and change strategy",
                "Workforce enablement and role evolution",
                "Adoption metrics and feedback loops",
                "Scaling an AI-first culture"
            ],
            detailedDescription: "Transformation is sustained by people, not only platforms. This module focuses on change management, upskilling, and adoption leadership so AI capabilities become part of day-to-day operations.",
            detailedContent: [
                {
                    title: "Build a Change Strategy",
                    content: `Adoption increases when users understand why change matters and how it helps them.

<strong>Core actions:</strong>
• Segment stakeholders by impact
• Define role-specific value messages
• Plan communications, training, and support`
                },
                {
                    title: "Enable the Workforce",
                    content: `Capability-building should align to job families.

<strong>Example pathways:</strong>
• Leaders: governance, portfolio steering, value tracking
• Builders: solution patterns, quality, safety, operations
• End users: prompt literacy, policy boundaries, escalation paths`
                },
                {
                    title: "Track Adoption and Outcomes",
                    content: `Adoption metrics reveal whether transformation is real.

<strong>Measure:</strong>
• Active usage by persona
• Task success and time saved
• User satisfaction and trust
• Escalation and override rates`,
                    code: `# Sample adoption KPI snapshot (concept)
kpis = {
    "weekly_active_users": 420,
    "task_time_reduction_pct": 31,
    "user_satisfaction": 4.3,
    "high_risk_escalation_rate_pct": 2.1,
}

for k, v in kpis.items():
    print(f"{k:30}: {v}")`
                },
                {
                    title: "Create a Continuous Improvement Loop",
                    content: `Mature AI organizations continuously learn from usage.

<strong>Loop:</strong>
1. Collect feedback and telemetry
2. Improve policy, prompts, workflows, and training
3. Re-measure outcomes
4. Scale successful patterns across teams`
                }
            ]
        },
        {
            number: "AB-731 · Module 5",
            title: "AI Value Cases, Metrics & Financial Governance",
            description: "Quantify AI value, define metrics and KPIs, and govern AI spend and ROI as a transformation leader.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Building an AI value case",
                "Outcome metrics & KPIs",
                "Cost & FinOps for AI",
                "ROI and portfolio decisions",
                "Communicating value to executives"
            ],
            detailedDescription: "AI Transformation Leaders must prove value, not just deliver capability. This module covers building value cases, defining metrics, governing AI spend, and communicating ROI to executives.",
            detailedContent: [
                {
                    title: "Building an AI Value Case",
                    content: `Every initiative needs a defensible value case.

<strong>Include:</strong>
• Problem and target outcome
• Baseline and expected improvement
• Costs (build, run, change management)
• Risks and mitigations

<strong>Leadership principle:</strong> tie every AI initiative to a measurable business outcome before funding it.`
                },
                {
                    title: "Outcome Metrics & KPIs",
                    content: `Define how success is measured up front.

<strong>Metric types:</strong>
• Leading indicators: adoption, cycle time, quality
• Lagging indicators: revenue, cost savings, risk reduction
• Guardrail metrics: safety, satisfaction, error rates

<strong>Exam tip:</strong> a strong leader pairs value metrics with guardrail metrics so speed does not erode trust.`
                },
                {
                    title: "Cost & FinOps for AI",
                    content: `AI cost is an operating discipline, not an afterthought.

<strong>Cost drivers:</strong>
• Token/compute usage and model choice
• Data storage, retrieval, and pipelines
• Human oversight and support

<strong>FinOps practices:</strong> tagging and showback, budgets and alerts, right-sizing, caching, and reserved/provisioned capacity where steady.`
                },
                {
                    title: "ROI and Portfolio Decisions",
                    content: `Leaders manage a portfolio, not a single project.

<strong>Decisions:</strong>
• Fund, scale, pivot, or stop based on evidence
• Balance quick wins with strategic bets
• Reallocate from low-ROI to high-ROI initiatives

<strong>Cadence:</strong> review the portfolio quarterly against value and guardrail metrics.`
                },
                {
                    title: "Communicating Value to Executives",
                    content: `Value must be communicated to sustain investment.

<strong>Executive narrative:</strong>
1. Outcome achieved vs target
2. Cost and ROI
3. Risks controlled
4. What to fund next and why

<strong>Exam tip:</strong> concise, outcome-first communication drives continued sponsorship and funding.`
                }
            ]
        },
        {
            number: "AB-731 · Module 6",
            title: "Playbooks & Exam Readiness",
            description: "Practical transformation playbooks plus a study plan and exam-day strategy for the AI Transformation Leader assessment.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Playbook: 90-day transformation start",
                "Playbook: Governance operating model",
                "Playbook: Adoption & change plan",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AB-731 concepts into action with concrete leadership playbooks, then map your study to the assessment areas and apply an exam-day strategy for this strategy-focused credential.",
            detailedContent: [
                {
                    title: "Playbook: 90-Day Transformation Start",
                    content: `A practical first-90-days plan.

<strong>Days 0-30:</strong> align on outcomes, assess readiness, pick 2-3 high-value use cases.
<strong>Days 31-60:</strong> stand up governance, run pilots, define metrics.
<strong>Days 61-90:</strong> evaluate pilots, decide scale/stop, publish a roadmap.

<strong>Outcome:</strong> momentum with evidence, not a big-bang rollout.`
                },
                {
                    title: "Playbook: Governance Operating Model",
                    content: `A right-sized governance model accelerates safe delivery.

<strong>Define:</strong>
• Decision rights (who approves what)
• Risk tiers and required controls by tier
• Review gates and escalation paths
• A center of excellence for standards

<strong>Principle:</strong> governance should enable safe speed, not block progress.`
                },
                {
                    title: "Playbook: Adoption & Change Plan",
                    content: `Adoption determines whether value is realized.

<strong>Plan elements:</strong>
• Stakeholder mapping by impact
• Role-specific value messaging
• Training and enablement by job family
• Feedback loops and champions

<strong>Principle:</strong> transformation is sustained by people, not only platforms.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the AI Transformation Leader assessment areas.

<strong>Strategy & vision:</strong> Module 1.
<strong>Responsible AI, governance & risk:</strong> Module 2 + Module 6 playbook.
<strong>Operating model & delivery:</strong> Module 3.
<strong>Adoption & culture:</strong> Module 4 + Module 6 playbook.
<strong>Value, metrics & FinOps:</strong> Module 5.

<strong>Recommended plan:</strong>
• Week 1: Modules 1-2
• Week 2: Modules 3-4 + playbooks
• Week 3: Modules 5-6 + practice assessment
• Week 4: Review + scenario practice

<strong>Pair with:</strong> Microsoft's official AI Transformation Leader practice assessment.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AB-731 is strategy- and leadership-focused.

<strong>During the exam:</strong>
• Read for the business outcome and constraint first
• Choose options that tie to measurable value and responsible AI
• Prefer pilot-then-scale over big-bang rollouts
• Favor governance that enables safe speed
• Eliminate options that ignore adoption, risk, or metrics

<strong>Mindset:</strong> you are the leader steering responsible, valuable AI transformation - not an engineer.`
                }
            ]
        }
    ],

    // ==========================================================
    // DP-100: Azure Data Scientist Associate
    // ==========================================================
    azureDataScientist: [
        {
            number: "DP-100 · Module 1",
            title: "Design & Prepare an ML Solution",
            description: "Set up an Azure Machine Learning workspace, compute, and data assets for data science workloads.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Azure ML workspace & resources",
                "Compute targets",
                "Datastores and data assets",
                "Environments and dependencies",
                "Access control and quotas"
            ],
            detailedDescription: "Every machine learning project starts with the right environment. This module covers provisioning an Azure Machine Learning workspace, choosing compute targets, registering data assets, and defining reusable environments — the foundation of the DP-100 exam.",
            detailedContent: [
                {
                    title: "Azure ML Workspace & Resources",
                    content: `The <strong>Azure Machine Learning workspace</strong> is the top-level resource that ties together compute, data, models, jobs, and endpoints.

<strong>Associated resources:</strong>
• <strong>Storage account</strong> — data and artifacts
• <strong>Key Vault</strong> — secrets and keys
• <strong>Application Insights</strong> — monitoring
• <strong>Container Registry</strong> — environment images

<strong>Ways to work:</strong> Azure ML Studio (web), the Python SDK v2, and the CLI v2 all target the same workspace.`,
                    code: `# Connect to a workspace (Azure ML SDK v2)
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg>",
    workspace_name="<workspace>",
)
print("Connected to:", ml_client.workspace_name)`
                },
                {
                    title: "Compute Targets",
                    content: `<strong>Compute targets</strong> are where code runs. Choosing the right one balances cost and speed.

<strong>Types:</strong>
• <strong>Compute instance:</strong> A personal cloud workstation for notebooks
• <strong>Compute cluster:</strong> Multi-node, auto-scaling compute for training jobs
• <strong>Kubernetes / inference:</strong> Hosts deployed models
• <strong>Serverless:</strong> On-demand compute with no cluster to manage

<strong>Tip:</strong> Clusters scale to zero when idle, so you only pay while jobs run.`,
                    code: `# Create an auto-scaling compute cluster (scales to 0 when idle)
from azure.ai.ml.entities import AmlCompute

cluster = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
)
ml_client.compute.begin_create_or_update(cluster).result()`
                },
                {
                    title: "Datastores and Data Assets",
                    content: `<strong>Datastores</strong> securely reference Azure storage without embedding credentials in code. <strong>Data assets</strong> are named, versioned pointers to data.

<strong>Data asset types:</strong>
• <strong>uri_file:</strong> A single file
• <strong>uri_folder:</strong> A folder of files
• <strong>mltable:</strong> Tabular data with a schema definition

<strong>Why version data:</strong> Reproducibility — a job trained on <em>diabetes-data:1</em> can always be re-run with the exact same input.`,
                    code: `# Register a versioned data asset (Azure ML SDK v2)
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data_asset = Data(
    name="diabetes-data",
    version="1",
    type=AssetTypes.URI_FILE,
    path="azureml://datastores/blob/paths/diabetes.csv",
)
ml_client.data.create_or_update(data_asset)`
                },
                {
                    title: "Environments and Dependencies",
                    content: `An <strong>environment</strong> captures the software (Python version, packages, Docker image) a job needs, so runs are reproducible.

<strong>Options:</strong>
• <strong>Curated environments:</strong> Prebuilt by Microsoft for common frameworks
• <strong>Custom environments:</strong> Defined from a conda file or Dockerfile

Environments are versioned and reusable across jobs and deployments — the same environment used to train can be used to deploy.`,
                    code: `# Define a custom environment from a conda spec
from azure.ai.ml.entities import Environment

env = Environment(
    name="sklearn-env",
    version="1",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="conda.yml",   # lists python + scikit-learn, mlflow, etc.
)
ml_client.environments.create_or_update(env)`
                },
                {
                    title: "Access Control and Quotas",
                    content: `Governing who can do what — and staying within limits — keeps solutions secure and predictable.

<strong>Access control (RBAC):</strong>
• Assign built-in roles (Reader, Contributor, AzureML Data Scientist)
• Apply least privilege at workspace or resource scope
• Use managed identities for keyless service-to-service auth

<strong>Quotas:</strong>
• Compute cores are capped per region and VM family
• Monitor usage and request increases before large training runs

<strong>Exam tip:</strong> Know that quota limits can block cluster scale-out.`
                }
            ]
        },
        {
            number: "DP-100 · Module 2",
            title: "Explore Data & Run Experiments",
            description: "Profile data, run training experiments, and track results with MLflow in Azure Machine Learning.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Exploratory data analysis",
                "Running jobs as experiments",
                "Tracking with MLflow",
                "Logging metrics and artifacts",
                "Comparing runs"
            ],
            detailedDescription: "Experimentation is the core of data science. This module covers exploring data, running training scripts as Azure ML jobs, and using MLflow to track metrics, parameters, and artifacts so experiments are reproducible and comparable.",
            detailedContent: [
                {
                    title: "Exploratory Data Analysis",
                    content: `<strong>Exploratory data analysis (EDA)</strong> builds understanding before modeling.

<strong>What to check:</strong>
• Shape, data types, and missing values
• Distributions and outliers
• Correlations between features and the target
• Class balance for classification

<strong>Where:</strong> Run EDA in notebooks on a compute instance, using pandas, matplotlib, and seaborn. Good EDA guides feature engineering and model choice.`,
                    code: `# Quick EDA with pandas
import pandas as pd

df = pd.read_csv("diabetes.csv")
print(df.shape)
print(df.describe())
print("Missing values:\\n", df.isna().sum())
print("Target balance:\\n", df["Diabetic"].value_counts(normalize=True))`
                },
                {
                    title: "Running Jobs as Experiments",
                    content: `Training scripts run on cloud compute as <strong>jobs</strong>, grouped under <strong>experiments</strong>.

<strong>A command job specifies:</strong>
• The script and its arguments
• The input data
• The environment
• The compute target

<strong>Benefit:</strong> Jobs run remotely, scale beyond your laptop, and are fully tracked and reproducible.`,
                    code: `# Submit a training script as a command job (Azure ML SDK v2)
from azure.ai.ml import command, Input

job = command(
    code="./src",
    command="python train.py --data \${{inputs.training_data}}",
    inputs={"training_data": Input(type="uri_file", path="azureml:diabetes-data:1")},
    environment="sklearn-env:1",
    compute="cpu-cluster",
    experiment_name="diabetes-training",
)
returned_job = ml_client.jobs.create_or_update(job)
print("Submitted:", returned_job.name)`
                },
                {
                    title: "Tracking with MLflow",
                    content: `Azure Machine Learning uses <strong>MLflow</strong> as its native tracking framework.

<strong>What you can track:</strong>
• <strong>Parameters:</strong> Hyperparameters and configuration
• <strong>Metrics:</strong> Accuracy, loss, AUC over time
• <strong>Artifacts:</strong> Models, plots, and files
• <strong>Models:</strong> Logged in a standard format for deployment

<strong>Autologging</strong> captures much of this automatically for popular frameworks.`,
                    code: `# Enable MLflow autologging in a training script
import mlflow
from sklearn.ensemble import RandomForestClassifier

mlflow.autolog()   # auto-captures params, metrics & the model

model = RandomForestClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)   # metrics logged automatically`
                },
                {
                    title: "Logging Metrics and Artifacts",
                    content: `Beyond autologging, you can log custom values explicitly.

<strong>Common calls:</strong>
• <strong>log_metric</strong> — a single scored value (accuracy, RMSE)
• <strong>log_param</strong> — a configuration value
• <strong>log_artifact</strong> — a file such as a confusion-matrix image

These appear on the run in Azure ML Studio, making results explainable and auditable.`,
                    code: `# Log custom metrics and an artifact
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("auc", auc)
mlflow.log_artifact("confusion_matrix.png")
print("Logged:", acc, auc)`
                },
                {
                    title: "Comparing Runs",
                    content: `Every job is recorded, so you can <strong>compare runs</strong> to pick the best model.

<strong>In Azure ML Studio:</strong>
• Select multiple runs to chart metrics side by side
• Sort experiments by a primary metric
• Inspect parameters that led to the best score

<strong>Programmatically:</strong> Query runs with MLflow to automate model selection and promotion.`,
                    code: `# Find the best run in an experiment by a metric (MLflow)
import mlflow

runs = mlflow.search_runs(
    experiment_names=["diabetes-training"],
    order_by=["metrics.accuracy DESC"],
)
best = runs.iloc[0]
print("Best run:", best["run_id"], "accuracy:", best["metrics.accuracy"])`
                }
            ]
        },
        {
            number: "DP-100 · Module 3",
            title: "Train & Deploy Models",
            description: "Train models at scale, tune hyperparameters, and deploy to managed online and batch endpoints.",
            duration: "60 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Command and sweep jobs",
                "Hyperparameter tuning with sweeps",
                "Registering models",
                "Managed online endpoints",
                "Batch endpoints",
                "Testing deployments"
            ],
            detailedDescription: "This module covers taking a model from training to production: running command jobs, tuning hyperparameters with sweep jobs, registering the best model, and deploying it to managed online or batch endpoints for real-time and large-scale scoring.",
            detailedContent: [
                {
                    title: "Command and Sweep Jobs",
                    content: `Two core job types drive training in Azure ML.

<strong>Command job:</strong>
Runs a single script with fixed inputs — the workhorse for training.

<strong>Sweep job:</strong>
Wraps a command job to run it many times across a <em>search space</em> of hyperparameters, tracking each trial.

<strong>Exam tip:</strong> A sweep job is how you do hyperparameter tuning natively in Azure ML.`,
                    code: `# A command job that accepts a tunable argument
from azure.ai.ml import command

job = command(
    code="./src",
    command="python train.py --reg_rate \${{inputs.reg_rate}}",
    inputs={"reg_rate": 0.1},
    environment="sklearn-env:1",
    compute="cpu-cluster",
)`
                },
                {
                    title: "Hyperparameter Tuning with Sweeps",
                    content: `A <strong>sweep</strong> explores hyperparameter combinations to maximize a metric.

<strong>Key choices:</strong>
• <strong>Search space:</strong> ranges/choices per parameter
• <strong>Sampling:</strong> grid, random, or Bayesian
• <strong>Primary metric:</strong> what to optimize (e.g., accuracy)
• <strong>Early termination:</strong> stop poor trials early (bandit, median)

Sweeps run trials in parallel on a cluster and surface the best configuration.`,
                    code: `# Turn a command job into a hyperparameter sweep
from azure.ai.ml.sweep import Choice, Uniform

sweep = job.sweep(
    compute="cpu-cluster",
    sampling_algorithm="random",
    primary_metric="accuracy",
    goal="Maximize",
)
sweep.search_space = {"reg_rate": Uniform(0.01, 1.0)}
sweep.set_limits(max_total_trials=20, max_concurrent_trials=4)
ml_client.jobs.create_or_update(sweep)`
                },
                {
                    title: "Registering Models",
                    content: `<strong>Registering</strong> a model stores it as a named, versioned asset in the workspace — the bridge from training to deployment.

<strong>Benefits:</strong>
• Version history and lineage back to the training run
• Reusable across deployments
• Supports MLflow model format for no-code deployment

<strong>Tip:</strong> Register the model produced by your best sweep trial.`,
                    code: `# Register a model from a completed job (Azure ML SDK v2)
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path="azureml://jobs/<job-name>/outputs/artifacts/paths/model/",
    name="diabetes-model",
    type=AssetTypes.MLFLOW_MODEL,
)
ml_client.models.create_or_update(model)`
                },
                {
                    title: "Managed Online Endpoints",
                    content: `A <strong>managed online endpoint</strong> hosts a registered model behind an HTTPS API for real-time scoring, with Azure handling the infrastructure.

<strong>Deployment steps:</strong>
1. Register the trained model
2. Create an endpoint (the stable URL)
3. Create a deployment (model + environment + compute)
4. Route traffic to the deployment

<strong>Blue/green:</strong> Deploy a new version alongside the old and shift traffic gradually for safe rollouts.`,
                    code: `# Deploy a registered model to an online endpoint (Azure ML SDK v2)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, ManagedOnlineDeployment)

endpoint = ManagedOnlineEndpoint(name="diabetes-endpoint")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="diabetes-endpoint",
    model="azureml:diabetes-model:1",
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()`
                },
                {
                    title: "Batch Endpoints",
                    content: `A <strong>batch endpoint</strong> scores large volumes of data asynchronously — ideal when you don't need instant responses.

<strong>When to use:</strong>
• Scoring millions of records on a schedule
• Long-running inference over files in storage
• Cost efficiency (compute scales up only for the job)

<strong>Online vs. batch:</strong> online = low-latency, always-on; batch = high-throughput, on-demand.`,
                    code: `# Invoke a batch endpoint over a data asset (Azure ML SDK v2)
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

job = ml_client.batch_endpoints.invoke(
    endpoint_name="diabetes-batch",
    input=Input(type=AssetTypes.URI_FOLDER, path="azureml:new-patients:1"),
)
print("Batch scoring job:", job.name)`
                },
                {
                    title: "Testing Deployments",
                    content: `Before sending real traffic, verify a deployment returns correct predictions.

<strong>How to test:</strong>
• Invoke the endpoint with a sample JSON payload
• Check the response schema and values
• Review logs for errors or cold-start issues

<strong>Tip:</strong> Keep a small held-out sample as a smoke test you can run after every deployment.`,
                    code: `# Test an online deployment with a sample request
result = ml_client.online_endpoints.invoke(
    endpoint_name="diabetes-endpoint",
    deployment_name="blue",
    request_file="sample-request.json",
)
print("Prediction:", result)`
                }
            ]
        },
        {
            number: "DP-100 · Module 4",
            title: "Optimize Language Models for AI Apps",
            description: "Fine-tune and ground language models using Azure Machine Learning and Azure AI for generative applications.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Foundation models in Azure ML",
                "Prompt engineering & prompt flow",
                "Fine-tuning language models",
                "Grounding with RAG",
                "Evaluating model quality"
            ],
            detailedDescription: "Modern data science increasingly involves language models. This module — a newer focus of DP-100 — covers using foundation models from the Azure ML catalog, building prompt flows, fine-tuning, grounding with RAG, and evaluating generative outputs.",
            detailedContent: [
                {
                    title: "Foundation Models in Azure ML",
                    content: `The <strong>model catalog</strong> in Azure ML and Foundry offers ready-to-use <strong>foundation models</strong>.

<strong>What you'll find:</strong>
• Open models (Llama, Mistral, Phi) and Azure OpenAI models
• Task filters (chat, embeddings, vision)
• Benchmarks and model cards to compare quality and cost

<strong>Deploy options:</strong> Managed compute or serverless API — pick based on control vs. simplicity.`,
                    code: `# List foundation models in the catalog (concept)
for model in ml_client.models.list():
    if model.name.startswith("Phi") or model.name.startswith("Llama"):
        print(model.name, model.version)`
                },
                {
                    title: "Prompt Engineering & Prompt Flow",
                    content: `<strong>Prompt flow</strong> in Azure Machine Learning orchestrates the steps of an LLM application — from input, through retrieval and prompting, to output — as a testable graph.

<strong>Why it matters:</strong>
• Version and reuse prompts like code
• Chain retrieval, LLM calls, and post-processing
• Run evaluations on the whole flow
• Compare variants before deploying

Prompt flow is the recommended way to build and test generative apps in Azure ML.`,
                    code: `# A prompt flow node calling a model (concept, YAML-like)
node = {
    "name": "answer",
    "type": "llm",
    "inputs": {
        "deployment_name": "gpt-4o",
        "temperature": 0.2,
        "prompt": "Answer using context: {{context}}\\nQ: {{question}}",
    },
}
print(node["name"])`
                },
                {
                    title: "Fine-Tuning Language Models",
                    content: `<strong>Fine-tuning</strong> adapts a base model to your domain using example input/output pairs — useful when prompting and RAG aren't enough.

<strong>When to fine-tune:</strong>
• Consistent style, format, or tone is required
• A specialized task with many labeled examples
• Reducing prompt length/cost for a repeated task

<strong>Trade-offs:</strong> Fine-tuning needs quality data and adds training cost; often RAG is cheaper and easier to keep current.`,
                    code: `# Submit a fine-tuning job (concept, Azure OpenAI)
client.fine_tuning.jobs.create(
    training_file="file-abc123",          # uploaded JSONL of examples
    model="gpt-4o-mini-2024-07-18",
)
# Monitor status, then deploy the resulting fine-tuned model`
                },
                {
                    title: "Grounding with RAG",
                    content: `<strong>Retrieval Augmented Generation (RAG)</strong> grounds a model on your data, improving accuracy without retraining.

<strong>The flow:</strong>
1. Index content in Azure AI Search (with embeddings)
2. Retrieve relevant passages for a question
3. Add them to the prompt as context
4. Generate an answer grounded in the passages

<strong>Exam tip:</strong> RAG is usually preferred over fine-tuning for keeping answers current and citeable.`,
                    code: `# Retrieve context to ground a prompt
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search = SearchClient(
    "https://<search>.search.windows.net", "docs",
    AzureKeyCredential("<key>"))

context = "\\n".join(d["content"] for d in search.search("model deployment", top=3))
prompt = f"Use this context to answer:\\n{context}"
print(prompt[:80], "...")`
                },
                {
                    title: "Evaluating Model Quality",
                    content: `Generative outputs must be measured, not eyeballed.

<strong>Evaluation dimensions:</strong>
• <strong>Groundedness:</strong> Is the answer supported by the context?
• <strong>Relevance:</strong> Does it address the question?
• <strong>Coherence & fluency:</strong> Is it well-formed?
• <strong>Safety:</strong> Is it free of harmful content?

Use the <strong>Azure AI Evaluation SDK</strong> to score responses at scale and compare model or prompt variants.`,
                    code: `# Score groundedness of generated answers (Azure AI Evaluation)
from azure.ai.evaluation import GroundednessEvaluator

groundedness = GroundednessEvaluator(model_config)
score = groundedness(
    query="What optimizer trains logistic regression?",
    context="Logistic regression is trained with gradient descent.",
    response="It is trained using gradient descent.",
)
print("Groundedness score:", score)`
                }
            ]
        },
        {
            number: "DP-100 · Module 5",
            title: "MLOps & Pipelines",
            description: "Automate the machine learning lifecycle with pipelines, CI/CD, and model monitoring.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Building Azure ML pipelines",
                "Reusable components",
                "Scheduling and triggers",
                "CI/CD with GitHub Actions / Azure DevOps",
                "Monitoring for data drift"
            ],
            detailedDescription: "MLOps brings engineering discipline to machine learning. This module covers composing reusable pipeline components, automating retraining with CI/CD, and monitoring deployed models for data drift so solutions stay reliable in production.",
            detailedContent: [
                {
                    title: "Building Azure ML Pipelines",
                    content: `An <strong>Azure ML pipeline</strong> chains steps — data prep, training, evaluation, registration — into a repeatable, schedulable workflow.

<strong>Benefits:</strong>
• Reproducibility and versioning
• Parallel, reusable steps
• Clear lineage from data to model
• A foundation for automated retraining

Pipelines are defined in the SDK or CLI and run on cluster compute.`,
                    code: `# Compose a two-step pipeline from components (Azure ML SDK v2)
from azure.ai.ml import dsl, Input

@dsl.pipeline(compute="cpu-cluster", description="Train diabetes model")
def training_pipeline(pipeline_data):
    prep = prep_component(input_data=pipeline_data)
    train = train_component(training_data=prep.outputs.output_data)
    return {"model_output": train.outputs.model_output}

pipeline_job = training_pipeline(
    pipeline_data=Input(type="uri_file", path="azureml:diabetes-data:1"))
ml_client.jobs.create_or_update(pipeline_job, experiment_name="diabetes-mlops")`
                },
                {
                    title: "Reusable Components",
                    content: `A <strong>component</strong> is a self-contained, versioned step (code + inputs + outputs + environment) that can be shared across pipelines.

<strong>Why components:</strong>
• Write a step once, reuse everywhere
• Version and test steps independently
• Share via the workspace registry

Think of components as functions for your ML pipelines.`,
                    code: `# Define a reusable component (Azure ML SDK v2)
from azure.ai.ml import command
from azure.ai.ml import Input, Output

train_component = command(
    name="train_model",
    version="1",
    inputs={"training_data": Input(type="uri_folder")},
    outputs={"model_output": Output(type="uri_folder")},
    code="./train_src",
    command="python train.py --data \${{inputs.training_data}} "
            "--out \${{outputs.model_output}}",
    environment="sklearn-env:1",
)`
                },
                {
                    title: "Scheduling and Triggers",
                    content: `Pipelines can run automatically instead of on demand.

<strong>Trigger types:</strong>
• <strong>Schedule:</strong> Cron or recurrence (e.g., nightly retrain)
• <strong>Event-based:</strong> Kick off when new data arrives (via Event Grid)

<strong>Use case:</strong> Retrain a model every week on the latest data, then register the new version automatically.`,
                    code: `# Schedule a pipeline to run daily (Azure ML SDK v2)
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger

trigger = RecurrenceTrigger(frequency="day", interval=1)
schedule = JobSchedule(
    name="daily-retrain",
    trigger=trigger,
    create_job=pipeline_job,
)
ml_client.schedules.begin_create_or_update(schedule).result()`
                },
                {
                    title: "CI/CD with GitHub Actions / Azure DevOps",
                    content: `<strong>CI/CD</strong> automates testing and deployment of ML code and models.

<strong>Typical stages:</strong>
• <strong>CI:</strong> Lint, unit test, and validate on every commit
• <strong>CD:</strong> Submit training, register the model, deploy to an endpoint

<strong>Tools:</strong> GitHub Actions or Azure DevOps pipelines call the Azure ML CLI v2, gated by approvals for production.`,
                    code: `# GitHub Actions step that triggers an Azure ML job (YAML)
# .github/workflows/train.yml
# - name: Run training pipeline
#   run: |
#     az ml job create --file pipeline.yml \\
#       --resource-group my-rg --workspace-name my-ws
echo "CI/CD invokes: az ml job create --file pipeline.yml"`
                },
                {
                    title: "Monitoring for Data Drift",
                    content: `After deployment, models can silently degrade as live data diverges from training data — called <strong>data drift</strong>.

<strong>What to monitor:</strong>
• Feature distribution changes (drift)
• Prediction distribution shifts
• Data quality issues (nulls, out-of-range)

<strong>Action:</strong> Set alerts on drift metrics and trigger a retraining pipeline when thresholds are crossed — closing the MLOps loop.`,
                    code: `# Enable model monitoring for drift (concept, Azure ML SDK v2)
from azure.ai.ml.entities import (
    MonitorSchedule, RecurrenceTrigger, MonitorDefinition)

monitor = MonitorSchedule(
    name="diabetes-drift-monitor",
    trigger=RecurrenceTrigger(frequency="week", interval=1),
    create_monitor=MonitorDefinition(compute="cpu-cluster"),
)
ml_client.schedules.begin_create_or_update(monitor).result()`
                }
            ]
        },
        {
            number: "DP-100 · Module 6",
            title: "Responsible ML, Fairness & Interpretability",
            description: "Apply the Responsible AI dashboard, Fairlearn, interpretability, and error analysis to build fair, explainable models on Azure ML.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Responsible AI dashboard",
                "Fairness with Fairlearn",
                "Model interpretability",
                "Error analysis",
                "Data drift & monitoring"
            ],
            detailedDescription: "Responsible ML is part of the data scientist's job. This module covers the Azure Machine Learning Responsible AI dashboard and its components - fairness (Fairlearn), interpretability, error analysis - plus monitoring deployed models for drift.",
            detailedContent: [
                {
                    title: "Responsible AI Dashboard",
                    content: `The <strong>Responsible AI dashboard</strong> in Azure ML brings model debugging tools into one view.

<strong>Components:</strong>
• Error analysis
• Model interpretability (feature importance)
• Fairness assessment (Fairlearn)
• Data explorer and counterfactuals

<strong>Exam tip:</strong> use the Responsible AI dashboard to diagnose where and why a model underperforms across groups.`
                },
                {
                    title: "Fairness with Fairlearn",
                    content: `<strong>Fairlearn</strong> measures and mitigates unfairness across sensitive groups.

<strong>Workflow:</strong>
• Choose sensitive features (e.g., age group)
• Compare metrics disaggregated by group
• Apply mitigation (e.g., reweighting) and re-measure

<strong>Exam tip:</strong> fairness is about disaggregated metrics, not overall accuracy alone.`
                },
                {
                    title: "Model Interpretability",
                    content: `<strong>Interpretability</strong> explains why a model makes predictions.

<strong>Approaches:</strong>
• Global importance (which features matter overall)
• Local explanations (why this prediction)

<strong>On Azure ML:</strong> the interpret tooling produces feature importances for transparency and debugging - key for trust and compliance.`,
                    code: `# Feature importance for transparency (Azure ML interpret)
from azureml.interpret import ExplanationClient

client = ExplanationClient.from_run(run)
explanation = client.download_model_explanation()
for feature, importance in explanation.get_feature_importance_dict().items():
    print(f"{feature:20} {importance:.3f}")`
                },
                {
                    title: "Error Analysis",
                    content: `<strong>Error analysis</strong> finds where a model fails, not just how often.

<strong>Value:</strong>
• Identify cohorts with high error rates
• Reveal data gaps or bias hotspots
• Prioritize fixes where they matter most

<strong>Exam tip:</strong> a model with good overall accuracy can still fail badly for a specific subgroup - error analysis surfaces that.`
                },
                {
                    title: "Data Drift & Monitoring",
                    content: `Deployed models degrade as live data diverges from training data.

<strong>Monitor for:</strong>
• Feature distribution changes (data drift)
• Prediction drift
• Data quality issues

<strong>Action:</strong> alert on drift and trigger retraining - closing the MLOps loop. On Azure ML, model monitoring automates this.`
                }
            ]
        },
        {
            number: "DP-100 · Module 7",
            title: "Hands-On Labs & Exam Readiness (Archived - see AI-300)",
            description: "Guided Azure ML labs and a study plan. Note: DP-100 retired June 2026; new learners should prepare for AI-300 instead.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Retirement note & AI-300 successor",
                "Lab: Train & register a model",
                "Lab: Deploy an online endpoint",
                "Lab: Pipeline & drift monitoring",
                "Exam-day strategy"
            ],
            detailedDescription: "DP-100 retired in June 2026 and is archived here for reference. New learners should prepare for AI-300 (ML Operations Engineer). These labs remain useful for Azure ML fundamentals that carry over to AI-300.",
            detailedContent: [
                {
                    title: "Retirement Note & AI-300 Successor",
                    content: `<strong>Important:</strong> DP-100 (Azure Data Scientist Associate) <strong>retired in June 2026</strong>.

<strong>What to do instead:</strong>
• New learners should prepare for <strong>AI-300 (ML Operations Engineer Associate)</strong>, the successor.
• AI-300 covers modern MLOps and GenAIOps on Azure.
• The Azure ML fundamentals in this track (workspaces, jobs, MLflow, endpoints, pipelines) still apply to AI-300.

Use this archived content to build core Azure ML skills, then switch to the AI-300 track for current exam prep.`
                },
                {
                    title: "Lab: Train & Register a Model",
                    content: `<strong>Goal:</strong> run a training job and register the model.

<strong>Steps:</strong>
1. Create an Azure ML workspace and a compute cluster.
2. Submit a training script as a command job.
3. Track metrics with MLflow.
4. Register the best model in the workspace.

<strong>Takeaway:</strong> jobs + MLflow + the model registry are the core Azure ML training flow.`,
                    code: `# Submit a training job (Azure ML SDK v2)
from azure.ai.ml import command, Input

job = command(
    code="./src",
    command="python train.py --data \${{inputs.data}}",
    inputs={"data": Input(type="uri_file", path="azureml:diabetes-data:1")},
    environment="sklearn-env:1", compute="cpu-cluster",
    experiment_name="diabetes-training")
ml_client.jobs.create_or_update(job)`
                },
                {
                    title: "Lab: Deploy an Online Endpoint",
                    content: `<strong>Goal:</strong> serve a model for real-time scoring.

<strong>Steps:</strong>
1. Register a trained model.
2. Create a managed online endpoint.
3. Create a deployment and route traffic.
4. Send a test request.

<strong>Takeaway:</strong> managed online endpoints host models behind HTTPS with blue/green rollout support.`
                },
                {
                    title: "Lab: Pipeline & Drift Monitoring",
                    content: `<strong>Goal:</strong> automate and observe.

<strong>Steps:</strong>
1. Compose a training pipeline from components.
2. Schedule it to run on a recurrence.
3. Enable data drift monitoring on the endpoint.
4. Review drift alerts.

<strong>Takeaway:</strong> pipelines + drift monitoring are the foundation of MLOps that carries into AI-300.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `If you must sit an archived DP-100 exam, or are transferring skills to AI-300:

<strong>Focus areas:</strong>
• Workspace, compute, data assets, environments
• Jobs, sweeps (hyperparameter tuning), MLflow tracking
• Online vs batch endpoints and safe rollout
• Pipelines, components, and drift monitoring
• Responsible ML (fairness, interpretability)

<strong>Recommendation:</strong> prioritize the <strong>AI-300</strong> track for current, exam-relevant preparation.`
                }
            ]
        }
    ],

    // ==========================================================
    // AI-300: Machine Learning Operations Engineer Associate
    // ==========================================================
    azureMlOps: [
        {
            number: "AI-300 · Module 1",
            title: "Design & Implement MLOps Infrastructure",
            description: "Provision and automate the Azure infrastructure for machine learning operations with IaC.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Azure ML workspace & resources",
                "Compute & environments",
                "Infrastructure as code (Bicep & CLI)",
                "Access, security & networking"
            ],
            detailedDescription: "AI-300 (the successor to DP-100) starts with MLOps infrastructure. This module covers provisioning an Azure Machine Learning workspace and its resources, defining compute and reusable environments, automating everything with Bicep and the Azure CLI, and securing the platform.",
            detailedContent: [
                {
                    title: "Azure ML Workspace & Resources",
                    content: `The <strong>Azure Machine Learning workspace</strong> is the control plane for MLOps, tying together compute, data, models, jobs, and endpoints.

<strong>Associated resources:</strong>
• Storage account (data & artifacts)
• Key Vault (secrets)
• Application Insights (monitoring)
• Container Registry (environment images)

<strong>MLOps focus:</strong> These resources should be provisioned repeatably per environment (dev / test / prod) rather than by hand.`
                },
                {
                    title: "Compute & Environments",
                    content: `Reproducible compute and software are the foundation of reliable operations.

<strong>Compute:</strong>
• <strong>Compute clusters</strong> (auto-scaling) for training
• <strong>Managed endpoints</strong> for serving
• <strong>Serverless</strong> compute for on-demand jobs

<strong>Environments:</strong> Versioned definitions (conda file or Docker image) reused across training and deployment to eliminate "works on my machine" drift.`,
                    code: `# Create an auto-scaling compute cluster (Azure ML SDK v2)
from azure.ai.ml.entities import AmlCompute

cluster = AmlCompute(
    name="cpu-cluster", type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0, max_instances=4,
    idle_time_before_scale_down=120,
)
ml_client.compute.begin_create_or_update(cluster).result()`
                },
                {
                    title: "Infrastructure as Code (Bicep & Azure CLI)",
                    content: `AI-300 emphasizes provisioning MLOps infrastructure with <strong>infrastructure as code (IaC)</strong>.

<strong>Why IaC:</strong>
• Repeatable, reviewable environments
• Consistent dev / test / prod parity
• Safe rollbacks and audit history

<strong>Tools:</strong> <strong>Bicep</strong> templates deployed via the <strong>Azure CLI</strong>, typically triggered from CI/CD (GitHub Actions).`,
                    code: `// Bicep: provision an Azure ML workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: mlWorkspaceName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storageAccountId
    keyVault: keyVaultId
    applicationInsights: appInsightsId
  }
}

// Deploy: az deployment group create --resource-group my-rg --template-file main.bicep`
                },
                {
                    title: "Access, Security & Networking",
                    content: `Securing the MLOps platform is a core responsibility.

<strong>Controls:</strong>
• <strong>Managed identities</strong> and RBAC (least privilege)
• <strong>Private endpoints / VNet</strong> isolation
• <strong>Key Vault</strong> for secrets and keys
• Customer-managed keys for encryption

<strong>Exam tip:</strong> Prefer managed identities over keys, and isolate workspaces on a virtual network for sensitive workloads.`
                }
            ]
        },
        {
            number: "AI-300 · Module 2",
            title: "ML Model Lifecycle & Operations",
            description: "Automate training, registration, deployment, CI/CD, and monitoring of traditional ML models.",
            duration: "60 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Training pipelines",
                "Model registry & versioning",
                "Deployment (online & batch)",
                "CI/CD with GitHub Actions",
                "Monitoring & data drift"
            ],
            detailedDescription: "This module covers operationalizing the classic ML lifecycle: building repeatable training pipelines, managing model versions, deploying to endpoints, automating with GitHub Actions, and monitoring models for drift in production.",
            detailedContent: [
                {
                    title: "Training Pipelines",
                    content: `<strong>Azure ML pipelines</strong> chain data prep, training, evaluation, and registration into a repeatable, schedulable workflow built from reusable components.

<strong>Benefits for MLOps:</strong>
• Reproducibility and lineage
• Step caching and parallelism
• Scheduled or event-triggered retraining`,
                    code: `# Compose a two-step training pipeline (Azure ML SDK v2)
from azure.ai.ml import dsl, Input

@dsl.pipeline(compute="cpu-cluster", description="Train & register")
def training_pipeline(pipeline_data):
    prep = prep_component(input_data=pipeline_data)
    train = train_component(training_data=prep.outputs.output_data)
    return {"model_output": train.outputs.model_output}

job = training_pipeline(Input(type="uri_file", path="azureml:data:1"))
ml_client.jobs.create_or_update(job, experiment_name="mlops")`
                },
                {
                    title: "Model Registry & Versioning",
                    content: `The <strong>model registry</strong> stores models as named, versioned assets with lineage back to the training run.

<strong>Governance:</strong>
• Version history and reproducibility
• Stage/approval workflows before production
• Reuse across deployments

<strong>Tip:</strong> Register the best model from a pipeline run and promote it via approvals rather than deploying ad hoc.`,
                    code: `# Register a model produced by a job
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path="azureml://jobs/<job-name>/outputs/artifacts/paths/model/",
    name="churn-model", type=AssetTypes.MLFLOW_MODEL)
ml_client.models.create_or_update(model)`
                },
                {
                    title: "Deployment (Online & Batch)",
                    content: `Serve models with the option that matches the workload.

• <strong>Managed online endpoints:</strong> Low-latency, real-time scoring
• <strong>Batch endpoints:</strong> Asynchronous scoring of large datasets

<strong>Safe rollouts:</strong> Use blue/green deployments and traffic splitting to release new versions without downtime.`,
                    code: `# Deploy to a managed online endpoint with traffic split
from azure.ai.ml.entities import ManagedOnlineDeployment

deployment = ManagedOnlineDeployment(
    name="blue", endpoint_name="churn-endpoint",
    model="azureml:churn-model:1",
    instance_type="Standard_DS3_v2", instance_count=1)
ml_client.online_deployments.begin_create_or_update(deployment).result()`
                },
                {
                    title: "CI/CD with GitHub Actions",
                    content: `AI-300 expects hands-on <strong>CI/CD</strong> for ML using <strong>GitHub Actions</strong>.

<strong>Typical workflow:</strong>
1. Lint and unit test on every commit (CI)
2. Run the training pipeline
3. Register and evaluate the model
4. Deploy to staging, then production (with approvals)

The Azure ML CLI v2 (<code>az ml</code>) is called from the workflow steps.`,
                    code: `# .github/workflows/train-deploy.yml (excerpt)
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/login@v2
        with: { creds: \${{ secrets.AZURE_CREDENTIALS }} }
      - name: Run training pipeline
        run: az ml job create --file pipeline.yml -g my-rg -w my-ws`
                },
                {
                    title: "Monitoring & Data Drift",
                    content: `Deployed models degrade as live data diverges from training data.

<strong>Azure ML Model Monitoring</strong> detects:
• Data drift and prediction drift
• Data quality issues
• Feature attribution drift

<strong>Closing the loop:</strong> Alerts (via Azure Monitor) trigger a retraining pipeline, keeping models accurate — the essence of MLOps.`
                }
            ]
        },
        {
            number: "AI-300 · Module 3",
            title: "Design & Implement GenAIOps Infrastructure",
            description: "Operationalize generative AI apps and agents with Microsoft Foundry.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Microsoft Foundry projects & connections",
                "Deploying foundation models",
                "Prompt flow orchestration",
                "Grounding with RAG"
            ],
            detailedDescription: "GenAIOps is what sets AI-300 apart from DP-100. This module covers building generative AI infrastructure with Microsoft Foundry: projects and connections, deploying foundation models, orchestrating with prompt flow, and grounding responses with retrieval.",
            detailedContent: [
                {
                    title: "Microsoft Foundry Projects & Connections",
                    content: `<strong>Microsoft Foundry</strong> is the platform for building, evaluating, and operating generative AI solutions.

<strong>Building blocks:</strong>
• <strong>Projects</strong> organize resources, data, and access
• <strong>Connections</strong> link to models, Azure AI Search, and storage
• <strong>Model catalog</strong> for foundation models

<strong>GenAIOps focus:</strong> Provision projects and connections as code so environments are reproducible.`,
                    code: `# Connect to a Microsoft Foundry project (Python)
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project = AIProjectClient(
    endpoint="https://<project>.services.ai.azure.com/api/projects/<name>",
    credential=DefaultAzureCredential())
print("Connected to project")`
                },
                {
                    title: "Deploying Foundation Models",
                    content: `Deploy foundation models so applications can call them through a stable endpoint.

<strong>Deployment options:</strong>
• Serverless API deployments (pay-per-token)
• Managed compute deployments (dedicated capacity)

<strong>Operational concerns:</strong> versioning deployments, capacity/quota (TPM/PTU), and content filters — all managed per environment.`,
                    code: `# Chat completion against a Foundry model deployment
client = project.inference.get_azure_openai_client(api_version="2024-10-21")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize MLOps in one line."}])
print(resp.choices[0].message.content)`
                },
                {
                    title: "Prompt Flow Orchestration",
                    content: `<strong>Prompt flow</strong> orchestrates the steps of a generative AI application — input, retrieval, prompting, and post-processing — as a testable, versioned graph.

<strong>Why it matters for GenAIOps:</strong>
• Treat prompts and flows like code (version, review, test)
• Run batch evaluations on the whole flow
• Deploy flows as endpoints with CI/CD`
                },
                {
                    title: "Grounding with RAG",
                    content: `<strong>Retrieval Augmented Generation (RAG)</strong> grounds models on your data so answers are accurate and current.

<strong>On Azure:</strong>
• Index content in <strong>Azure AI Search</strong> (with vector search)
• Retrieve relevant chunks at query time
• Add them to the prompt as context

<strong>GenAIOps:</strong> Automate index build/refresh and connect the search resource to the Foundry project.`,
                    code: `# Retrieve grounding context from Azure AI Search
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search = SearchClient("https://<search>.search.windows.net", "docs",
                      AzureKeyCredential("<key>"))
context = "\\n".join(d["content"] for d in search.search("deploy model", top=3))
print(context[:80], "...")`
                }
            ]
        },
        {
            number: "AI-300 · Module 4",
            title: "Generative AI Quality Assurance & Observability",
            description: "Evaluate, trace, and safeguard generative AI applications in production.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Evaluating generative AI",
                "Groundedness & safety evaluations",
                "Tracing & observability",
                "Content safety & guardrails"
            ],
            detailedDescription: "Operating generative AI responsibly requires measurement and observability. This module covers evaluating generative outputs, running groundedness and safety evaluations, tracing requests end to end, and enforcing guardrails.",
            detailedContent: [
                {
                    title: "Evaluating Generative AI",
                    content: `Generative quality must be measured, not assumed.

<strong>Quality metrics:</strong>
• Relevance, coherence, fluency
• Similarity to a reference (when available)

<strong>On Azure:</strong> The <strong>Azure AI Evaluation SDK</strong> scores responses at scale, and Foundry supports batch evaluation runs to compare prompts, models, or flow versions.`,
                    code: `# Score relevance of generated answers (Azure AI Evaluation)
from azure.ai.evaluation import RelevanceEvaluator

relevance = RelevanceEvaluator(model_config)
score = relevance(
    query="How do I deploy a model?",
    response="Create an endpoint, then a deployment, then route traffic.")
print("Relevance:", score)`
                },
                {
                    title: "Groundedness & Safety Evaluations",
                    content: `Two risk-focused evaluation areas are central to GenAIOps.

<strong>Groundedness:</strong> Is the answer supported by the retrieved context (not hallucinated)?

<strong>Safety:</strong> Does output avoid harmful content (hate, violence, self-harm, sexual) and resist jailbreaks?

<strong>Practice:</strong> Run these evaluators on representative datasets before release and continuously in production sampling.`,
                    code: `# Evaluate groundedness against retrieved context
from azure.ai.evaluation import GroundednessEvaluator

groundedness = GroundednessEvaluator(model_config)
score = groundedness(
    query="What trains logistic regression?",
    context="Logistic regression is trained with gradient descent.",
    response="It is trained using gradient descent.")
print("Groundedness:", score)`
                },
                {
                    title: "Tracing & Observability",
                    content: `<strong>Tracing</strong> captures each step of a generative request — retrieval, prompts, tool calls, and responses — for debugging and monitoring.

<strong>On Azure:</strong>
• Foundry tracing (OpenTelemetry-based)
• <strong>Application Insights</strong> for metrics, latency, and token usage
• Dashboards and alerts in Azure Monitor

<strong>Why it matters:</strong> Observability makes generative systems debuggable and auditable in production.`
                },
                {
                    title: "Content Safety & Guardrails",
                    content: `<strong>Azure AI Content Safety</strong> filters harmful content across categories (hate, sexual, violence, self-harm) with severity levels.

<strong>Guardrails for generative AI:</strong>
• Input and output content filters
• Prompt-shield against jailbreaks
• Groundedness detection for hallucinations

<strong>GenAIOps:</strong> Apply guardrails consistently across deployments and monitor safety events.`,
                    code: `# Screen text with Azure AI Content Safety
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential

client = ContentSafetyClient(
    "https://<resource>.cognitiveservices.azure.com/",
    AzureKeyCredential("<key>"))
result = client.analyze_text(AnalyzeTextOptions(text="You are welcome here!"))
for c in result.categories_analysis:
    print(c.category, "severity:", c.severity)`
                }
            ]
        },
        {
            number: "AI-300 · Module 5",
            title: "Optimize Generative AI Systems & Model Performance",
            description: "Tune RAG, models, cost, and latency, and drive continuous improvement for AIOps.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Optimizing RAG pipelines",
                "Fine-tuning & model selection",
                "Cost & latency optimization",
                "Continuous improvement (AIOps)"
            ],
            detailedDescription: "The final AI-300 domain focuses on optimization. This module covers improving RAG quality, choosing and fine-tuning models, reducing cost and latency, and building a continuous-improvement loop across the AI operations lifecycle.",
            detailedContent: [
                {
                    title: "Optimizing RAG Pipelines",
                    content: `RAG quality depends on retrieval quality.

<strong>Levers:</strong>
• Chunking strategy and overlap
• Embedding model choice
• Hybrid (vector + keyword) search and semantic ranking
• Top-k and re-ranking

<strong>Practice:</strong> Evaluate retrieval and end-to-end answer quality, then tune these knobs iteratively.`
                },
                {
                    title: "Fine-tuning & Model Selection",
                    content: `Choose the right model, and customize only when needed.

<strong>Selection:</strong> Balance quality, latency, and cost across the model catalog (GPT, Phi, Llama, etc.).

<strong>Customization:</strong>
• Prefer prompt engineering and RAG first
• <strong>Fine-tune</strong> for consistent style or a narrow task with quality data
• Consider <strong>distillation</strong> to a smaller, cheaper model

<strong>Exam tip:</strong> Justify customization choices by cost and maintenance, not just accuracy.`
                },
                {
                    title: "Cost & Latency Optimization",
                    content: `Operating generative AI economically is a key skill.

<strong>Cost levers:</strong>
• Right-size model and throughput (TPM/PTU)
• Cache frequent responses and embeddings
• Trim prompts and context length

<strong>Latency levers:</strong>
• Streaming responses
• Parallel retrieval
• Smaller/faster models for simple tasks

<strong>Monitor:</strong> token usage and latency via Application Insights.`
                },
                {
                    title: "Continuous Improvement (AIOps)",
                    content: `AI-300 unifies MLOps and GenAIOps into <strong>AI operations (AIOps)</strong> — a continuous loop.

<strong>The loop:</strong>
1. Monitor quality, drift, cost, and safety
2. Collect feedback and failures
3. Update data, prompts, or models
4. Re-evaluate and redeploy via CI/CD

<strong>Outcome:</strong> Reliable, safe, and cost-effective AI systems that improve over time.`
                }
            ]
        },
        {
            number: "AI-300 · Module 6",
            title: "Security, Networking & Governance for MLOps",
            description: "Secure the MLOps platform end to end: managed identities, private networking, secrets, encryption, and governance across environments.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Managed identities & RBAC",
                "Private endpoints & VNet isolation",
                "Secrets with Key Vault",
                "Encryption & CMEK",
                "Governance across environments"
            ],
            detailedDescription: "AI-300 expects you to secure and govern the ML platform. This module covers identity and access, network isolation, secret management, encryption, and multi-environment governance for production MLOps and GenAIOps.",
            detailedContent: [
                {
                    title: "Managed Identities & RBAC",
                    content: `Prefer keyless, least-privilege access.

<strong>Practices:</strong>
• Use managed identities for service-to-service auth
• Apply built-in roles (Reader, Contributor, AzureML Data Scientist)
• Scope access at workspace or resource level

<strong>Exam tip:</strong> managed identities + RBAC beat storing keys, and least privilege is the default answer.`
                },
                {
                    title: "Private Endpoints & VNet Isolation",
                    content: `Sensitive workloads should not traverse the public internet.

<strong>Controls:</strong>
• Private endpoints for the workspace and dependent resources
• VNet integration for compute
• Restrict public network access

<strong>Exam tip:</strong> private endpoints + VNet isolation is the standard answer for network-sensitive ML platforms.`
                },
                {
                    title: "Secrets with Key Vault",
                    content: `Never hardcode secrets.

<strong>Practices:</strong>
• Store keys, connection strings, and certs in Azure Key Vault
• Access them at runtime via managed identity
• Rotate and audit secrets

<strong>Exam tip:</strong> reference Key Vault secrets rather than embedding them in code or config.`
                },
                {
                    title: "Encryption & CMEK",
                    content: `Protect data at rest and in transit.

<strong>Options:</strong>
• Platform-managed keys by default
• <strong>Customer-managed keys (CMEK)</strong> for regulatory control
• Encryption in transit via TLS

<strong>Exam tip:</strong> choose CMEK when compliance requires customer control over encryption keys.`
                },
                {
                    title: "Governance Across Environments",
                    content: `Production MLOps spans dev, test, and prod.

<strong>Governance:</strong>
• Infrastructure as code (Bicep) for consistent environments
• Policy controls and audit logging
• Approvals in CI/CD before production

<strong>Exam tip:</strong> reproducible, governed environments via IaC and CI/CD approvals are recurring AI-300 themes.`,
                    code: `// Bicep: provision an Azure ML workspace per environment
resource ml 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: mlName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storageId
    keyVault: keyVaultId
    applicationInsights: appInsightsId
  }
}`
                }
            ]
        },
        {
            number: "AI-300 · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided MLOps/GenAIOps labs plus a domain-mapped study plan and exam-day strategy for AI-300.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: IaC workspace with Bicep",
                "Lab: CI/CD with GitHub Actions",
                "Lab: GenAIOps with Foundry",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AI-300 skills into practice with guided labs for infrastructure as code, CI/CD, and GenAIOps, then map your study to the exam domains and apply an exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: IaC Workspace with Bicep",
                    content: `<strong>Goal:</strong> provision MLOps infrastructure as code.

<strong>Steps:</strong>
1. Write a Bicep template for an Azure ML workspace and dependencies.
2. Deploy it with the Azure CLI to a dev resource group.
3. Re-deploy to a test environment to prove repeatability.

<strong>Takeaway:</strong> IaC gives consistent, reviewable dev/test/prod environments.`,
                    code: `# Deploy the Bicep template (Azure CLI)
az deployment group create \\
  --resource-group my-rg \\
  --template-file main.bicep \\
  --parameters mlName=my-ws location=eastus`
                },
                {
                    title: "Lab: CI/CD with GitHub Actions",
                    content: `<strong>Goal:</strong> automate training and deployment.

<strong>Steps:</strong>
1. Add a workflow that authenticates to Azure.
2. Run an Azure ML job on push.
3. Register and deploy the model with approvals for production.

<strong>Takeaway:</strong> GitHub Actions calling the Azure ML CLI is the backbone of AI-300 MLOps.`,
                    code: `# .github/workflows/train.yml (excerpt)
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/login@v2
        with: { creds: \${{ secrets.AZURE_CREDENTIALS }} }
      - run: az ml job create --file pipeline.yml -g my-rg -w my-ws`
                },
                {
                    title: "Lab: GenAIOps with Foundry",
                    content: `<strong>Goal:</strong> operationalize a generative solution.

<strong>Steps:</strong>
1. Create a Microsoft Foundry project and connections.
2. Deploy a foundation model.
3. Build a prompt flow and run batch evaluations.
4. Add content safety and tracing.

<strong>Takeaway:</strong> GenAIOps = provisioning, prompt flow, evaluation, safety, and observability for generative AI.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official AI-300 domains.

<strong>MLOps infrastructure (IaC):</strong> Module 1 + Module 6 + IaC lab.
<strong>ML model lifecycle & operations:</strong> Module 2 + CI/CD lab.
<strong>GenAIOps infrastructure:</strong> Module 3 + Foundry lab.
<strong>GenAI QA & observability:</strong> Module 4.
<strong>Optimization & AIOps:</strong> Module 5 + Module 7.

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-3 + IaC and CI/CD labs
• Week 3: Modules 4-6 + GenAIOps lab
• Week 4: Module 7 + Microsoft Learn practice assessment

<strong>Pair with:</strong> the Microsoft Learn AI-300 learning path and hands-on Azure practice.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AI-300 is an operations exam - expect IaC, CI/CD, and monitoring scenarios.

<strong>During the exam:</strong>
• For repeatable environments, think Bicep + CLI (IaC)
• For automation, think GitHub Actions + Azure ML CLI
• For generative solutions, think Foundry + prompt flow + evaluations
• For security, think managed identities, private endpoints, Key Vault, CMEK
• Watch for drift, retraining, cost, and safety cues

<strong>Mindset:</strong> you are the ML operations engineer building a secure, automated, observable platform.`
                }
            ]
        }
    ],

    // ==========================================================
    // AI-200: Azure AI Cloud Developer Associate
    // ==========================================================
    azureAiCloudDev: [
        {
            number: "AI-200 · Module 1",
            title: "Containerized Compute & Hosting",
            description: "Host AI applications on Azure with Container Apps, AKS, and serverless Azure Functions.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Hosting options for AI apps",
                "Azure Container Apps",
                "Azure Kubernetes Service (AKS)",
                "Serverless APIs with Azure Functions"
            ],
            detailedDescription: "The Azure AI Cloud Developer track begins with hosting. This module covers the compute and containerization patterns for running AI applications on Azure — Container Apps for serverless containers, AKS for orchestration at scale, and Azure Functions for event-driven serverless APIs.",
            detailedContent: [
                {
                    title: "Hosting Options for AI Apps",
                    content: `Azure offers a spectrum of compute for AI-driven backends.

<strong>Choosing a host:</strong>
• <strong>Azure Container Apps:</strong> Serverless containers with autoscaling — great default
• <strong>Azure Kubernetes Service (AKS):</strong> Full orchestration for complex, large-scale workloads
• <strong>Azure Functions:</strong> Event-driven serverless functions/APIs
• <strong>App Service:</strong> Managed web app hosting

<strong>Exam tip:</strong> Match the host to scale, control, and operational overhead — prefer the simplest option that meets the requirements.`
                },
                {
                    title: "Azure Container Apps",
                    content: `<strong>Azure Container Apps</strong> runs containerized apps and microservices serverlessly, with built-in autoscaling (including scale-to-zero) powered by KEDA.

<strong>Key features:</strong>
• Scale on HTTP traffic, events, or CPU/memory
• Revisions for blue/green and traffic splitting
• Dapr integration for microservices
• Managed ingress and HTTPS

<strong>Use case:</strong> Hosting AI APIs and workers without managing Kubernetes.`,
                    code: `# Deploy a container to Azure Container Apps (Azure CLI)
az containerapp create \\
  --name ai-api \\
  --resource-group my-rg \\
  --environment my-aca-env \\
  --image myregistry.azurecr.io/ai-api:latest \\
  --target-port 8000 --ingress external \\
  --min-replicas 0 --max-replicas 10`
                },
                {
                    title: "Azure Kubernetes Service (AKS)",
                    content: `<strong>Azure Kubernetes Service (AKS)</strong> provides managed Kubernetes for complex, large-scale AI workloads that need fine-grained orchestration.

<strong>When to choose AKS:</strong>
• GPU workloads for model serving
• Complex microservice topologies
• Advanced networking and scaling control

<strong>Operations:</strong> Deploy with manifests/Helm, scale with the cluster autoscaler, and monitor with Container Insights.`,
                    code: `# Deploy a model-serving app to AKS (kubectl)
kubectl apply -f - <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata: { name: model-server }
spec:
  replicas: 3
  selector: { matchLabels: { app: model-server } }
  template:
    metadata: { labels: { app: model-server } }
    spec:
      containers:
        - name: server
          image: myregistry.azurecr.io/model-server:latest
          ports: [{ containerPort: 8080 }]
YAML`
                },
                {
                    title: "Serverless APIs with Azure Functions",
                    content: `<strong>Azure Functions</strong> runs event-driven code without managing servers — ideal for lightweight AI APIs and glue logic.

<strong>Triggers & bindings:</strong>
• HTTP triggers for APIs
• Queue/Service Bus/Event Grid triggers for async work
• Input/output bindings to storage and databases

<strong>Use case:</strong> A function that receives a request, calls a model, and returns the result — scaling automatically with load.`,
                    code: `# HTTP-triggered Azure Function calling a model (Python v2)
import azure.functions as func

app = func.FunctionApp()

@app.route(route="score", auth_level=func.AuthLevel.FUNCTION)
def score(req: func.HttpRequest) -> func.HttpResponse:
    text = req.get_json().get("text", "")
    # call your model / Foundry endpoint here
    return func.HttpResponse(f"Scored: {text}")`
                }
            ]
        },
        {
            number: "AI-200 · Module 2",
            title: "Azure Data Services for AI",
            description: "Store and retrieve AI data with Cosmos DB, PostgreSQL + pgvector, and Azure Managed Redis.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Azure Cosmos DB for NoSQL",
                "Azure Database for PostgreSQL & pgvector",
                "Azure Managed Redis (caching & vector search)"
            ],
            detailedDescription: "AI applications need fast, flexible data stores. This module covers the Azure data services that back AI workloads: Cosmos DB for NoSQL documents, PostgreSQL with pgvector for relational + vector data, and Azure Managed Redis for caching and vector search.",
            detailedContent: [
                {
                    title: "Azure Cosmos DB for NoSQL",
                    content: `<strong>Azure Cosmos DB for NoSQL</strong> is a globally distributed, low-latency document database — great for AI app state, chat history, and metadata.

<strong>Highlights:</strong>
• Single-digit-millisecond reads/writes
• Automatic, elastic scaling (RU/s or serverless)
• Native <strong>vector search</strong> for embeddings
• Flexible JSON schema

<strong>Use case:</strong> Store conversations and documents, and query them (including by vector similarity) for RAG.`,
                    code: `# Write and read a document with Cosmos DB (Python)
from azure.cosmos import CosmosClient

client = CosmosClient("https://<acct>.documents.azure.com:443/", "<key>")
container = client.get_database_client("ai").get_container_client("chats")

container.upsert_item({"id": "c1", "user": "u42", "message": "Hello AI"})
item = container.read_item(item="c1", partition_key="c1")
print(item["message"])`
                },
                {
                    title: "Azure Database for PostgreSQL & pgvector",
                    content: `<strong>Azure Database for PostgreSQL</strong> with the <strong>pgvector</strong> extension stores embeddings alongside relational data, enabling vector similarity search in SQL.

<strong>Why it matters:</strong>
• Combine structured data + vectors in one store
• Familiar SQL and transactions
• Powers RAG retrieval directly from the database

<strong>Exam tip:</strong> pgvector adds a <code>vector</code> column type and distance operators for nearest-neighbor search.`,
                    code: `-- Enable pgvector and query by similarity
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE docs (id serial PRIMARY KEY, content text, embedding vector(1536));

-- Find the 3 most similar documents to a query embedding
SELECT id, content
FROM docs
ORDER BY embedding <-> '[0.12, 0.03, ...]'
LIMIT 3;`
                },
                {
                    title: "Azure Managed Redis (Caching & Vector Search)",
                    content: `<strong>Azure Managed Redis</strong> provides in-memory caching, streaming, and vector search for high-performance AI apps.

<strong>Uses in AI solutions:</strong>
• <strong>Caching</strong> model responses and embeddings to cut cost and latency
• <strong>Session state</strong> for conversational apps
• <strong>Vector search</strong> for fast semantic retrieval
• <strong>Streams</strong> for real-time pipelines

<strong>Tip:</strong> Cache frequent prompts/embeddings to dramatically reduce token spend.`,
                    code: `# Cache a model response in Redis (Python)
import redis

r = redis.Redis(host="<name>.redis.cache.windows.net", port=6380,
                password="<key>", ssl=True)

key = "prompt:summarize:doc42"
cached = r.get(key)
if cached is None:
    cached = "<call model and store result>"
    r.setex(key, 3600, cached)   # cache for 1 hour
print(cached)`
                }
            ]
        },
        {
            number: "AI-200 · Module 3",
            title: "Event-Driven Integration",
            description: "Connect AI services with message- and event-based architectures using Service Bus and Event Grid.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Messaging with Azure Service Bus",
                "Events with Azure Event Grid",
                "Orchestrating AI workflows"
            ],
            detailedDescription: "Scalable AI systems decouple components with messaging and events. This module covers Azure Service Bus for reliable messaging, Event Grid for event routing, and how to orchestrate resilient, asynchronous AI workflows.",
            detailedContent: [
                {
                    title: "Messaging with Azure Service Bus",
                    content: `<strong>Azure Service Bus</strong> is an enterprise message broker for reliable, decoupled communication between services.

<strong>Features:</strong>
• Queues (point-to-point) and topics/subscriptions (pub-sub)
• Ordered, at-least-once delivery
• Dead-lettering and retries
• Sessions for related messages

<strong>Use case:</strong> Queue long-running AI jobs (e.g., batch scoring) so workers process them reliably and scale independently.`,
                    code: `# Send a message to a Service Bus queue (Python)
from azure.servicebus import ServiceBusClient, ServiceBusMessage

client = ServiceBusClient.from_connection_string("<conn-str>")
with client.get_queue_sender("ai-jobs") as sender:
    sender.send_messages(ServiceBusMessage('{"doc": "42", "task": "score"}'))
print("Job queued")`
                },
                {
                    title: "Events with Azure Event Grid",
                    content: `<strong>Azure Event Grid</strong> routes events from sources to handlers with a publish-subscribe model — ideal for reactive, event-driven AI pipelines.

<strong>Patterns:</strong>
• React to blob uploads (e.g., new document → trigger ingestion)
• Fan out events to multiple handlers
• Integrate with Functions, Logic Apps, and webhooks

<strong>Service Bus vs. Event Grid:</strong> Service Bus = reliable commands/work queues; Event Grid = lightweight event notifications.`
                },
                {
                    title: "Orchestrating AI Workflows",
                    content: `Combining these services builds resilient, scalable AI workflows.

<strong>Example pipeline:</strong>
1. Document uploaded to Blob Storage
2. <strong>Event Grid</strong> fires an event
3. An <strong>Azure Function</strong> enqueues a job on <strong>Service Bus</strong>
4. A <strong>Container Apps</strong> worker embeds and indexes the document
5. Results stored in Cosmos DB / PostgreSQL

<strong>Benefits:</strong> Loose coupling, independent scaling, and fault tolerance across the AI solution.`
                }
            ]
        },
        {
            number: "AI-200 · Module 4",
            title: "Application Security & Configuration",
            description: "Secure AI applications with Key Vault, App Configuration, and managed identities.",
            duration: "45 min",
            lessons: "2 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Managing secrets with Key Vault",
                "App configuration & managed identities"
            ],
            detailedDescription: "Production AI apps must be secure by design. This module covers managing secrets with Azure Key Vault, centralizing settings with App Configuration, and eliminating credentials in code using managed identities.",
            detailedContent: [
                {
                    title: "Managing Secrets with Key Vault",
                    content: `<strong>Azure Key Vault</strong> securely stores secrets, keys, and certificates — no more API keys in code or config files.

<strong>Best practices:</strong>
• Store model/API keys and connection strings in Key Vault
• Access them at runtime via managed identity (no secrets in code)
• Rotate secrets and audit access

<strong>Exam tip:</strong> Reference Key Vault secrets from Container Apps/Functions rather than hardcoding them.`,
                    code: `# Read a secret from Key Vault with a managed identity (Python)
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

client = SecretClient(
    vault_url="https://<vault>.vault.azure.net/",
    credential=DefaultAzureCredential())

api_key = client.get_secret("openai-api-key").value
print("Loaded secret without storing it in code")`
                },
                {
                    title: "App Configuration & Managed Identities",
                    content: `<strong>Azure App Configuration</strong> centralizes application settings and feature flags across environments and services.

<strong>Managed identities:</strong>
• Give each app an Azure AD identity
• Grant RBAC access to Key Vault, Cosmos DB, storage, etc.
• <strong>No credentials</strong> stored or rotated by you

<strong>Result:</strong> Consistent configuration and keyless, least-privilege access to the AI solution's dependencies.`
                }
            ]
        },
        {
            number: "AI-200 · Module 5",
            title: "Observability & Troubleshooting",
            description: "Monitor, trace, and troubleshoot AI applications with Application Insights and Azure Monitor.",
            duration: "45 min",
            lessons: "2 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Monitoring with Application Insights",
                "Troubleshooting AI applications"
            ],
            detailedDescription: "The final module makes AI solutions observable and debuggable. It covers instrumenting apps with Application Insights and Azure Monitor, and systematically troubleshooting performance, reliability, and dependency issues.",
            detailedContent: [
                {
                    title: "Monitoring with Application Insights",
                    content: `<strong>Application Insights</strong> (part of Azure Monitor) provides telemetry for AI applications: requests, dependencies, exceptions, and custom metrics.

<strong>What to track:</strong>
• Request latency and failure rates
• Dependency calls (models, databases, queues)
• Token usage and cost signals
• Custom events for AI outcomes

<strong>Tools:</strong> Live Metrics, Application Map, and Kusto (KQL) queries with alerts.`,
                    code: `# Instrument a Python app with Azure Monitor OpenTelemetry
from azure.monitor.opentelemetry import configure_azure_monitor
import logging

configure_azure_monitor(connection_string="<app-insights-conn-str>")
logging.getLogger(__name__).info("AI request handled", extra={"tokens": 128})`
                },
                {
                    title: "Troubleshooting AI Applications",
                    content: `A systematic approach resolves production issues quickly.

<strong>Common issues & signals:</strong>
• <strong>Latency spikes:</strong> slow model/DB calls → check dependency telemetry
• <strong>Failures/429s:</strong> throttling → add retries and scale
• <strong>Bad outputs:</strong> data/prompt issues → trace inputs and context
• <strong>Cost spikes:</strong> token usage → review caching and prompt size

<strong>Approach:</strong> Use the Application Map to isolate the failing component, inspect traces, then reproduce and fix. Instrument thoroughly so issues are diagnosable.`
                }
            ]
        },
        {
            number: "AI-200 · Module 6",
            title: "Integrating Azure OpenAI & AI Services into Apps",
            description: "Wire Azure OpenAI, AI services, and RAG into cloud apps with resilient, secure integration patterns.",
            duration: "50 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Calling Azure OpenAI from apps",
                "RAG integration pattern",
                "Resilience: retries & backpressure",
                "Streaming responses",
                "Securing AI calls"
            ],
            detailedDescription: "The AI-200 developer builds AI-powered cloud apps. This module covers integrating Azure OpenAI and AI services into applications with RAG, resilience, streaming, and security patterns.",
            detailedContent: [
                {
                    title: "Calling Azure OpenAI from Apps",
                    content: `Applications call your Azure OpenAI <strong>deployment</strong>, not the raw model.

<strong>Essentials:</strong>
• Use the deployment name and endpoint
• Authenticate with Entra ID (managed identity) in production
• Handle tokens, timeouts, and errors

<strong>Exam tip:</strong> you call the deployment name; secure it with managed identity, not keys, in production.`,
                    code: `# Chat completion from an app (Azure OpenAI)
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<key>", api_version="2024-10-21")
resp = client.chat.completions.create(
    model="gpt-4o",  # deployment name
    messages=[{"role": "user", "content": "Summarize this ticket."}])
print(resp.choices[0].message.content)`
                },
                {
                    title: "RAG Integration Pattern",
                    content: `Ground app responses on your data.

<strong>Flow:</strong>
1. Embed the user query
2. Retrieve top matches from Azure AI Search (or a vector DB like pgvector)
3. Add context to the prompt
4. Call the model and return a grounded answer

<strong>Exam tip:</strong> for AI-200, the vector store may be Azure AI Search, Cosmos DB, PostgreSQL (pgvector), or Redis - choose based on the app's data layer.`
                },
                {
                    title: "Resilience: Retries & Backpressure",
                    content: `Production AI calls must handle failure and load.

<strong>Patterns:</strong>
• Exponential backoff on HTTP 429 (throttling)
• Circuit breakers and timeouts
• Queues (Service Bus) to smooth spikes
• Fallback models or cached responses

<strong>Exam tip:</strong> retries with backoff plus queue-based load leveling keep AI apps reliable under load.`,
                    code: `# Retry on throttling (HTTP 429) with backoff
import time
from openai import RateLimitError

def call_with_retry(fn, **kw):
    for attempt in range(5):
        try:
            return fn(**kw)
        except RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("Exceeded retries")`
                },
                {
                    title: "Streaming Responses",
                    content: `Streaming improves perceived latency for chat UIs.

<strong>How:</strong>
• Request a streamed completion
• Send tokens to the client as they arrive (e.g., via SSE or WebSockets)

<strong>Exam tip:</strong> streaming is the answer when users need fast, incremental responses in a conversational app.`
                },
                {
                    title: "Securing AI Calls",
                    content: `Protect the AI integration end to end.

<strong>Controls:</strong>
• Managed identity + Key Vault for secrets
• Private networking for AI endpoints
• Content Safety on input and output
• Rate limiting and per-user quotas

<strong>Exam tip:</strong> combine identity, network isolation, and content safety for a secure AI app.`
                }
            ]
        },
        {
            number: "AI-200 · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Azure app + AI labs plus a domain-mapped study plan and exam-day strategy for AI-200.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Deploy an AI API to Container Apps",
                "Lab: RAG with PostgreSQL/pgvector",
                "Lab: Event-driven processing",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AI-200 skills into practice with guided labs for containerized hosting, RAG data services, and event-driven integration, then map your study to the exam and apply an exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Deploy an AI API to Container Apps",
                    content: `<strong>Goal:</strong> host an AI-backed API serverlessly.

<strong>Steps:</strong>
1. Containerize a small API that calls Azure OpenAI.
2. Push the image to Azure Container Registry.
3. Deploy to Azure Container Apps with external ingress.
4. Scale rules on HTTP concurrency.

<strong>Takeaway:</strong> Container Apps host AI APIs with autoscaling and no cluster management.`,
                    code: `# Deploy to Azure Container Apps (Azure CLI)
az containerapp create --name ai-api --resource-group my-rg \\
  --environment my-env \\
  --image myregistry.azurecr.io/ai-api:latest \\
  --target-port 8000 --ingress external \\
  --min-replicas 0 --max-replicas 10`
                },
                {
                    title: "Lab: RAG with PostgreSQL/pgvector",
                    content: `<strong>Goal:</strong> store and query embeddings in a relational DB.

<strong>Steps:</strong>
1. Enable the pgvector extension on Azure Database for PostgreSQL.
2. Store document embeddings in a vector column.
3. Query nearest neighbors for a user question.
4. Ground the model on the retrieved rows.

<strong>Takeaway:</strong> pgvector keeps relational data and embeddings together for RAG.`,
                    code: `-- Query nearest neighbors with pgvector
CREATE EXTENSION IF NOT EXISTS vector;
SELECT id, content
FROM docs
ORDER BY embedding <-> '[0.12, 0.03, ...]'
LIMIT 3;`
                },
                {
                    title: "Lab: Event-Driven Processing",
                    content: `<strong>Goal:</strong> build a resilient async AI pipeline.

<strong>Steps:</strong>
1. Upload a document to Blob Storage.
2. Event Grid fires an event to a Function.
3. The Function enqueues a job on Service Bus.
4. A Container Apps worker embeds and indexes the document.

<strong>Takeaway:</strong> events + queues decouple components and absorb spikes.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the AI-200 skill areas.

<strong>Containerized hosting:</strong> Module 1 + Container Apps lab.
<strong>Data services for AI:</strong> Module 2 + pgvector RAG lab.
<strong>Event-driven integration:</strong> Module 3 + event lab.
<strong>Security & configuration:</strong> Module 4 + Module 6.
<strong>Observability & AI integration:</strong> Module 5 + Module 6.

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-3 + Container Apps and RAG labs
• Week 3: Modules 4-6 + event-driven lab
• Week 4: Module 7 + Microsoft Learn practice assessment

<strong>Pair with:</strong> the Microsoft Learn AI-200 learning path and hands-on Azure practice.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AI-200 is a cloud developer exam - expect hosting, data, and integration scenarios.

<strong>During the exam:</strong>
• Match hosting to needs: Container Apps (default), AKS (complex scale), Functions (event-driven)
• For vectors, pick the store that fits the app (AI Search, Cosmos, PostgreSQL, Redis)
• For decoupling, think Event Grid (events) and Service Bus (work queues)
• For secrets, think Key Vault + managed identity
• For reliability, think retries, backpressure, and observability

<strong>Mindset:</strong> you are the cloud developer building, securing, and operating an AI application.`
                }
            ]
        }
    ],

    // ==========================================================
    // AIF-C01: AWS Certified AI Practitioner
    // ==========================================================
    awsAiPractitioner: [
        {
            number: "AIF-C01 · Module 1",
            title: "Fundamentals of AI & ML",
            description: "Understand core AI, machine learning, and deep learning concepts and where they fit in the AWS AI/ML stack.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is AI, ML & deep learning?",
                "Types of machine learning",
                "The ML lifecycle",
                "The AWS AI/ML stack"
            ],
            detailedDescription: "This module introduces the foundational concepts assessed on the AWS Certified AI Practitioner exam: the relationship between AI, ML, and deep learning, the main types of learning, the machine learning lifecycle, and the layers of AWS AI and ML services.",
            detailedContent: [
                {
                    title: "What is AI, ML & Deep Learning?",
                    content: `<strong>Artificial Intelligence (AI)</strong> is the broad field of building systems that perform tasks requiring human-like intelligence.

<strong>The hierarchy:</strong>
• <strong>AI:</strong> Any technique that mimics human intelligence
• <strong>Machine Learning:</strong> Systems that learn patterns from data
• <strong>Deep Learning:</strong> ML using multi-layer neural networks
• <strong>Generative AI:</strong> Deep learning that creates new content

<strong>Exam tip:</strong> Know that generative AI ⊂ deep learning ⊂ machine learning ⊂ AI, and be able to match a scenario to the right level.`
                },
                {
                    title: "Types of Machine Learning",
                    content: `The exam expects you to distinguish the main learning styles.

<strong>Supervised learning:</strong>
Trained on labeled data — classification (categories) and regression (numbers).

<strong>Unsupervised learning:</strong>
Finds structure in unlabeled data — clustering and dimensionality reduction.

<strong>Reinforcement learning:</strong>
An agent learns by trial and error using rewards.

<strong>On AWS:</strong> Amazon SageMaker supports all three; many use cases are also solved with prebuilt AI services (no training required).`,
                    code: `# Match an AWS AI service to a use case
aws_ai_services = {
    "Amazon Rekognition":   "Image & video analysis",
    "Amazon Transcribe":    "Speech-to-text",
    "Amazon Comprehend":    "NLP: sentiment, entities",
    "Amazon Textract":      "Extract text & data from documents",
    "Amazon Personalize":   "Recommendations",
    "Amazon Bedrock":       "Generative AI foundation models",
    "Amazon SageMaker":     "Build/train/deploy custom ML",
}
for service, use_case in aws_ai_services.items():
    print(f"{service:20} -> {use_case}")`
                },
                {
                    title: "The ML Lifecycle",
                    content: `Machine learning solutions follow a repeatable lifecycle:

1. <strong>Business problem framing</strong>
2. <strong>Data collection & preparation</strong>
3. <strong>Feature engineering</strong>
4. <strong>Model training</strong>
5. <strong>Evaluation & tuning</strong>
6. <strong>Deployment</strong>
7. <strong>Monitoring & maintenance</strong>

<strong>On AWS:</strong> Amazon SageMaker provides tools for every stage, from SageMaker Data Wrangler (prep) to SageMaker Model Monitor (monitoring).`
                },
                {
                    title: "The AWS AI/ML Stack",
                    content: `AWS organizes its AI/ML offerings into three layers:

<strong>1. AI Services (top):</strong>
Prebuilt, API-driven — Rekognition, Comprehend, Textract, Transcribe, Polly. No ML expertise needed.

<strong>2. ML Services (middle):</strong>
Amazon SageMaker for building, training, and deploying custom models.

<strong>3. Frameworks & Infrastructure (bottom):</strong>
TensorFlow, PyTorch on EC2, and accelerators like AWS Trainium and Inferentia.

<strong>Generative AI:</strong> Amazon Bedrock sits alongside these, offering foundation models via a single API.`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 2",
            title: "Fundamentals of Generative AI",
            description: "Learn how generative AI and foundation models work and how Amazon Bedrock delivers them on AWS.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is generative AI?",
                "Foundation models, tokens & embeddings",
                "Amazon Bedrock overview",
                "Prompt engineering basics"
            ],
            detailedDescription: "Generative AI is a major focus of the AIF-C01 exam. This module explains how foundation models work, key terms like tokens and embeddings, and how Amazon Bedrock provides access to leading models through a single API.",
            detailedContent: [
                {
                    title: "What is Generative AI?",
                    content: `<strong>Generative AI</strong> creates new content — text, images, code, audio — from natural-language prompts using large foundation models.

<strong>Common use cases:</strong>
• Chatbots and virtual assistants
• Text summarization and generation
• Code generation
• Image creation
• Search and question answering

<strong>Limitations to know:</strong> hallucinations, knowledge cutoffs, and non-determinism — all reasons to ground and evaluate models.`
                },
                {
                    title: "Foundation Models, Tokens & Embeddings",
                    content: `A <strong>foundation model (FM)</strong> is a large model pretrained on massive data that can be adapted to many tasks.

<strong>Key terms:</strong>
• <strong>Tokens:</strong> Word pieces the model processes; usage is billed per token
• <strong>Embeddings:</strong> Numeric vectors representing meaning, used for search and RAG
• <strong>Context window:</strong> Max tokens the model can consider at once
• <strong>Temperature / top-p:</strong> Control randomness of output

<strong>Model types:</strong> text (LLMs), embedding models, and image models.`,
                    code: `# Generate embeddings with Amazon Bedrock (Titan Embeddings)
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "What is gradient descent?"}),
)
embedding = json.loads(response["body"].read())["embedding"]
print("Vector length:", len(embedding))`
                },
                {
                    title: "Amazon Bedrock Overview",
                    content: `<strong>Amazon Bedrock</strong> is a fully managed service that offers foundation models from Amazon and leading providers (Anthropic Claude, Meta Llama, Mistral, and more) through a single API.

<strong>Key features:</strong>
• Serverless — no infrastructure to manage
• Model choice via one consistent API
• Knowledge Bases for RAG
• Agents for multi-step tasks
• Guardrails for safety
• Fine-tuning and customization

<strong>Exam tip:</strong> Bedrock is AWS's primary generative AI service and keeps your data private to your account.`,
                    code: `# Invoke a chat model on Amazon Bedrock (Anthropic Claude)
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 200,
    "messages": [{"role": "user", "content": "Explain overfitting briefly."}],
}
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    body=json.dumps(body),
)
print(json.loads(response["body"].read())["content"][0]["text"])`
                },
                {
                    title: "Prompt Engineering Basics",
                    content: `<strong>Prompt engineering</strong> shapes model behavior through carefully written instructions.

<strong>Techniques:</strong>
• <strong>Zero-shot:</strong> Ask directly with no examples
• <strong>Few-shot:</strong> Provide sample input/output pairs
• <strong>Chain-of-thought:</strong> Ask the model to reason step by step
• <strong>System prompts:</strong> Set role and constraints

<strong>Negative prompting</strong> tells the model what to avoid. Lower temperature yields more deterministic answers.`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 3",
            title: "Applications of Foundation Models",
            description: "Apply foundation models with prompting, retrieval augmented generation, agents, and customization on AWS.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Designing effective prompts",
                "Retrieval Augmented Generation (RAG)",
                "Amazon Bedrock Agents",
                "Customizing models (fine-tuning)",
                "Evaluating model performance"
            ],
            detailedDescription: "This module covers how foundation models are applied in real solutions: crafting prompts, grounding models on your data with RAG and Bedrock Knowledge Bases, automating tasks with Agents, customizing models, and evaluating quality.",
            detailedContent: [
                {
                    title: "Designing Effective Prompts",
                    content: `Good prompts are specific, structured, and constrained.

<strong>A strong prompt includes:</strong>
• <strong>Role/persona:</strong> "You are a helpful ML tutor"
• <strong>Task:</strong> The precise request
• <strong>Context:</strong> Supporting information
• <strong>Constraints:</strong> Format, length, tone
• <strong>Examples:</strong> Few-shot demonstrations

<strong>Inference parameters:</strong> temperature, top-p, and max tokens tune creativity and length.`
                },
                {
                    title: "Retrieval Augmented Generation (RAG)",
                    content: `<strong>RAG</strong> grounds a model on your own data so answers are accurate and current — without retraining.

<strong>The pattern:</strong>
1. Store documents as embeddings in a vector store
2. Retrieve the most relevant chunks for a question
3. Add them to the prompt as context
4. The model answers using that context

<strong>On AWS:</strong> <strong>Amazon Bedrock Knowledge Bases</strong> manage the ingestion, embedding, and retrieval, often backed by Amazon OpenSearch Serverless or Aurora pgvector.`,
                    code: `# Query a Bedrock Knowledge Base with RAG (retrieve + generate)
import boto3

agent_rt = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

response = agent_rt.retrieve_and_generate(
    input={"text": "How do I deploy a SageMaker endpoint?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "KB123456",
            "modelArn": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        },
    },
)
print(response["output"]["text"])`
                },
                {
                    title: "Amazon Bedrock Agents",
                    content: `<strong>Amazon Bedrock Agents</strong> let a foundation model plan and take actions by calling APIs and knowledge bases.

<strong>How an agent works:</strong>
• You define <strong>action groups</strong> (APIs via Lambda) it can call
• Attach <strong>knowledge bases</strong> for grounding
• The agent reasons, calls tools, and returns a result

<strong>Use cases:</strong> booking systems, IT automation, customer support — anywhere the model must <em>act</em>, not just answer.`
                },
                {
                    title: "Customizing Models (Fine-tuning)",
                    content: `When prompting and RAG aren't enough, <strong>customize</strong> a foundation model.

<strong>Customization options on Bedrock:</strong>
• <strong>Fine-tuning:</strong> Train on labeled examples to adapt style/behavior
• <strong>Continued pre-training:</strong> Adapt to a domain with unlabeled data

<strong>Trade-offs:</strong> Customization needs quality data and cost; RAG is usually cheaper and easier to keep current. Choose based on the problem.`
                },
                {
                    title: "Evaluating Model Performance",
                    content: `Generative outputs must be measured, not assumed.

<strong>Evaluation approaches:</strong>
• <strong>Human evaluation:</strong> Reviewers rate responses
• <strong>Automatic metrics:</strong> BLEU, ROUGE for text similarity
• <strong>Model-based evaluation:</strong> An LLM scores responses

<strong>On AWS:</strong> Amazon Bedrock provides <strong>model evaluation</strong> jobs (automatic and human) to compare models on accuracy, robustness, and toxicity before you choose one.`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 4",
            title: "Guidelines for Responsible AI",
            description: "Apply responsible AI principles and use Guardrails for Amazon Bedrock to build safe, fair systems.",
            duration: "40 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Dimensions of responsible AI",
                "Bias and fairness",
                "Transparency and explainability",
                "Guardrails for Amazon Bedrock"
            ],
            detailedDescription: "Responsible AI is a core exam domain. This module covers AWS's dimensions of responsible AI, how to detect bias, the importance of transparency, and how Guardrails for Amazon Bedrock enforce safety.",
            detailedContent: [
                {
                    title: "Dimensions of Responsible AI",
                    content: `AWS defines several dimensions of responsible AI:

• <strong>Fairness:</strong> Avoid bias across groups
• <strong>Explainability:</strong> Understand model decisions
• <strong>Robustness:</strong> Perform reliably under varied inputs
• <strong>Privacy & security:</strong> Protect data
• <strong>Governance:</strong> Policies and oversight
• <strong>Transparency:</strong> Communicate capabilities and limits

<strong>Exam tip:</strong> Expect scenario questions asking which dimension applies.`
                },
                {
                    title: "Bias and Fairness",
                    content: `<strong>Bias</strong> occurs when a model systematically disadvantages certain groups, often due to unrepresentative data.

<strong>Sources of bias:</strong>
• Skewed or incomplete training data
• Proxy features correlated with sensitive attributes
• Feedback loops that reinforce bias

<strong>On AWS:</strong> <strong>Amazon SageMaker Clarify</strong> detects bias in data and models and measures feature importance to support fairness.`,
                    code: `# Configure a SageMaker Clarify bias analysis (concept)
from sagemaker.clarify import BiasConfig, DataConfig

bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="gender",          # sensitive attribute
    facet_values_or_threshold=[0],
)
# SageMakerClarifyProcessor runs pre- and post-training bias metrics
print("Analyzing bias for facet:", bias_config.analysis_config["facet"])`
                },
                {
                    title: "Transparency and Explainability",
                    content: `<strong>Transparency</strong> means communicating how a system works and its limits; <strong>explainability</strong> means understanding why it made a decision.

<strong>Techniques:</strong>
• Feature importance (which inputs mattered)
• SHAP values for individual predictions
• Model cards documenting intended use and limits

<strong>On AWS:</strong> SageMaker Clarify produces explainability reports, and <strong>AI Service Cards</strong> document AWS AI service capabilities and limitations.`
                },
                {
                    title: "Guardrails for Amazon Bedrock",
                    content: `<strong>Guardrails for Amazon Bedrock</strong> enforce safety policies on generative AI applications.

<strong>What guardrails can do:</strong>
• Filter harmful content (hate, violence, sexual, insults)
• Block denied topics
• Redact or block PII
• Apply word filters
• Detect and block prompt attacks and hallucinations (contextual grounding)

<strong>Exam tip:</strong> Guardrails apply to both user input and model output and can be reused across models.`,
                    code: `# Apply a guardrail when invoking a Bedrock model
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    guardrailIdentifier="gr-abc123",
    guardrailVersion="1",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Tell me about our courses."}],
    }),
)
print(json.loads(response["body"].read()))`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 5",
            title: "Security, Compliance & Governance for AI",
            description: "Secure AI systems on AWS with IAM, data protection, governance, and monitoring.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Securing AI systems with IAM",
                "Data protection & privacy",
                "Governance & compliance",
                "Monitoring & auditing"
            ],
            detailedDescription: "The final AIF-C01 domain covers securing AI workloads. This module explains access control with IAM, protecting data, governance and compliance frameworks, and monitoring and auditing AI systems on AWS.",
            detailedContent: [
                {
                    title: "Securing AI Systems with IAM",
                    content: `<strong>AWS Identity and Access Management (IAM)</strong> controls who can access AI resources.

<strong>Best practices:</strong>
• Grant <strong>least privilege</strong> with fine-grained policies
• Use <strong>IAM roles</strong> for services, not long-lived keys
• Separate duties across environments

<strong>On AWS:</strong> Restrict which Bedrock models and SageMaker resources a principal can invoke using IAM policies and resource conditions.`,
                    code: `// IAM policy: allow invoking only one Bedrock model
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "bedrock:InvokeModel",
    "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
  }]
}`
                },
                {
                    title: "Data Protection & Privacy",
                    content: `Protecting data is central to responsible AI.

<strong>Controls:</strong>
• <strong>Encryption at rest</strong> with AWS KMS
• <strong>Encryption in transit</strong> with TLS
• <strong>VPC endpoints (PrivateLink)</strong> to keep traffic off the public internet
• Data residency by region choice

<strong>Bedrock privacy:</strong> Your prompts and data are not used to train the base models and stay within your account.`
                },
                {
                    title: "Governance & Compliance",
                    content: `<strong>Governance</strong> establishes policies and accountability for AI use.

<strong>Key ideas:</strong>
• Define acceptable-use and data policies
• Track model lineage and versions
• Meet compliance frameworks (SOC, ISO, HIPAA, GDPR)

<strong>On AWS:</strong> <strong>AWS Audit Manager</strong>, <strong>AWS Config</strong>, and <strong>AWS Artifact</strong> support compliance evidence and continuous governance.`
                },
                {
                    title: "Monitoring & Auditing",
                    content: `Observability keeps AI systems safe and accountable.

<strong>Tools:</strong>
• <strong>Amazon CloudWatch:</strong> Metrics, logs, and alarms
• <strong>AWS CloudTrail:</strong> Audit every API call (who did what, when)
• <strong>Bedrock model invocation logging:</strong> Capture prompts and responses to S3/CloudWatch

<strong>Exam tip:</strong> CloudTrail = auditing API activity; CloudWatch = operational metrics and logs.`,
                    code: `# Enable Bedrock model invocation logging to CloudWatch/S3
import boto3

bedrock = boto3.client("bedrock", region_name="us-east-1")

bedrock.put_model_invocation_logging_configuration(
    loggingConfig={
        "cloudWatchConfig": {"logGroupName": "/bedrock/invocations",
                              "roleArn": "arn:aws:iam::111122223333:role/BedrockLogs"},
        "textDataDeliveryEnabled": True,
    }
)
print("Invocation logging enabled")`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 6",
            title: "AWS AI/ML Service Landscape & Choosing the Right Service",
            description: "Know the full AWS AI stack and pick the right service: Bedrock, SageMaker, and the managed AI services, plus foundation model selection and managed RAG.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "The AWS AI/ML stack",
                "Amazon Bedrock deep dive",
                "Managed AI services",
                "Foundation model selection",
                "Knowledge Bases & Agents",
                "Choosing the right service"
            ],
            detailedDescription: "AIF-C01 tests whether you can map business problems to the right AWS AI service. This module covers the layered AWS AI/ML stack, Amazon Bedrock in depth, the managed AI services, how to choose a foundation model, and managed RAG and agents.",
            detailedContent: [
                {
                    title: "The AWS AI/ML Stack",
                    content: `AWS organizes AI/ML into layers, simplest first.

<strong>AI services (no ML expertise):</strong>
• Rekognition (vision), Comprehend (NLP), Textract (documents)
• Transcribe (speech-to-text), Polly (text-to-speech), Translate
• Lex (chatbots), Personalize (recommendations), Kendra (search)

<strong>Generative AI:</strong>
• Amazon Bedrock (foundation models via one API)
• Amazon Q (AI assistant)

<strong>ML platform:</strong>
• Amazon SageMaker (build/train/deploy custom models)

<strong>Exam tip:</strong> prefer a managed AI service for common tasks; use Bedrock for generative AI; use SageMaker for custom models.`
                },
                {
                    title: "Amazon Bedrock Deep Dive",
                    content: `<strong>Amazon Bedrock</strong> provides serverless access to foundation models from Amazon and partners (Anthropic Claude, Meta Llama, Mistral, and more) through a single API.

<strong>Key features:</strong>
• <strong>Model choice</strong> across providers
• <strong>Knowledge Bases</strong> for managed RAG
• <strong>Agents</strong> for multi-step tasks and tool use
• <strong>Guardrails</strong> for safety and PII controls
• <strong>Fine-tuning</strong> and custom models

<strong>Exam tip:</strong> Bedrock is the default answer for building generative AI on AWS without managing infrastructure.`,
                    code: `# Invoke a foundation model on Amazon Bedrock (Python)
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
resp = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Explain generative AI in one line."}],
    }))
print(json.loads(resp["body"].read())["content"][0]["text"])`
                },
                {
                    title: "Managed AI Services",
                    content: `AWS managed AI services solve common problems with a single API call.

<strong>Match service to task:</strong>
• <strong>Rekognition</strong> - image/video analysis, moderation
• <strong>Textract</strong> - text, forms, and tables from documents
• <strong>Comprehend</strong> - sentiment, entities, key phrases, PII
• <strong>Transcribe / Polly</strong> - speech-to-text / text-to-speech
• <strong>Translate</strong> - language translation
• <strong>Lex</strong> - conversational bots
• <strong>Personalize</strong> - recommendations
• <strong>Kendra</strong> - intelligent enterprise search

<strong>Rule:</strong> pick the managed service that matches the task before building a custom model.`
                },
                {
                    title: "Foundation Model Selection",
                    content: `Choosing a foundation model balances quality, cost, latency, and modality.

<strong>Consider:</strong>
• <strong>Capability</strong>: reasoning, coding, vision, long context
• <strong>Cost</strong>: price per input/output token
• <strong>Latency</strong>: smaller models respond faster
• <strong>Modality</strong>: text-only vs multimodal

<strong>On Bedrock:</strong> you can evaluate and switch models through one API, so start with a smaller/cheaper model and scale up only if quality requires it.`
                },
                {
                    title: "Knowledge Bases & Agents",
                    content: `Two Bedrock features power real applications.

<strong>Knowledge Bases (managed RAG):</strong>
• Connect foundation models to your data sources
• Handle chunking, embeddings, and retrieval for you
• Ground answers and add citations

<strong>Agents:</strong>
• Plan multi-step tasks
• Call APIs and tools (action groups)
• Combine with Knowledge Bases for grounded actions

<strong>Exam tip:</strong> Knowledge Bases = managed RAG; Agents = reasoning + tool use.`
                },
                {
                    title: "Choosing the Right Service",
                    content: `The exam frequently asks you to map a scenario to a service.

<strong>Quick map:</strong>
• Analyze images/video -> Rekognition
• Extract data from documents -> Textract
• Sentiment/entities from text -> Comprehend
• Speech-to-text -> Transcribe
• Build a chatbot -> Lex (or Bedrock Agents for generative)
• Recommendations -> Personalize
• Enterprise search -> Kendra
• Generative AI app -> Bedrock (+ Knowledge Bases/Agents/Guardrails)
• Custom ML model -> SageMaker`,
                    code: `# Map scenarios to the right AWS AI service
services = {
    "Moderate user-uploaded images":     "Amazon Rekognition",
    "Extract fields from invoices":      "Amazon Textract",
    "Detect sentiment in reviews":       "Amazon Comprehend",
    "Answer from company docs (RAG)":    "Bedrock Knowledge Bases",
    "Train a custom fraud model":        "Amazon SageMaker",
}
for scenario, service in services.items():
    print(f"{scenario:32} -> {service}")`
                }
            ]
        },
        {
            number: "AIF-C01 · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Amazon Bedrock labs plus a domain-mapped study plan and exam-day strategy for AIF-C01.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Bedrock playground",
                "Lab: Knowledge Bases (RAG)",
                "Lab: Guardrails",
                "Lab: A managed AI service",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn AIF-C01 concepts into practice with guided labs in the AWS console, then map your study to the official exam domains and apply a practical exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Bedrock Playground",
                    content: `<strong>Goal:</strong> experience foundation models and prompting.

<strong>Steps:</strong>
1. In the AWS console, open Amazon Bedrock and request model access.
2. Open the Chat/Text playground and pick a model (e.g., Claude).
3. Try a prompt, then adjust temperature and max tokens.
4. Compare two different models on the same prompt.

<strong>Takeaway:</strong> Bedrock gives one interface to many models; parameters shape output.`
                },
                {
                    title: "Lab: Knowledge Bases (RAG)",
                    content: `<strong>Goal:</strong> see managed retrieval-augmented generation.

<strong>Steps:</strong>
1. Upload a few documents to an S3 bucket.
2. Create a Bedrock Knowledge Base pointing to that bucket.
3. Ask questions and observe grounded answers with citations.

<strong>Takeaway:</strong> Knowledge Bases handle chunking, embeddings, and retrieval so answers are grounded in your data.`
                },
                {
                    title: "Lab: Guardrails",
                    content: `<strong>Goal:</strong> apply responsible AI controls.

<strong>Steps:</strong>
1. Create a Bedrock Guardrail.
2. Add denied topics and a PII filter.
3. Test prompts that should be blocked or redacted.

<strong>Takeaway:</strong> Guardrails enforce content policies, denied topics, and PII protection on input and output.`
                },
                {
                    title: "Lab: A Managed AI Service",
                    content: `<strong>Goal:</strong> solve a task with no ML training.

<strong>Steps:</strong>
1. Open Amazon Comprehend (or Rekognition) in the console.
2. Run the built-in analysis on sample text or an image.
3. Review the structured results and confidence scores.

<strong>Takeaway:</strong> managed AI services deliver instant results for common tasks - no model building required.`,
                    code: `# Detect sentiment with Amazon Comprehend (Python)
import boto3

comprehend = boto3.client("comprehend", region_name="us-east-1")
result = comprehend.detect_sentiment(
    Text="This AWS course is excellent and easy to follow!",
    LanguageCode="en")
print(result["Sentiment"], result["SentimentScore"])`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official AIF-C01 domains.

<strong>Fundamentals of AI & ML:</strong> Module 1 (AI/ML/DL hierarchy, learning types, lifecycle).
<strong>Fundamentals of Generative AI:</strong> Module 2 plus Module 6 (foundation models, Bedrock).
<strong>Applications of foundation models:</strong> Modules 3 and 6 (prompting, RAG, agents, model selection).
<strong>Guidelines for responsible AI:</strong> Module 4 plus Guardrails lab.
<strong>Security, compliance & governance:</strong> Module 5 plus Module 6 (IAM, data protection, monitoring).

<strong>Recommended plan:</strong>
• Week 1: Modules 1-2 + Bedrock playground lab
• Week 2: Modules 3 + 6 + Knowledge Bases lab
• Week 3: Modules 4-5 + Guardrails lab + AWS Skill Builder practice
• Week 4: Review + official practice question set

<strong>Pair with:</strong> the free AWS Skill Builder AIF-C01 course and official practice questions.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `AIF-C01 is conceptual and scenario-based - no coding required.

<strong>During the exam:</strong>
• Map each scenario to the right layer (managed AI service, Bedrock, or SageMaker)
• Choose the managed service that matches the task before custom ML
• For generative AI apps, think Bedrock (+ Knowledge Bases/Agents/Guardrails)
• When responsible AI or PII is mentioned, think Guardrails and governance
• Watch for "least effort," "no ML expertise," or "fully managed" qualifiers

<strong>Mindset:</strong> you are selecting the right AWS AI capability for a business need, not writing code.`
                }
            ]
        }
    ],

    // ==========================================================
    // MLA-C01: AWS Certified Machine Learning Engineer - Associate
    // ==========================================================
    awsMlEngineer: [
        {
            number: "MLA-C01 · Module 1",
            title: "Data Preparation for ML",
            description: "Ingest, store, transform, and engineer features for machine learning on AWS.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Ingesting & storing data (Amazon S3)",
                "Transforming data (Glue & Data Wrangler)",
                "Feature engineering",
                "SageMaker Feature Store"
            ],
            detailedDescription: "Data preparation is the first and largest domain of MLA-C01. This module covers ingesting and storing data on Amazon S3, transforming it with AWS Glue and SageMaker Data Wrangler, engineering features, and reusing them with SageMaker Feature Store.",
            detailedContent: [
                {
                    title: "Ingesting & Storing Data (Amazon S3)",
                    content: `<strong>Amazon S3</strong> is the foundation for ML data on AWS — durable, scalable object storage that serves as the data lake.

<strong>Ingestion options:</strong>
• Batch uploads and AWS DataSync
• Streaming with Amazon Kinesis / Firehose
• Database exports via AWS DMS

<strong>Formats:</strong> Prefer columnar formats like Parquet for analytics; partition data (e.g., by date) for efficient access.`,
                    code: `# Upload a training dataset to S3 (boto3)
import boto3

s3 = boto3.client("s3")
s3.upload_file(
    Filename="train.parquet",
    Bucket="my-ml-bucket",
    Key="datasets/diabetes/train.parquet",
)
print("Uploaded to s3://my-ml-bucket/datasets/diabetes/train.parquet")`
                },
                {
                    title: "Transforming Data (Glue & Data Wrangler)",
                    content: `Raw data must be cleaned and reshaped before training.

<strong>AWS Glue:</strong>
Serverless ETL for large-scale transformation, with a Data Catalog of table schemas.

<strong>SageMaker Data Wrangler:</strong>
A visual tool with 300+ built-in transforms for cleaning, encoding, and joining — exportable to a pipeline.

<strong>Exam tip:</strong> Glue for big serverless ETL; Data Wrangler for interactive feature prep inside SageMaker.`,
                    code: `# Read, clean, and write data with AWS Glue (PySpark)
from awsglue.context import GlueContext
from pyspark.context import SparkContext

glue = GlueContext(SparkContext.getOrCreate())
df = glue.create_dynamic_frame.from_catalog(
    database="ml_db", table_name="raw_patients").toDF()

df = df.dropna().dropDuplicates()
df.write.mode("overwrite").parquet("s3://my-ml-bucket/clean/patients/")`
                },
                {
                    title: "Feature Engineering",
                    content: `<strong>Feature engineering</strong> transforms raw data into inputs that improve model accuracy.

<strong>Common techniques:</strong>
• Scaling and normalization
• One-hot / label encoding for categories
• Handling missing values
• Creating interaction and date-based features
• Text vectorization / embeddings

<strong>Tip:</strong> Apply the exact same transformations at training and inference to avoid <em>training/serving skew</em>.`
                },
                {
                    title: "SageMaker Feature Store",
                    content: `<strong>Amazon SageMaker Feature Store</strong> is a central repository to store, share, and reuse features.

<strong>Two stores:</strong>
• <strong>Online store:</strong> Low-latency features for real-time inference
• <strong>Offline store:</strong> Historical features (in S3) for training

<strong>Benefits:</strong> Consistency across teams, reuse across models, and elimination of training/serving skew.`,
                    code: `# Ingest features into a SageMaker Feature Group
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

session = sagemaker.Session()
fg = FeatureGroup(name="patient-features", sagemaker_session=session)

# df must include a record identifier + event time column
fg.ingest(data_frame=df, max_workers=4, wait=True)
print("Features ingested into online & offline stores")`
                }
            ]
        },
        {
            number: "MLA-C01 · Module 2",
            title: "ML Model Development",
            description: "Choose an approach, train, tune, and evaluate models with Amazon SageMaker.",
            duration: "60 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Choosing a modeling approach",
                "Training models with SageMaker",
                "Hyperparameter tuning",
                "Evaluating models"
            ],
            detailedDescription: "This module covers developing ML models on AWS: selecting an approach (built-in algorithm, custom, or pretrained), training on SageMaker, tuning hyperparameters automatically, and evaluating with the right metrics.",
            detailedContent: [
                {
                    title: "Choosing a Modeling Approach",
                    content: `Pick the approach that fits the problem and your resources.

<strong>Options on SageMaker:</strong>
• <strong>Built-in algorithms:</strong> XGBoost, Linear Learner, etc. — optimized and easy
• <strong>Script mode:</strong> Bring your own TensorFlow/PyTorch/scikit-learn code
• <strong>Bring your own container:</strong> Full control
• <strong>JumpStart:</strong> Pretrained models and solutions to fine-tune

<strong>Exam tip:</strong> Favor built-in algorithms or JumpStart when they fit — less code and cost.`
                },
                {
                    title: "Training Models with SageMaker",
                    content: `<strong>SageMaker training jobs</strong> run your training on managed, scalable compute.

<strong>You specify:</strong>
• An estimator (algorithm/image)
• Instance type and count
• Input data channels (S3)
• Hyperparameters

SageMaker provisions the compute, runs the job, saves the model artifact to S3, and tears everything down — you pay only for training time.`,
                    code: `# Train an XGBoost model with SageMaker (built-in algorithm)
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session()
image = sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, "1.7-1")

estimator = Estimator(
    image_uri=image,
    role="arn:aws:iam::111122223333:role/SageMakerRole",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-ml-bucket/models/",
)
estimator.set_hyperparameters(objective="binary:logistic", num_round=100)
estimator.fit({"train": "s3://my-ml-bucket/datasets/diabetes/train/"})`
                },
                {
                    title: "Hyperparameter Tuning",
                    content: `<strong>Automatic Model Tuning (AMT)</strong> searches hyperparameters to maximize a metric.

<strong>Configure:</strong>
• Parameter ranges (continuous, integer, categorical)
• Objective metric (e.g., validation AUC)
• Strategy: Bayesian (default), random, grid, or Hyperband
• Max jobs and parallelism

<strong>Tip:</strong> Bayesian search is efficient for expensive models; Hyperband stops weak trials early.`,
                    code: `# Automatic hyperparameter tuning with SageMaker
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges={"eta": ContinuousParameter(0.01, 0.3),
                           "max_depth": ContinuousParameter(3, 10)},
    max_jobs=20, max_parallel_jobs=4, strategy="Bayesian",
)
tuner.fit({"train": "s3://my-ml-bucket/datasets/diabetes/train/"})`
                },
                {
                    title: "Evaluating Models",
                    content: `Choose evaluation metrics that match the problem.

<strong>Classification:</strong> accuracy, precision, recall, F1, AUC-ROC. Watch for class imbalance.

<strong>Regression:</strong> RMSE, MAE, R².

<strong>Guard against overfitting:</strong> evaluate on a held-out set and use cross-validation.

<strong>On AWS:</strong> SageMaker captures metrics from training logs, and SageMaker Clarify adds bias and explainability reports.`
                }
            ]
        },
        {
            number: "MLA-C01 · Module 3",
            title: "Deployment & Orchestration of ML Workflows",
            description: "Deploy models to endpoints and automate ML workflows with pipelines and CI/CD.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Deployment options (endpoints)",
                "Infrastructure as code",
                "SageMaker Pipelines",
                "CI/CD for ML"
            ],
            detailedDescription: "This module covers taking models to production: choosing the right SageMaker deployment option, defining infrastructure as code, orchestrating steps with SageMaker Pipelines, and automating with CI/CD.",
            detailedContent: [
                {
                    title: "Deployment Options (Endpoints)",
                    content: `SageMaker offers several inference options — pick based on latency and traffic.

• <strong>Real-time endpoints:</strong> Always-on, low latency
• <strong>Serverless inference:</strong> Auto-scales, pay per request, good for spiky traffic
• <strong>Asynchronous inference:</strong> Large payloads, near-real-time
• <strong>Batch transform:</strong> Score large datasets offline

<strong>Exam tip:</strong> Match the option to the workload — serverless for intermittent, batch for bulk scoring.`,
                    code: `# Deploy a trained model to a real-time endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="diabetes-endpoint",
)
result = predictor.predict([[5, 116, 74, 0, 0, 25.6, 0.201, 30]])
print("Prediction:", result)`
                },
                {
                    title: "Infrastructure as Code",
                    content: `Define ML infrastructure declaratively so it is repeatable and versioned.

<strong>Options on AWS:</strong>
• <strong>AWS CloudFormation:</strong> JSON/YAML templates
• <strong>AWS CDK:</strong> Infrastructure in Python/TypeScript
• <strong>Terraform:</strong> Multi-cloud IaC

<strong>Why it matters:</strong> Reproducible environments, peer review, and safe rollbacks — essential for production MLOps.`
                },
                {
                    title: "SageMaker Pipelines",
                    content: `<strong>Amazon SageMaker Pipelines</strong> is a purpose-built CI/CD service for ML that chains steps into a repeatable workflow.

<strong>Typical steps:</strong>
• Processing (data prep)
• Training
• Evaluation (conditional)
• Model registration
• Deployment

<strong>Benefits:</strong> Lineage tracking, caching, and integration with the SageMaker Model Registry for approvals.`,
                    code: `# Define a minimal SageMaker Pipeline (SDK)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

train_step = TrainingStep(name="TrainModel", estimator=estimator,
                          inputs={"train": "s3://my-ml-bucket/datasets/diabetes/train/"})

pipeline = Pipeline(name="diabetes-pipeline", steps=[train_step])
pipeline.upsert(role_arn="arn:aws:iam::111122223333:role/SageMakerRole")
pipeline.start()`
                },
                {
                    title: "CI/CD for ML",
                    content: `<strong>CI/CD</strong> automates testing and releasing ML code and models.

<strong>AWS building blocks:</strong>
• <strong>CodePipeline / CodeBuild:</strong> Orchestrate build and deploy
• <strong>SageMaker Model Registry:</strong> Version and approve models
• <strong>SageMaker Projects:</strong> Prebuilt MLOps templates

<strong>Flow:</strong> commit → build/test → train → register → approve → deploy, with rollbacks on failure.`
                }
            ]
        },
        {
            number: "MLA-C01 · Module 4",
            title: "ML Solution Monitoring, Maintenance & Security",
            description: "Monitor models for drift, maintain quality, secure workloads, and control cost.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Monitoring with SageMaker Model Monitor",
                "Detecting drift & retraining",
                "Securing ML workloads",
                "Cost optimization"
            ],
            detailedDescription: "The final MLA-C01 domain covers keeping models healthy in production: monitoring with SageMaker Model Monitor, detecting drift and retraining, securing workloads, and optimizing cost.",
            detailedContent: [
                {
                    title: "Monitoring with SageMaker Model Monitor",
                    content: `<strong>Amazon SageMaker Model Monitor</strong> continuously checks deployed models for quality issues.

<strong>What it monitors:</strong>
• <strong>Data quality:</strong> Schema and statistics vs. a baseline
• <strong>Model quality:</strong> Accuracy vs. ground truth
• <strong>Bias drift</strong> and <strong>feature attribution drift</strong> (with Clarify)

Alerts are sent via CloudWatch when metrics breach thresholds.`,
                    code: `# Enable data capture on an endpoint for monitoring
from sagemaker.model_monitor import DataCaptureConfig

capture = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri="s3://my-ml-bucket/monitor/captured/",
)
predictor = estimator.deploy(
    initial_instance_count=1, instance_type="ml.m5.large",
    data_capture_config=capture)`
                },
                {
                    title: "Detecting Drift & Retraining",
                    content: `<strong>Drift</strong> happens when live data diverges from training data, degrading accuracy.

<strong>Types:</strong>
• <strong>Data drift:</strong> Input distribution changes
• <strong>Concept drift:</strong> The input-to-output relationship changes

<strong>Response:</strong> When Model Monitor flags drift, trigger an automated retraining pipeline (SageMaker Pipelines) and deploy a refreshed model — closing the MLOps loop.`
                },
                {
                    title: "Securing ML Workloads",
                    content: `Security spans data, models, and endpoints.

<strong>Controls:</strong>
• <strong>IAM</strong> least-privilege roles for jobs and endpoints
• <strong>VPC</strong> isolation and PrivateLink for endpoints
• <strong>KMS</strong> encryption for data and model artifacts
• <strong>CloudTrail</strong> auditing of API calls

<strong>Exam tip:</strong> Run training and endpoints in a VPC without internet access for sensitive data.`
                },
                {
                    title: "Cost Optimization",
                    content: `Control ML spend without sacrificing performance.

<strong>Levers:</strong>
• Right-size instance types; use GPU only when needed
• <strong>Spot instances</strong> for training (up to ~90% savings)
• <strong>Serverless</strong> or <strong>multi-model endpoints</strong> for intermittent traffic
• Auto-scaling and scheduled shutdown of notebooks

<strong>Tools:</strong> AWS Cost Explorer and CloudWatch usage metrics to track spend.`,
                    code: `# Use managed Spot training to cut training cost
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=image,
    role="arn:aws:iam::111122223333:role/SageMakerRole",
    instance_count=1, instance_type="ml.m5.xlarge",
    use_spot_instances=True,
    max_run=3600, max_wait=7200,   # required with spot
)`
                }
            ]
        },
        {
            number: "MLA-C01 · Module 6",
            title: "SageMaker MLOps: Pipelines, CI/CD & Monitoring",
            description: "Operationalize ML on AWS with SageMaker Pipelines, model registry, CI/CD, and Model Monitor for production reliability.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "SageMaker Pipelines",
                "Model Registry & approval",
                "CI/CD for ML on AWS",
                "SageMaker Model Monitor",
                "Cost & deployment options"
            ],
            detailedDescription: "MLA-C01 is heavily focused on operationalizing ML. This module covers SageMaker Pipelines, the model registry and approval workflow, CI/CD, and Model Monitor for drift and quality in production.",
            detailedContent: [
                {
                    title: "SageMaker Pipelines",
                    content: `<strong>SageMaker Pipelines</strong> define repeatable, CI/CD-style ML workflows.

<strong>Steps can include:</strong>
• Processing (data prep)
• Training and tuning
• Evaluation with conditions
• Model registration

<strong>Exam tip:</strong> Pipelines provide reproducibility and lineage - the backbone of SageMaker MLOps.`,
                    code: `# A minimal SageMaker Pipeline (concept)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

pipeline = Pipeline(
    name="churn-pipeline",
    steps=[TrainingStep(name="Train", estimator=est, inputs=train_input)])
pipeline.upsert(role_arn="arn:aws:iam::111122223333:role/SageMakerRole")
pipeline.start()`
                },
                {
                    title: "Model Registry & Approval",
                    content: `The <strong>SageMaker Model Registry</strong> versions models and gates deployment.

<strong>Workflow:</strong>
• Register model versions in a model package group
• Set approval status (Pending / Approved)
• Deploy only approved versions via CI/CD

<strong>Exam tip:</strong> the approval workflow is how you govern what reaches production.`
                },
                {
                    title: "CI/CD for ML on AWS",
                    content: `Automate build, test, and deploy of ML.

<strong>Tools:</strong>
• <strong>SageMaker Projects</strong> provision CI/CD templates
• <strong>CodePipeline / CodeBuild</strong> (or GitHub Actions) run the stages
• Approvals gate production deployment

<strong>Exam tip:</strong> SageMaker Projects + CodePipeline is the native MLOps CI/CD path on AWS.`
                },
                {
                    title: "SageMaker Model Monitor",
                    content: `<strong>Model Monitor</strong> watches deployed models for problems.

<strong>Detects:</strong>
• Data quality issues
• Data and model quality drift
• Bias drift and feature attribution drift (with Clarify)

<strong>Action:</strong> alert and trigger retraining when drift crosses thresholds - closing the loop.`
                },
                {
                    title: "Cost & Deployment Options",
                    content: `Choose the right serving and cost strategy.

<strong>Deployment options:</strong>
• Real-time endpoints (low latency)
• Serverless inference (spiky traffic)
• Asynchronous inference (large payloads)
• Batch transform (bulk, offline)

<strong>Cost levers:</strong> right-size instances, use spot for training, and pick serverless/batch to avoid idle endpoints.`
                }
            ]
        },
        {
            number: "MLA-C01 · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided SageMaker labs plus a domain-mapped study plan and exam-day strategy for MLA-C01.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Train & deploy in SageMaker",
                "Lab: Pipeline & registry",
                "Lab: Model Monitor",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn MLA-C01 skills into practice with guided SageMaker labs, then map your study to the official exam domains and apply an exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Train & Deploy in SageMaker",
                    content: `<strong>Goal:</strong> run the core SageMaker flow.

<strong>Steps:</strong>
1. In SageMaker Studio, prepare data in S3.
2. Train with a built-in algorithm (e.g., XGBoost).
3. Deploy to a real-time endpoint.
4. Invoke the endpoint with a test record.

<strong>Takeaway:</strong> data in S3 → train → deploy → invoke is the SageMaker foundation.`
                },
                {
                    title: "Lab: Pipeline & Registry",
                    content: `<strong>Goal:</strong> automate and govern.

<strong>Steps:</strong>
1. Define a SageMaker Pipeline (process → train → evaluate → register).
2. Run it and inspect lineage.
3. Register the model and set approval status.

<strong>Takeaway:</strong> pipelines + registry provide reproducibility and governed deployment.`
                },
                {
                    title: "Lab: Model Monitor",
                    content: `<strong>Goal:</strong> observe a deployed model.

<strong>Steps:</strong>
1. Enable data capture on an endpoint.
2. Create a monitoring schedule with a baseline.
3. Review data-quality and drift reports.

<strong>Takeaway:</strong> Model Monitor detects drift so you can retrain before performance drops.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official MLA-C01 domains.

<strong>Data preparation:</strong> Modules 1-2 (ingest, transform, Feature Store).
<strong>Model development:</strong> Modules 3-4 (training, tuning, evaluation, Clarify).
<strong>Deployment & orchestration:</strong> Module 5 + Module 6 (endpoints, pipelines, CI/CD).
<strong>Monitoring, maintenance & security:</strong> Module 6 + Module 7 (Model Monitor, IAM).

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-4 + train/deploy lab
• Week 3: Modules 5-6 + pipeline and monitor labs
• Week 4: Module 7 + AWS Skill Builder practice

<strong>Pair with:</strong> AWS Skill Builder MLA-C01 course and official practice questions.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `MLA-C01 is technical and scenario-based.

<strong>During the exam:</strong>
• Match serving to needs: real-time, serverless, async, or batch
• For automation, think SageMaker Pipelines + Projects + CodePipeline
• For drift/quality, think Model Monitor (and Clarify for bias)
• For cost, think spot training, right-sizing, and avoiding idle endpoints
• For security, think IAM least privilege and VPC/KMS

<strong>Mindset:</strong> you are the ML engineer operationalizing models reliably and cost-effectively on AWS.`
                }
            ]
        }
    ],

    // ==========================================================
    // MLS-C01: AWS Certified Machine Learning - Specialty
    // ==========================================================
    awsMlSpecialty: [
        {
            number: "MLS-C01 · Module 1",
            title: "Data Engineering",
            description: "Build data lakes and ingestion pipelines to feed machine learning on AWS.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Building data lakes on S3",
                "Data ingestion (Kinesis & Glue)",
                "Storage formats & partitioning"
            ],
            detailedDescription: "The first MLS-C01 domain covers data engineering: designing data lakes on Amazon S3, ingesting batch and streaming data, and choosing storage formats and partitioning strategies for efficient ML.",
            detailedContent: [
                {
                    title: "Building Data Lakes on S3",
                    content: `A <strong>data lake</strong> on Amazon S3 centralizes structured and unstructured data for analytics and ML.

<strong>Key services:</strong>
• <strong>Amazon S3:</strong> Durable, scalable storage
• <strong>AWS Lake Formation:</strong> Governance and fine-grained access
• <strong>AWS Glue Data Catalog:</strong> Central metadata/schema registry

<strong>Zones:</strong> Organize into raw, processed, and curated prefixes for a clean pipeline.`
                },
                {
                    title: "Data Ingestion (Kinesis & Glue)",
                    content: `Data reaches the lake through batch and streaming pipelines.

<strong>Streaming:</strong>
• <strong>Amazon Kinesis Data Streams:</strong> Real-time ingestion
• <strong>Kinesis Data Firehose:</strong> Load streams into S3/Redshift

<strong>Batch:</strong>
• <strong>AWS Glue:</strong> Serverless ETL jobs
• <strong>AWS DMS:</strong> Database migration/replication

<strong>Exam tip:</strong> Firehose for easy stream-to-S3 delivery; Kinesis Data Streams when you need custom real-time processing.`,
                    code: `# Send records to a Kinesis Data Firehose delivery stream
import boto3, json

firehose = boto3.client("firehose")
firehose.put_record(
    DeliveryStreamName="clickstream-to-s3",
    Record={"Data": json.dumps({"user": "u42", "event": "view"}) + "\\n"},
)
print("Record delivered to S3 via Firehose")`
                },
                {
                    title: "Storage Formats & Partitioning",
                    content: `Format and layout have a big impact on ML performance and cost.

<strong>Formats:</strong>
• <strong>Parquet / ORC:</strong> Columnar, compressed — best for analytics/ML
• <strong>CSV / JSON:</strong> Simple but larger and slower

<strong>Partitioning:</strong>
Split data by keys (e.g., year/month/day) so queries scan less data.

<strong>Tip:</strong> Columnar + partitioning dramatically reduces Athena/Glue scan cost and speeds training data reads.`
                }
            ]
        },
        {
            number: "MLS-C01 · Module 2",
            title: "Exploratory Data Analysis",
            description: "Clean, engineer, and visualize data to prepare it for modeling.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Cleaning & preparing data",
                "Feature engineering",
                "Data visualization"
            ],
            detailedDescription: "This domain covers exploratory data analysis: cleaning and preparing data, engineering informative features, and visualizing distributions and relationships before modeling.",
            detailedContent: [
                {
                    title: "Cleaning & Preparing Data",
                    content: `Real data is messy. Cleaning improves model quality.

<strong>Common tasks:</strong>
• Handle missing values (impute or drop)
• Remove duplicates and outliers
• Fix inconsistent types and units
• Balance classes (SMOTE, resampling)

<strong>On AWS:</strong> SageMaker Data Wrangler and Glue DataBrew provide visual, low-code cleaning at scale.`,
                    code: `# Clean data with pandas in a SageMaker notebook
import pandas as pd

df = pd.read_parquet("s3://my-ml-bucket/raw/patients.parquet")
df = df.drop_duplicates()
df["glucose"] = df["glucose"].replace(0, df["glucose"].median())
df = df[df["bmi"] < df["bmi"].quantile(0.99)]   # trim outliers
print(df.describe())`
                },
                {
                    title: "Feature Engineering",
                    content: `<strong>Feature engineering</strong> often improves accuracy more than changing algorithms.

<strong>Techniques:</strong>
• Scaling/normalization for numeric features
• Encoding categoricals (one-hot, target)
• Binning and interaction features
• Date/time decomposition
• Text vectorization (TF-IDF, embeddings)

<strong>Consistency:</strong> Persist transformations (e.g., in Feature Store) to apply identically at inference.`
                },
                {
                    title: "Data Visualization",
                    content: `Visualization reveals patterns, outliers, and relationships.

<strong>Useful plots:</strong>
• Histograms and box plots for distributions
• Scatter plots for relationships
• Correlation heatmaps
• Class-balance bar charts

<strong>On AWS:</strong> Explore in SageMaker Studio notebooks (matplotlib/seaborn), or use Amazon QuickSight for BI dashboards.`
                }
            ]
        },
        {
            number: "MLS-C01 · Module 3",
            title: "Modeling",
            description: "Frame the problem, choose algorithms, train, tune, and evaluate models on AWS.",
            duration: "60 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Framing the ML problem",
                "Choosing algorithms",
                "Training & tuning",
                "Evaluation metrics"
            ],
            detailedDescription: "The largest MLS-C01 domain covers modeling: framing the business problem as an ML task, selecting algorithms, training and tuning on SageMaker, and evaluating with appropriate metrics.",
            detailedContent: [
                {
                    title: "Framing the ML Problem",
                    content: `Translate a business goal into a well-defined ML problem.

<strong>Questions to answer:</strong>
• Is this classification, regression, clustering, or recommendation?
• What is the target and what are the features?
• How will success be measured (business + model metric)?
• Is enough labeled data available?

<strong>Tip:</strong> A clear problem framing prevents wasted effort downstream.`
                },
                {
                    title: "Choosing Algorithms",
                    content: `Select an algorithm suited to the task and data.

<strong>SageMaker built-in algorithms:</strong>
• <strong>XGBoost:</strong> Tabular classification/regression
• <strong>Linear Learner:</strong> Linear models at scale
• <strong>K-Means:</strong> Clustering
• <strong>Image Classification / Object Detection:</strong> Vision
• <strong>seq2seq / BlazingText:</strong> NLP

<strong>Deep learning:</strong> Bring TensorFlow/PyTorch via script mode when needed.`
                },
                {
                    title: "Training & Tuning",
                    content: `Train on managed SageMaker compute and tune for the best result.

<strong>Training:</strong> Use estimators, distributed training for large data, and Spot instances to save cost.

<strong>Tuning:</strong> Automatic Model Tuning searches hyperparameters (Bayesian/Hyperband) toward an objective metric.

<strong>Avoid overfitting:</strong> regularization, early stopping, and validation on held-out data.`,
                    code: `# Distributed training + early stopping (XGBoost estimator)
estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=500,
    early_stopping_rounds=20,   # stop when validation stops improving
    eval_metric="auc",
)
estimator.fit({
    "train": "s3://my-ml-bucket/datasets/diabetes/train/",
    "validation": "s3://my-ml-bucket/datasets/diabetes/val/",
})`
                },
                {
                    title: "Evaluation Metrics",
                    content: `Choose metrics that reflect the business goal.

<strong>Classification:</strong>
• Accuracy (balanced data)
• Precision/recall & F1 (imbalanced)
• AUC-ROC (ranking quality)
• Confusion matrix for error types

<strong>Regression:</strong> RMSE, MAE, R².

<strong>Tip:</strong> For imbalanced fraud/disease problems, recall and precision matter far more than accuracy.`
                }
            ]
        },
        {
            number: "MLS-C01 · Module 4",
            title: "ML Implementation & Operations",
            description: "Deploy, monitor, and secure ML solutions, and leverage AWS AI services.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Deploying models",
                "Monitoring & logging",
                "Security & compliance",
                "AWS AI services"
            ],
            detailedDescription: "The final MLS-C01 domain covers production operations: deploying models with the right inference option, monitoring and logging, securing workloads, and using managed AWS AI services where appropriate.",
            detailedContent: [
                {
                    title: "Deploying Models",
                    content: `Choose an inference option based on latency, throughput, and cost.

• <strong>Real-time endpoints:</strong> Low-latency, always-on
• <strong>Multi-model endpoints:</strong> Many models behind one endpoint
• <strong>Serverless inference:</strong> Spiky/intermittent traffic
• <strong>Batch transform:</strong> Offline scoring of large datasets

<strong>Rollouts:</strong> Use production variants for A/B testing and blue/green deployments.`,
                    code: `# A/B test two model variants on one endpoint (concept)
from sagemaker.session import production_variant

variant_a = production_variant("ModelA", "ml.m5.large",
                               initial_weight=0.9, variant_name="A")
variant_b = production_variant("ModelB", "ml.m5.large",
                               initial_weight=0.1, variant_name="B")
# session.endpoint_from_production_variants([variant_a, variant_b])
print("90/10 traffic split configured")`
                },
                {
                    title: "Monitoring & Logging",
                    content: `Operational visibility keeps ML services reliable.

<strong>Tools:</strong>
• <strong>Amazon CloudWatch:</strong> Latency, invocations, errors, custom metrics + alarms
• <strong>SageMaker Model Monitor:</strong> Data and model quality drift
• <strong>AWS CloudTrail:</strong> API-level audit trail

<strong>Auto scaling:</strong> Scale endpoint instances on invocation metrics to handle load cost-effectively.`
                },
                {
                    title: "Security & Compliance",
                    content: `Secure the full ML lifecycle.

<strong>Controls:</strong>
• <strong>IAM</strong> least-privilege roles
• <strong>VPC</strong> isolation, PrivateLink, and security groups
• <strong>KMS</strong> encryption at rest; TLS in transit
• Network isolation mode for training containers

<strong>Compliance:</strong> Use CloudTrail, AWS Config, and Audit Manager for evidence and continuous checks.`
                },
                {
                    title: "AWS AI Services",
                    content: `Sometimes a <strong>managed AI service</strong> beats building a custom model.

<strong>Prebuilt services:</strong>
• <strong>Amazon Rekognition:</strong> Vision
• <strong>Amazon Comprehend:</strong> NLP
• <strong>Amazon Textract:</strong> Document data extraction
• <strong>Amazon Transcribe / Polly:</strong> Speech
• <strong>Amazon Forecast / Personalize:</strong> Forecasting & recommendations

<strong>Exam tip:</strong> If a prebuilt service solves the problem, prefer it over training a custom model — less cost and maintenance.`,
                    code: `# Detect sentiment with Amazon Comprehend (no training needed)
import boto3

comprehend = boto3.client("comprehend", region_name="us-east-1")
result = comprehend.detect_sentiment(
    Text="This course made AWS ML easy to understand!",
    LanguageCode="en",
)
print("Sentiment:", result["Sentiment"])`
                }
            ]
        }
    ],

    // ==========================================================
    // AWS Certified Generative AI Developer - Professional
    // ==========================================================
    awsGenAiDeveloper: [
        {
            number: "AWS GenAI Dev · Module 1",
            title: "Foundation Models & Amazon Bedrock",
            description: "Understand foundation models and how to access them through Amazon Bedrock.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Foundation models & modalities",
                "Amazon Bedrock overview",
                "Invoking models with the API"
            ],
            detailedDescription: "This domain introduces foundation models (FMs) and the Amazon Bedrock service. You learn the model families available, how Bedrock provides a unified serverless API, and how to invoke models for text and multimodal generation.",
            detailedContent: [
                {
                    title: "Foundation Models & Modalities",
                    content: `<strong>Foundation models (FMs)</strong> are large models pretrained on broad data and adaptable to many tasks.

<strong>Common modalities:</strong>
• <strong>Text:</strong> Chat, summarization, extraction, code
• <strong>Image:</strong> Text-to-image generation and editing
• <strong>Multimodal:</strong> Reason over text + images
• <strong>Embeddings:</strong> Numeric vectors for search and RAG

<strong>Exam tip:</strong> Match the model family to the task and cost/latency budget rather than always choosing the largest model.`
                },
                {
                    title: "Amazon Bedrock Overview",
                    content: `<strong>Amazon Bedrock</strong> is a fully managed, serverless service that offers a choice of FMs through a single API.

<strong>Key capabilities:</strong>
• Access models from Amazon (Nova, Titan), Anthropic, Meta, Mistral, and others
• <strong>Knowledge Bases</strong> for managed RAG
• <strong>Agents</strong> and <strong>AgentCore</strong> for agentic workflows
• <strong>Guardrails</strong> for safety and responsible AI

<strong>Benefit:</strong> No infrastructure to manage — you call an endpoint and pay per token.`
                },
                {
                    title: "Invoking Models with the API",
                    content: `Bedrock exposes a consistent runtime API to invoke any supported model.

<strong>Two main calls:</strong>
• <strong>InvokeModel:</strong> Model-specific request/response body
• <strong>Converse:</strong> Unified message format across models

<strong>Tip:</strong> Prefer the Converse API for portable, multi-turn chat code.`,
                    code: `# Invoke a foundation model with the Bedrock Converse API
import boto3

bedrock = boto3.client("bedrock-runtime")
response = bedrock.converse(
    modelId="amazon.nova-lite-v1:0",
    messages=[{"role": "user", "content": [{"text": "Summarize RAG in one sentence."}]}],
)
print(response["output"]["message"]["content"][0]["text"])`
                }
            ]
        },
        {
            number: "AWS GenAI Dev · Module 2",
            title: "Prompt Engineering & Application Design",
            description: "Design effective prompts and structure production-ready generative AI applications.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prompt engineering techniques",
                "Controlling output & parameters",
                "Application design patterns"
            ],
            detailedDescription: "This domain covers prompt engineering and application architecture: crafting reliable prompts, tuning inference parameters, and choosing the right pattern (prompting, RAG, or fine-tuning) for a use case.",
            detailedContent: [
                {
                    title: "Prompt Engineering Techniques",
                    content: `<strong>Prompt engineering</strong> shapes model behavior without retraining.

<strong>Techniques:</strong>
• <strong>Zero-shot:</strong> Instruction only
• <strong>Few-shot:</strong> Provide worked examples
• <strong>Chain-of-thought:</strong> Ask the model to reason step by step
• <strong>Role prompting:</strong> Set persona and constraints

<strong>Tip:</strong> Be specific, provide context, and state the output format explicitly.`
                },
                {
                    title: "Controlling Output & Parameters",
                    content: `Inference parameters control creativity and determinism.

<strong>Key parameters:</strong>
• <strong>Temperature:</strong> Higher = more random
• <strong>Top-p / Top-k:</strong> Nucleus sampling limits
• <strong>Max tokens:</strong> Caps response length
• <strong>Stop sequences:</strong> End generation cleanly

<strong>Exam tip:</strong> Lower temperature for factual/extraction tasks; higher for creative generation.`
                },
                {
                    title: "Application Design Patterns",
                    content: `Choose an approach based on data freshness and customization needs.

<strong>Patterns:</strong>
• <strong>Prompting:</strong> Fastest, no external data
• <strong>RAG:</strong> Ground responses in your own data
• <strong>Fine-tuning:</strong> Adapt model weights for style/domain
• <strong>Agents:</strong> Let the model call tools and take actions

<strong>Rule of thumb:</strong> Start with prompting + RAG; fine-tune only when needed.`
                }
            ]
        },
        {
            number: "AWS GenAI Dev · Module 3",
            title: "Retrieval-Augmented Generation (RAG)",
            description: "Ground foundation models in your data using embeddings, vector stores, and Bedrock Knowledge Bases.",
            duration: "65 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Embeddings & vector databases",
                "Chunking & indexing",
                "Bedrock Knowledge Bases",
                "Grounded responses & citations"
            ],
            detailedDescription: "This core domain covers Retrieval-Augmented Generation: creating embeddings, storing them in vector databases, building indexes with chunking strategies, and using Amazon Bedrock Knowledge Bases to return grounded, cited answers.",
            detailedContent: [
                {
                    title: "Embeddings & Vector Databases",
                    content: `<strong>Embeddings</strong> convert text into vectors so similar meaning maps to nearby points.

<strong>Vector stores on AWS:</strong>
• <strong>Amazon OpenSearch Serverless</strong>
• <strong>Amazon Aurora / RDS PostgreSQL (pgvector)</strong>
• <strong>Amazon Neptune Analytics</strong>

<strong>Retrieval:</strong> A query is embedded, then nearest-neighbor search returns the most relevant chunks.`,
                    code: `# Create an embedding with Amazon Bedrock
import boto3, json

bedrock = boto3.client("bedrock-runtime")
resp = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "How do I reset my password?"}),
)
vector = json.loads(resp["body"].read())["embedding"]
print(len(vector), "dimensions")`
                },
                {
                    title: "Chunking & Indexing",
                    content: `Documents are split into <strong>chunks</strong> before embedding.

<strong>Chunking strategies:</strong>
• Fixed-size with overlap
• Sentence/paragraph aware
• Semantic chunking

<strong>Trade-off:</strong> Small chunks improve precision; large chunks preserve context. Tune size and overlap for your corpus.`
                },
                {
                    title: "Bedrock Knowledge Bases",
                    content: `<strong>Amazon Bedrock Knowledge Bases</strong> provide managed, end-to-end RAG.

<strong>What it handles:</strong>
• Ingesting data from Amazon S3 and other sources
• Chunking and embedding automatically
• Storing vectors in a managed index
• Retrieving and augmenting prompts at query time

<strong>Benefit:</strong> You skip building the retrieval pipeline yourself and call a single RetrieveAndGenerate API.`,
                    code: `# Query a Bedrock Knowledge Base with RetrieveAndGenerate
import boto3

agent = boto3.client("bedrock-agent-runtime")
resp = agent.retrieve_and_generate(
    input={"text": "What is our refund policy?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "KB123456",
            "modelArn": "amazon.nova-pro-v1:0",
        },
    },
)
print(resp["output"]["text"])`
                },
                {
                    title: "Grounded Responses & Citations",
                    content: `RAG reduces hallucinations by grounding answers in retrieved sources.

<strong>Good practices:</strong>
• Return <strong>citations</strong> to source documents
• Instruct the model to answer only from context
• Handle "no relevant context" gracefully

<strong>Exam tip:</strong> If answers must be traceable, surface the retrieved passages and their source references.`
                }
            ]
        },
        {
            number: "AWS GenAI Dev · Module 4",
            title: "Agentic AI with Bedrock Agents",
            description: "Build agents that reason, call tools, and orchestrate multi-step workflows with Bedrock AgentCore.",
            duration: "60 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Agents & tool use",
                "Bedrock AgentCore",
                "Orchestrating workflows"
            ],
            detailedDescription: "This domain covers agentic AI: how foundation models can call tools and APIs, how Amazon Bedrock Agents and AgentCore structure reasoning and actions, and how to orchestrate reliable multi-step workflows.",
            detailedContent: [
                {
                    title: "Agents & Tool Use",
                    content: `An <strong>agent</strong> lets a model decide when to call external tools to complete a task.

<strong>Building blocks:</strong>
• <strong>Instructions:</strong> The agent's goal and rules
• <strong>Action groups / tools:</strong> APIs or functions it can invoke
• <strong>Reasoning loop:</strong> Plan → act → observe → repeat

<strong>Use cases:</strong> Booking, data lookups, and multi-API tasks that need decisions.`
                },
                {
                    title: "Bedrock AgentCore",
                    content: `<strong>Amazon Bedrock AgentCore</strong> provides infrastructure to run production agents securely at scale.

<strong>Capabilities:</strong>
• Secure runtime for agent execution
• Memory for context across turns
• Gateway to connect tools and APIs
• Identity and access controls

<strong>Benefit:</strong> Focus on agent logic while AgentCore handles scaling, isolation, and observability.`
                },
                {
                    title: "Orchestrating Workflows",
                    content: `Complex tasks combine agents with orchestration for reliability.

<strong>Options:</strong>
• Single agent with multiple tools
• Multi-agent collaboration (specialist agents)
• <strong>AWS Step Functions</strong> for deterministic orchestration

<strong>Tip:</strong> Add guardrails, retries, and human-in-the-loop checkpoints for high-stakes actions.`
                }
            ]
        },
        {
            number: "AWS GenAI Dev · Module 5",
            title: "Security, Responsible AI & Operations",
            description: "Secure, monitor, and responsibly operate generative AI applications in production.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Guardrails & responsible AI",
                "Security & data protection",
                "Monitoring & cost optimization"
            ],
            detailedDescription: "The final domain covers production concerns: applying Bedrock Guardrails for responsible AI, securing data and access, and monitoring performance, quality, and cost of generative AI workloads.",
            detailedContent: [
                {
                    title: "Guardrails & Responsible AI",
                    content: `<strong>Amazon Bedrock Guardrails</strong> enforce safety policies across models.

<strong>Controls:</strong>
• Block denied topics and harmful content
• Filter and redact <strong>PII</strong>
• Contextual grounding checks to reduce hallucinations
• Word and profanity filters

<strong>Responsible AI:</strong> Test for bias, be transparent about AI use, and keep humans in the loop for sensitive decisions.`
                },
                {
                    title: "Security & Data Protection",
                    content: `Protect data and access across the application.

<strong>Practices:</strong>
• <strong>IAM</strong> least-privilege roles for model and tool access
• Encrypt data at rest (KMS) and in transit (TLS)
• Keep prompts/outputs private; avoid logging secrets
• Use VPC endpoints (PrivateLink) for private connectivity

<strong>Note:</strong> Bedrock does not use your prompts/completions to train the base models.`
                },
                {
                    title: "Monitoring & Cost Optimization",
                    content: `Operate generative AI efficiently and observably.

<strong>Monitoring:</strong>
• <strong>Amazon CloudWatch</strong> metrics and logs
• Track latency, errors, and token usage
• Evaluate output quality with test sets

<strong>Cost control:</strong>
• Right-size models per task
• Cache and reuse embeddings/results
• Cap max tokens and use batching

<strong>Exam tip:</strong> Balance quality, latency, and cost — the cheapest model that meets requirements wins.`
                }
            ]
        },
        {
            number: "GenAI Dev · Module 6",
            title: "Production GenAI: Evaluation, Safety & Observability",
            description: "Ship reliable generative AI on AWS: evaluation, Guardrails, tracing, and cost/latency optimization for production apps.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Evaluating generative apps",
                "Guardrails & responsible AI",
                "Observability & tracing",
                "Security & governance",
                "Cost & latency optimization"
            ],
            detailedDescription: "The Generative AI Developer - Professional exam expects production-grade rigor. This module covers evaluating generative apps, enforcing Guardrails, observability, security/governance, and optimizing cost and latency.",
            detailedContent: [
                {
                    title: "Evaluating Generative Apps",
                    content: `Generative quality must be measured, not assumed.

<strong>Dimensions:</strong>
• Relevance, coherence, fluency
• Groundedness (supported by retrieved context)
• Faithfulness and citation accuracy

<strong>On AWS:</strong> use evaluation datasets and automated scoring (including model-graded evaluation) to compare prompts, models, and RAG configurations.`
                },
                {
                    title: "Guardrails & Responsible AI",
                    content: `<strong>Amazon Bedrock Guardrails</strong> enforce safety policies.

<strong>Controls:</strong>
• Content filters (hate, violence, etc.)
• Denied topics
• PII detection and redaction
• Word/phrase blocklists

<strong>Exam tip:</strong> apply Guardrails to both input and output, and layer them consistently across your app.`
                },
                {
                    title: "Observability & Tracing",
                    content: `Production GenAI must be debuggable.

<strong>Practices:</strong>
• Log prompts, retrieved context, tool calls, and responses
• Trace requests end to end (e.g., with CloudWatch and X-Ray)
• Track token usage, latency, and error rates
• Capture user feedback signals

<strong>Exam tip:</strong> observability is what lets you diagnose hallucinations, latency, and cost issues in production.`
                },
                {
                    title: "Security & Governance",
                    content: `Secure the generative application end to end.

<strong>Controls:</strong>
• IAM least privilege for Bedrock and data sources
• Encryption (KMS) and private networking (VPC endpoints)
• Data residency and no-training guarantees
• Audit logging with CloudTrail

<strong>Exam tip:</strong> combine IAM, encryption, network isolation, and logging for a compliant GenAI app.`
                },
                {
                    title: "Cost & Latency Optimization",
                    content: `Operate generative AI economically.

<strong>Levers:</strong>
• Right-size models (smaller for simple tasks)
• Cache frequent responses and embeddings
• Trim prompts and context length
• Stream responses; parallelize retrieval
• Use provisioned throughput only when needed

<strong>Exam tip:</strong> the cheapest configuration that meets quality and latency targets is the right answer.`
                }
            ]
        },
        {
            number: "GenAI Dev · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Amazon Bedrock labs plus a domain-mapped study plan and exam-day strategy for the Generative AI Developer - Professional exam.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: RAG with Knowledge Bases",
                "Lab: Bedrock Agents with tools",
                "Lab: Guardrails & evaluation",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn Generative AI Developer skills into practice with guided Amazon Bedrock labs, then map your study to the exam domains and apply an exam-day strategy for this advanced, production-focused exam.",
            detailedContent: [
                {
                    title: "Lab: RAG with Knowledge Bases",
                    content: `<strong>Goal:</strong> build managed RAG.

<strong>Steps:</strong>
1. Load documents into S3.
2. Create a Bedrock Knowledge Base over that data.
3. Query and observe grounded answers with citations.
4. Tune chunking and retrieval settings.

<strong>Takeaway:</strong> Knowledge Bases provide managed retrieval so answers are grounded and cited.`
                },
                {
                    title: "Lab: Bedrock Agents with Tools",
                    content: `<strong>Goal:</strong> build an agent that acts.

<strong>Steps:</strong>
1. Define an action group (an API/Lambda the agent can call).
2. Create a Bedrock Agent with instructions.
3. Attach the action group and a Knowledge Base.
4. Test a multi-step task.

<strong>Takeaway:</strong> Agents combine reasoning, tool use, and grounding to complete tasks.`
                },
                {
                    title: "Lab: Guardrails & Evaluation",
                    content: `<strong>Goal:</strong> make the app safe and measurable.

<strong>Steps:</strong>
1. Create a Guardrail with denied topics and a PII filter.
2. Attach it to your model calls.
3. Run an evaluation set and score groundedness/relevance.

<strong>Takeaway:</strong> Guardrails + evaluation are what make a generative app production-ready.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the Generative AI Developer - Professional focus areas.

<strong>Foundations & model selection:</strong> Modules 1-2 (foundation models, Bedrock, selection).
<strong>RAG & knowledge:</strong> Modules 3 + 7 (embeddings, vector stores, Knowledge Bases).
<strong>Agents & orchestration:</strong> Module 4 + Module 7 (Agents, tools, workflows).
<strong>Production quality:</strong> Module 5-6 (evaluation, Guardrails, observability, security).
<strong>Cost & performance:</strong> Module 6 (optimization).

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-3 + Knowledge Bases lab
• Week 3: Modules 4-6 + Agents and Guardrails labs
• Week 4: Module 7 + AWS practice questions

<strong>Pair with:</strong> AWS Skill Builder generative AI content and hands-on Bedrock practice.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `This is an advanced, production-focused exam.

<strong>During the exam:</strong>
• For grounding, think Knowledge Bases (managed RAG) and vector stores
• For actions, think Bedrock Agents with action groups
• For safety, think Guardrails on input and output
• For reliability, think evaluation, observability, and retries
• For cost/latency, prefer the smallest model and caching that meet targets

<strong>Mindset:</strong> you are the senior developer shipping a secure, reliable, cost-effective generative AI application.`
                }
            ]
        }
    ],

    // ==========================================================
    // Google Cloud Certified: Generative AI Leader
    // ==========================================================
    gcpGenAiLeader: [
        {
            number: "Gen AI Leader · Module 1",
            title: "Fundamentals of Generative AI",
            description: "Understand core AI, ML, and generative AI concepts and how large language models work.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is AI, ML & generative AI?",
                "Foundation models & how LLMs work",
                "Tokens, embeddings & context",
                "Capabilities and limitations"
            ],
            detailedDescription: "The Generative AI Leader certification starts with the concepts. This module covers the relationship between AI, ML, and generative AI, how large language models work, and the key capabilities and limitations business leaders must understand.",
            detailedContent: [
                {
                    title: "What is AI, ML & Generative AI?",
                    content: `<strong>Artificial Intelligence (AI)</strong> builds systems that perform tasks needing human intelligence.

<strong>The hierarchy:</strong>
• <strong>AI:</strong> The broad field
• <strong>Machine Learning:</strong> Systems that learn from data
• <strong>Deep Learning:</strong> ML with neural networks
• <strong>Generative AI:</strong> Creates new content with foundation models

<strong>Exam focus:</strong> The Generative AI Leader exam is business-oriented — know what generative AI <em>is</em> and where it delivers value, not how to code it.`
                },
                {
                    title: "Foundation Models & How LLMs Work",
                    content: `A <strong>foundation model</strong> is a large model pretrained on vast data that adapts to many tasks.

<strong>Large Language Models (LLMs):</strong>
• Trained to predict the next token
• Power chat, summarization, and generation
• Google's family is <strong>Gemini</strong> (multimodal: text, image, audio, video)

<strong>Multimodal models</strong> accept and produce multiple content types — a core Gemini strength.`
                },
                {
                    title: "Tokens, Embeddings & Context",
                    content: `Key vocabulary for working with generative AI:

• <strong>Tokens:</strong> Word pieces the model processes; usage is billed per token
• <strong>Context window:</strong> Max tokens considered at once (Gemini offers very large windows)
• <strong>Embeddings:</strong> Numeric vectors capturing meaning, used for search and RAG
• <strong>Temperature:</strong> Controls randomness of output

<strong>Why it matters:</strong> These concepts explain cost, capability, and how to ground models on your data.`,
                    code: `# Generate text with Gemini on Vertex AI (Python)
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Explain generative AI in one sentence.")
print(response.text)`
                },
                {
                    title: "Capabilities and Limitations",
                    content: `Leaders must understand both the power and the risks.

<strong>Capabilities:</strong>
• Content creation, summarization, Q&A
• Code generation and data analysis
• Multimodal understanding

<strong>Limitations:</strong>
• <strong>Hallucinations</strong> (confident but wrong)
• Knowledge cutoffs
• Bias from training data
• Non-determinism

<strong>Mitigations:</strong> grounding (RAG), human oversight, and evaluation — covered later in this track.`
                }
            ]
        },
        {
            number: "Gen AI Leader · Module 2",
            title: "Google Cloud's AI Offerings",
            description: "Explore Gemini, Vertex AI, Model Garden, and AI agents on Google Cloud.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Gemini models & Vertex AI",
                "Model Garden & pretrained APIs",
                "AI agents & Agentspace",
                "Gemini for Google Workspace"
            ],
            detailedDescription: "This module surveys Google Cloud's generative AI portfolio: the Gemini model family, the Vertex AI platform, Model Garden, prebuilt AI APIs, agent tooling, and Gemini across Google Workspace.",
            detailedContent: [
                {
                    title: "Gemini Models & Vertex AI",
                    content: `<strong>Vertex AI</strong> is Google Cloud's unified platform to build, deploy, and manage AI and ML.

<strong>For generative AI it provides:</strong>
• Access to <strong>Gemini</strong> models via API
• Vertex AI Studio for prompt design
• Grounding, tuning, and evaluation tools
• Enterprise security and data governance

<strong>Exam tip:</strong> Vertex AI is the one-stop platform; Gemini is the flagship multimodal model family.`,
                    code: `# Multimodal prompt: text + image with Gemini
import vertexai
from vertexai.generative_models import GenerativeModel, Part

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content([
    "Describe this diagram:",
    Part.from_uri("gs://my-bucket/architecture.png", mime_type="image/png"),
])
print(response.text)`
                },
                {
                    title: "Model Garden & Pretrained APIs",
                    content: `Not every problem needs a custom model.

<strong>Model Garden</strong> (in Vertex AI) is a catalog of Google, partner, and open models to discover, test, and deploy.

<strong>Prebuilt AI APIs:</strong>
• <strong>Cloud Vision API</strong> — image analysis
• <strong>Cloud Natural Language API</strong> — text analysis
• <strong>Speech-to-Text / Text-to-Speech</strong>
• <strong>Document AI</strong> — document data extraction
• <strong>Translation API</strong>

<strong>Tip:</strong> Prefer a prebuilt API when it solves the problem — faster and cheaper.`
                },
                {
                    title: "AI Agents & Agentspace",
                    content: `<strong>AI agents</strong> extend generative AI from answering to <em>acting</em> toward goals.

<strong>Google Cloud agent tooling:</strong>
• <strong>Vertex AI Agent Builder</strong> — build grounded search and conversational agents
• <strong>Google Agentspace</strong> — enterprise agents over company knowledge
• <strong>Gemini Enterprise / Agent Platform</strong> — the evolving agent stack

<strong>Use cases:</strong> customer support, enterprise search, and workflow automation.`
                },
                {
                    title: "Gemini for Google Workspace",
                    content: `<strong>Gemini for Google Workspace</strong> embeds generative AI directly into everyday apps.

<strong>Examples:</strong>
• Draft and summarize in Gmail and Docs
• Generate slides and images in Slides
• Analyze and build formulas in Sheets
• Take notes and summarize in Meet

<strong>Business value:</strong> productivity gains without building anything — a common leadership talking point on the exam.`
                }
            ]
        },
        {
            number: "Gen AI Leader · Module 3",
            title: "Improving Model Output",
            description: "Use prompt engineering, grounding, RAG, and tuning to get better, more reliable results.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prompt engineering",
                "Grounding & Retrieval Augmented Generation",
                "Fine-tuning & customization",
                "Embeddings & vector search"
            ],
            detailedDescription: "This module covers the techniques that make generative AI accurate and useful: prompt engineering, grounding responses with RAG, customizing models with tuning, and using embeddings for semantic search.",
            detailedContent: [
                {
                    title: "Prompt Engineering",
                    content: `<strong>Prompt engineering</strong> shapes model behavior through well-crafted instructions.

<strong>Techniques:</strong>
• <strong>Zero-shot:</strong> Ask directly
• <strong>Few-shot:</strong> Provide examples
• <strong>Chain-of-thought:</strong> Ask for step-by-step reasoning
• <strong>Role/system instructions:</strong> Set persona and constraints

<strong>Parameters:</strong> temperature and top-p tune creativity; lower values give more deterministic answers.`
                },
                {
                    title: "Grounding & Retrieval Augmented Generation",
                    content: `<strong>Grounding</strong> connects a model to trusted data so answers are accurate and current.

<strong>RAG pattern:</strong>
1. Retrieve relevant content (e.g., from Vertex AI Search)
2. Add it to the prompt as context
3. The model answers using that context

<strong>On Google Cloud:</strong> Vertex AI supports grounding with <strong>Google Search</strong> and with <strong>your own data</strong> via Vertex AI Search — reducing hallucinations.`,
                    code: `# Ground Gemini responses with Google Search (Vertex AI)
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash")

tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
response = model.generate_content(
    "What are the latest Vertex AI features?", tools=[tool])
print(response.text)`
                },
                {
                    title: "Fine-tuning & Customization",
                    content: `When prompting and RAG aren't enough, <strong>tune</strong> a model.

<strong>Options on Vertex AI:</strong>
• <strong>Supervised fine-tuning:</strong> Adapt with labeled examples
• <strong>Distillation:</strong> Train a smaller, cheaper model from a larger one

<strong>Trade-offs:</strong> Tuning needs quality data and cost; grounding (RAG) is usually cheaper and easier to keep current. Choose based on the need for style vs. fresh knowledge.`
                },
                {
                    title: "Embeddings & Vector Search",
                    content: `<strong>Embeddings</strong> turn text (or images) into vectors that capture meaning, enabling semantic search.

<strong>On Google Cloud:</strong>
• Generate embeddings with Vertex AI embedding models
• Store and query them with <strong>Vertex AI Vector Search</strong> (fast nearest-neighbor)

<strong>Why it matters:</strong> Vector search is the retrieval engine behind RAG and recommendation systems.`,
                    code: `# Generate text embeddings with Vertex AI
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-004")
embeddings = model.get_embeddings(["What is gradient descent?"])
print("Vector length:", len(embeddings[0].values))`
                }
            ]
        },
        {
            number: "Gen AI Leader · Module 4",
            title: "Business Strategy & Responsible AI",
            description: "Drive business value with generative AI while applying responsible AI, security, and governance.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Identifying business value",
                "Responsible AI principles",
                "Security & governance",
                "Driving AI adoption"
            ],
            detailedDescription: "The final Generative AI Leader domain is strategic. This module covers finding high-value use cases, applying Google's responsible AI principles, securing and governing AI, and leading successful adoption.",
            detailedContent: [
                {
                    title: "Identifying Business Value",
                    content: `Leaders must connect generative AI to measurable outcomes.

<strong>High-value patterns:</strong>
• Productivity (drafting, summarizing, coding)
• Customer experience (support agents, search)
• Knowledge access (enterprise RAG)
• Content and marketing generation

<strong>Frameworks:</strong> Prioritize use cases by value vs. feasibility, and define clear success metrics (ROI, time saved, CSAT).`
                },
                {
                    title: "Responsible AI Principles",
                    content: `Google's <strong>AI Principles</strong> guide responsible development.

<strong>Key ideas:</strong>
• Be socially beneficial
• Avoid unfair bias
• Be built and tested for safety
• Be accountable to people
• Incorporate privacy
• Uphold high standards of scientific excellence

<strong>On Google Cloud:</strong> Responsible AI tooling includes safety filters, citations/grounding, and model evaluation.`
                },
                {
                    title: "Security & Governance",
                    content: `Enterprise AI must be secure and governed.

<strong>Controls on Google Cloud:</strong>
• <strong>IAM</strong> for least-privilege access
• <strong>VPC Service Controls</strong> to prevent data exfiltration
• <strong>Customer-managed encryption keys (CMEK)</strong>
• Data residency and no-training guarantees for enterprise data

<strong>Governance:</strong> model registries, audit logging (Cloud Audit Logs), and policy controls support compliance.`
                },
                {
                    title: "Driving AI Adoption",
                    content: `Technology succeeds only when people adopt it.

<strong>Leadership levers:</strong>
• Executive sponsorship and a clear vision
• Upskilling and change management
• Start with pilots, measure, then scale
• Establish an AI governance/center of excellence

<strong>Exam focus:</strong> Expect scenario questions on how a leader should evaluate, pilot, and scale a generative AI initiative responsibly.`
                }
            ]
        },
        {
            number: "Gen AI Leader · Module 5",
            title: "AI Development Approaches, Data & the ML Lifecycle",
            description: "Choose the right build approach on Google Cloud, understand ML types and the lifecycle, and treat data as the foundation of trustworthy AI.",
            duration: "60 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Types of machine learning",
                "The ML lifecycle",
                "Data quality & governance",
                "Choosing a build approach on Google Cloud",
                "Agents deep dive",
                "Cost, ROI & sizing"
            ],
            detailedDescription: "The Generative AI Leader exam expects leaders to reason about how AI solutions are actually built and operated. This module deepens the core exam domains: the kinds of ML, the end-to-end ML lifecycle, why data quality and governance decide success, how to choose between pre-trained APIs, BigQuery ML, AutoML, custom training and Gemini, how agents work, and how to think about cost and ROI.",
            detailedContent: [
                {
                    title: "Types of Machine Learning",
                    content: `Leaders should recognize the three main learning styles and match them to business problems.

<strong>Supervised learning:</strong>
• Trained on labeled data
• Classification (categories) and regression (numbers)
• Example: churn prediction, demand forecasting

<strong>Unsupervised learning:</strong>
• Finds structure in unlabeled data
• Clustering and anomaly detection
• Example: customer segmentation

<strong>Reinforcement learning:</strong>
• Learns by trial, reward, and feedback
• Example: robotics, recommendation tuning

<strong>Exam tip:</strong> Generative AI is built on deep learning (a form of ML). Know where it sits and when a simpler ML approach is a better fit.`
                },
                {
                    title: "The ML Lifecycle",
                    content: `Every AI solution follows a lifecycle. Leaders fund and govern each stage.

<strong>Stages:</strong>
1. <strong>Business framing:</strong> Define the outcome and success metric
2. <strong>Data collection & preparation:</strong> Source, clean, and label data
3. <strong>Model development:</strong> Choose, train, or tune a model
4. <strong>Evaluation:</strong> Measure quality, safety, and fairness
5. <strong>Deployment:</strong> Serve the model (endpoint or app)
6. <strong>Monitoring:</strong> Watch for drift, cost, and safety in production

<strong>Why leaders care:</strong> Most failures come from weak framing or poor data, not the model. Budget for the full lifecycle, not just the build.`
                },
                {
                    title: "Data Quality & Governance",
                    content: `Data is the foundation of every AI outcome.

<strong>Data quality dimensions:</strong>
• Accuracy, completeness, consistency
• Timeliness and representativeness (avoid bias)

<strong>Governance on Google Cloud:</strong>
• <strong>Dataplex</strong> for data governance and cataloging
• <strong>BigQuery</strong> for governed analytics at scale
• Access control with <strong>IAM</strong>, encryption with <strong>CMEK</strong>
• Data residency and enterprise "no-training on your data" guarantees

<strong>Exam focus:</strong> Poor or biased data leads to poor or biased AI. Leaders must invest in data quality and governance before scaling.`
                },
                {
                    title: "Choosing a Build Approach on Google Cloud",
                    content: `Google Cloud offers a spectrum from no-code to full custom. Choosing well controls cost and time-to-value.

<strong>Options, simplest first:</strong>
• <strong>Pre-trained APIs</strong> (Vision, Natural Language, Speech, Document AI, Translation) — common tasks, no training
• <strong>Gemini via Vertex AI</strong> — generative and multimodal tasks with prompting/grounding
• <strong>BigQuery ML</strong> — models in SQL where data already lives
• <strong>AutoML</strong> — custom models with minimal code
• <strong>Custom training</strong> — full control for specialized needs

<strong>Decision rule:</strong> pick the simplest option that meets accuracy, control, and cost requirements.`,
                    code: `# Decision helper: match a scenario to the simplest Google Cloud approach
scenarios = {
    "Detect objects in product photos":      "Pre-trained Vision API",
    "Draft and summarize marketing copy":    "Gemini on Vertex AI",
    "Predict churn from a BigQuery table":   "BigQuery ML",
    "Custom image classifier, little code":  "Vertex AI AutoML",
    "Novel model with special architecture": "Custom training on Vertex AI",
}
for scenario, approach in scenarios.items():
    print(f"{scenario:38} -> {approach}")`
                },
                {
                    title: "Agents Deep Dive",
                    content: `Agents move generative AI from answering to <em>acting</em> toward goals — a growing exam focus.

<strong>What makes an agent:</strong>
• A model with instructions (its role)
• Tools it can call (search, functions, APIs)
• Memory of the interaction
• A loop that plans, acts, and observes

<strong>On Google Cloud:</strong>
• <strong>Vertex AI Agent Builder</strong> — grounded search and conversational agents
• <strong>Google Agentspace</strong> — enterprise agents over company knowledge
• <strong>Gemini Enterprise</strong> — the evolving enterprise agent stack

<strong>Business value:</strong> agents automate multi-step workflows (support, research, operations), not just single questions.`
                },
                {
                    title: "Cost, ROI & Sizing",
                    content: `Leaders are accountable for value, not just capability.

<strong>Cost drivers:</strong>
• Tokens processed (input + output)
• Model choice (Flash vs Pro tiers)
• Grounding/retrieval and storage
• Volume and peak traffic

<strong>Levers to control cost:</strong>
• Use smaller/faster models where quality allows
• Cache frequent responses and embeddings
• Trim prompts and context length
• Prefer prompting/RAG before expensive tuning

<strong>ROI framing:</strong> tie each use case to a measurable outcome (time saved, deflection rate, revenue) and compare against total cost of ownership.`
                }
            ]
        },
        {
            number: "Gen AI Leader · Module 6",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided hands-on labs in Vertex AI Studio plus a domain-by-domain study plan and exam-day strategy.",
            duration: "60 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Prompting in Vertex AI Studio",
                "Lab: Grounding a Gemini response",
                "Lab: Gemini in Google Workspace",
                "Lab: Embeddings & Vector Search",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Concepts alone do not pass exams. This module turns theory into practice with four guided labs you can run in a free Google Cloud project, then maps your study to the official exam domains and gives a practical exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Prompting in Vertex AI Studio",
                    content: `<strong>Goal:</strong> experience prompting, system instructions, and temperature firsthand.

<strong>Steps:</strong>
1. Open the Google Cloud console and go to Vertex AI Studio.
2. Start a new prompt with the Gemini Flash model.
3. Add a system instruction: "You are a concise business analyst."
4. Ask the same question at temperature 0.1 and 0.9 and compare.
5. Add a few-shot example and observe the format change.

<strong>What to notice:</strong> lower temperature is more deterministic; system instructions and examples strongly shape output.`,
                    code: `# The same prompt in code (Vertex AI, Python)
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="You are a concise business analyst.")

for temp in (0.1, 0.9):
    resp = model.generate_content(
        "Give one risk of adopting generative AI.",
        generation_config=GenerationConfig(temperature=temp))
    print(temp, "->", resp.text)`
                },
                {
                    title: "Lab: Grounding a Gemini Response",
                    content: `<strong>Goal:</strong> see how grounding reduces hallucinations.

<strong>Steps:</strong>
1. Ask Gemini a question about very recent events without grounding — note any uncertainty.
2. Enable grounding with Google Search (a tool in Vertex AI).
3. Ask again and compare accuracy and the presence of citations.

<strong>Takeaway:</strong> grounding connects the model to trusted, current data. For enterprise data, ground with Vertex AI Search over your own documents.`,
                    code: `# Grounding with Google Search (Vertex AI, Python)
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash")
tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

resp = model.generate_content(
    "Summarize the newest Vertex AI features.", tools=[tool])
print(resp.text)`
                },
                {
                    title: "Lab: Gemini in Google Workspace",
                    content: `<strong>Goal:</strong> connect the business value story to real productivity.

<strong>Steps (with Gemini for Workspace enabled):</strong>
1. In Gmail, use "Help me write" to draft a reply from a short prompt.
2. In Docs, summarize a long document.
3. In Sheets, ask Gemini to suggest a formula.
4. In Slides, generate an image for a slide.

<strong>Leadership framing:</strong> most organizations get first value from embedded AI (Workspace) before building custom solutions. This is a common exam talking point.`
                },
                {
                    title: "Lab: Embeddings & Vector Search",
                    content: `<strong>Goal:</strong> understand the retrieval engine behind RAG.

<strong>Steps:</strong>
1. Generate embeddings for a handful of sentences.
2. Compare two similar sentences vs two unrelated ones.
3. Observe that similar meanings produce closer vectors.

<strong>Takeaway:</strong> embeddings + Vertex AI Vector Search power semantic search and grounding. This is why RAG can find relevant context even when keywords differ.`,
                    code: `# Compare semantic similarity with Vertex AI embeddings
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-004")
texts = ["How do I reset my password?",
         "I forgot my login credentials",
         "What is the capital of France?"]
vectors = [e.values for e in model.get_embeddings(texts)]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    return dot / (na * nb)

print("similar:", round(cosine(vectors[0], vectors[1]), 3))
print("unrelated:", round(cosine(vectors[0], vectors[2]), 3))`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official Generative AI Leader domains.

<strong>Domain 1 - Fundamentals of Gen AI:</strong> Modules 1 and 5 (ML types, LLMs, tokens, limitations).
<strong>Domain 2 - Google Cloud's Gen AI offerings:</strong> Module 2 and Module 5 (Gemini, Vertex AI, Model Garden, APIs, agents, Workspace).
<strong>Domain 3 - Improving model output:</strong> Module 3 and Module 6 labs (prompting, grounding/RAG, tuning, embeddings).
<strong>Domain 4 - Business strategy & responsible AI:</strong> Modules 4 and 5 (value, ROI, responsible AI, security, adoption).

<strong>Recommended plan:</strong>
• Week 1: Modules 1-2 + Lab 1
• Week 2: Modules 3 + 5 + Labs 2 and 4
• Week 3: Modules 4 + 6 + Google's official Skill Badges and practice assessment
• Week 4: Review + full practice test

<strong>Pair with:</strong> Google Cloud Skills Boost labs and the official practice assessment for full readiness.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `The Generative AI Leader exam is scenario- and business-focused.

<strong>During the exam:</strong>
• Read the scenario for the business goal and constraint first
• Prefer the simplest Google Cloud service that meets the need
• When unsure, choose the option that improves accuracy responsibly (grounding, human oversight)
• Watch for "cost-effective," "fastest to value," or "least management" qualifiers - they point to the answer
• Eliminate options that ignore responsible AI, security, or data governance

<strong>Mindset:</strong> you are answering as a leader guiding responsible, valuable adoption - not as an engineer optimizing code.`
                }
            ]
        }
    ],

    // ==========================================================
    // Google Cloud Certified: Professional Machine Learning Engineer
    // ==========================================================
    gcpMlEngineer: [
        {
            number: "GCP PMLE · Module 1",
            title: "Architecting Low-Code AI Solutions",
            description: "Build AI solutions quickly with BigQuery ML, pretrained APIs, and AutoML on Vertex AI.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Choosing a build approach",
                "BigQuery ML",
                "Pretrained AI APIs",
                "AutoML on Vertex AI"
            ],
            detailedDescription: "The Professional ML Engineer exam starts with low-code options. This module covers choosing between prebuilt APIs, BigQuery ML, AutoML, and custom training, and when each is the right tool.",
            detailedContent: [
                {
                    title: "Choosing a Build Approach",
                    content: `Google Cloud offers a spectrum from no-code to full custom.

<strong>The options:</strong>
• <strong>Pretrained APIs:</strong> Vision, NLP, Speech — no training
• <strong>BigQuery ML:</strong> Train models with SQL on data in BigQuery
• <strong>AutoML:</strong> Train custom models with minimal code
• <strong>Custom training:</strong> Full control with your own code

<strong>Exam tip:</strong> Pick the simplest approach that meets accuracy and control requirements to minimize cost and effort.`
                },
                {
                    title: "BigQuery ML",
                    content: `<strong>BigQuery ML</strong> lets you build and run models using SQL, directly where your data lives.

<strong>Supported models:</strong>
• Linear & logistic regression
• Boosted trees (XGBoost)
• K-means clustering
• Time-series (ARIMA+)
• Deep neural networks and imported models

<strong>Benefit:</strong> No data movement and no separate ML infrastructure — great for analysts and fast baselines.`,
                    code: `-- Train a logistic regression model with BigQuery ML
CREATE OR REPLACE MODEL \`mydataset.churn_model\`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['churned']) AS
SELECT tenure, monthly_charges, contract_type, churned
FROM \`mydataset.customers\`;

-- Evaluate the model
SELECT * FROM ML.EVALUATE(MODEL \`mydataset.churn_model\`);`
                },
                {
                    title: "Pretrained AI APIs",
                    content: `<strong>Pretrained APIs</strong> add AI with a single call — no training or ML expertise.

<strong>Key APIs:</strong>
• <strong>Cloud Vision:</strong> Labels, OCR, faces
• <strong>Cloud Natural Language:</strong> Sentiment, entities
• <strong>Speech-to-Text / Text-to-Speech</strong>
• <strong>Translation</strong>
• <strong>Document AI:</strong> Structured data from documents

<strong>Tip:</strong> Choose these when your problem is common and you don't need a custom model.`,
                    code: `# Analyze sentiment with the Cloud Natural Language API
from google.cloud import language_v1

client = language_v1.LanguageServiceClient()
document = language_v1.Document(
    content="This ML course is excellent and easy to follow!",
    type_=language_v1.Document.Type.PLAIN_TEXT,
)
sentiment = client.analyze_sentiment(document=document).document_sentiment
print("Score:", sentiment.score, "Magnitude:", sentiment.magnitude)`
                },
                {
                    title: "AutoML on Vertex AI",
                    content: `<strong>Vertex AI AutoML</strong> trains high-quality custom models with minimal code.

<strong>Supported data types:</strong>
• Tabular (classification, regression, forecasting)
• Image (classification, object detection)
• Text (classification, entity extraction)
• Video

<strong>Workflow:</strong> Create a managed dataset → train AutoML → evaluate → deploy to an endpoint. AutoML handles feature engineering and model search.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 2",
            title: "Managing Data & Models",
            description: "Prepare data, manage features, and govern models with Vertex AI Feature Store and Model Registry.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Managed datasets",
                "Data preparation & pipelines",
                "Vertex AI Feature Store",
                "Model Registry & versioning"
            ],
            detailedDescription: "This module covers collaborating to manage data and models: creating managed datasets, preparing data at scale, reusing features with Feature Store, and governing models with the Vertex AI Model Registry.",
            detailedContent: [
                {
                    title: "Managed Datasets",
                    content: `<strong>Vertex AI managed datasets</strong> centralize data with labels, splits, and lineage.

<strong>Benefits:</strong>
• Consistent train/validation/test splits
• Data lineage and versioning
• Integration with AutoML and custom training

<strong>Sources:</strong> BigQuery, Cloud Storage, and more. Managed datasets keep experiments reproducible.`
                },
                {
                    title: "Data Preparation & Pipelines",
                    content: `Quality data drives quality models.

<strong>Tools on Google Cloud:</strong>
• <strong>BigQuery:</strong> SQL-based transformation at scale
• <strong>Dataflow:</strong> Batch and streaming data processing (Apache Beam)
• <strong>Dataproc:</strong> Managed Spark/Hadoop
• <strong>Dataprep:</strong> Visual data cleaning

<strong>Consistency:</strong> Apply the same transformations at training and serving to prevent skew.`,
                    code: `# Read a training table from BigQuery into a DataFrame
from google.cloud import bigquery

client = bigquery.Client()
df = client.query('''
    SELECT tenure, monthly_charges, contract_type, churned
    FROM \`mydataset.customers\`
''').to_dataframe()
print(df.shape)`
                },
                {
                    title: "Vertex AI Feature Store",
                    content: `<strong>Vertex AI Feature Store</strong> centralizes features for reuse and consistency.

<strong>Benefits:</strong>
• Share features across teams and models
• Serve features online (low latency) and offline (training)
• Eliminate training/serving skew
• Track feature lineage and monitoring

<strong>Tip:</strong> Store engineered features once and reuse them, rather than recomputing per project.`
                },
                {
                    title: "Model Registry & Versioning",
                    content: `The <strong>Vertex AI Model Registry</strong> is the central place to manage model versions and lifecycle.

<strong>Capabilities:</strong>
• Register and version models
• Track lineage back to training runs
• Manage aliases (e.g., "production")
• Deploy directly to endpoints

<strong>Governance:</strong> The registry supports approvals and reproducibility for MLOps.`,
                    code: `# Register a model in the Vertex AI Model Registry
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")
model = aiplatform.Model.upload(
    display_name="churn-model",
    artifact_uri="gs://my-bucket/models/churn/",
    serving_container_image_uri=(
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"),
)
print("Registered:", model.resource_name)`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 3",
            title: "Scaling Prototypes into ML Models",
            description: "Move from notebooks to scalable custom training on Vertex AI.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Custom training on Vertex AI",
                "Distributed training & hardware",
                "Hyperparameter tuning",
                "Frameworks (TensorFlow / PyTorch)"
            ],
            detailedDescription: "This module covers scaling prototypes into production models: running custom training jobs on Vertex AI, using distributed training and accelerators, tuning hyperparameters, and working with major frameworks.",
            detailedContent: [
                {
                    title: "Custom Training on Vertex AI",
                    content: `<strong>Vertex AI custom training</strong> runs your training code on managed, scalable infrastructure.

<strong>You provide:</strong>
• A training script or container
• Machine type and accelerators
• Input data (Cloud Storage / BigQuery)

Vertex AI provisions compute, runs the job, saves artifacts, and tears down — you pay only for what you use.`,
                    code: `# Launch a custom training job on Vertex AI
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1",
                staging_bucket="gs://my-bucket")

job = aiplatform.CustomTrainingJob(
    display_name="train-churn",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-13:latest",
    requirements=["scikit-learn"],
)
model = job.run(replica_count=1, machine_type="n1-standard-4")`
                },
                {
                    title: "Distributed Training & Hardware",
                    content: `Large models and datasets need more than one machine.

<strong>Scaling options:</strong>
• <strong>Multi-worker</strong> distributed training
• <strong>GPUs</strong> for deep learning
• <strong>TPUs</strong> — Google's custom accelerators for large-scale training
• <strong>Reduction Server</strong> to speed up gradient aggregation

<strong>Tip:</strong> Match hardware to the workload — TPUs excel at large neural networks; CPUs suffice for classic ML.`
                },
                {
                    title: "Hyperparameter Tuning",
                    content: `<strong>Vertex AI hyperparameter tuning</strong> searches parameter combinations to optimize a metric.

<strong>Configure:</strong>
• Parameters and ranges
• The metric to optimize (e.g., accuracy)
• Search algorithm (Bayesian by default)
• Trial and parallelism limits

<strong>Benefit:</strong> Automated, parallel search finds better models faster than manual tuning.`
                },
                {
                    title: "Frameworks (TensorFlow / PyTorch)",
                    content: `Vertex AI supports the major ML frameworks.

<strong>Options:</strong>
• <strong>TensorFlow</strong> and <strong>Keras</strong>
• <strong>PyTorch</strong>
• <strong>scikit-learn</strong> and <strong>XGBoost</strong>
• Custom containers for anything else

<strong>Prebuilt containers</strong> for training and serving simplify setup, while custom containers give full control.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 4",
            title: "Serving & Scaling Models",
            description: "Deploy models to Vertex AI endpoints for real-time and batch prediction.",
            duration: "50 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Online (real-time) prediction",
                "Batch prediction",
                "Scaling & optimization",
                "Testing deployments"
            ],
            detailedDescription: "This module covers serving and scaling models: deploying to online endpoints for real-time inference, running batch predictions, optimizing for latency and cost, and testing deployments.",
            detailedContent: [
                {
                    title: "Online (Real-time) Prediction",
                    content: `<strong>Vertex AI endpoints</strong> host models behind an HTTPS API for low-latency, real-time prediction.

<strong>Steps:</strong>
1. Upload/register the model
2. Create an endpoint
3. Deploy the model with a machine type
4. Split traffic across model versions

<strong>Traffic splitting</strong> enables safe A/B tests and gradual rollouts.`,
                    code: `# Deploy a registered model to an online endpoint
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")
model = aiplatform.Model("projects/.../models/churn-model")

endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,   # auto-scaling
)
print(endpoint.predict(instances=[[12, 79.9, "month-to-month"]]))`
                },
                {
                    title: "Batch Prediction",
                    content: `<strong>Batch prediction</strong> scores large datasets asynchronously — no endpoint required.

<strong>When to use:</strong>
• Millions of records on a schedule
• No need for instant responses
• Cost efficiency (compute runs only for the job)

<strong>Inputs/outputs:</strong> Read from and write to BigQuery or Cloud Storage.`
                },
                {
                    title: "Scaling & Optimization",
                    content: `Balance latency, throughput, and cost.

<strong>Levers:</strong>
• <strong>Auto-scaling</strong> replicas on traffic
• Right-size machine types and accelerators
• Use GPUs only when needed
• Optimize models (quantization, distillation)

<strong>Cost tip:</strong> Set min replicas appropriately and scale to demand to avoid paying for idle capacity.`
                },
                {
                    title: "Testing Deployments",
                    content: `Validate a deployment before sending production traffic.

<strong>How:</strong>
• Send sample requests and check outputs
• Verify response schema and latency
• Use traffic splitting for canary/A-B tests
• Review prediction logs for errors

<strong>Tip:</strong> Keep a small held-out sample as a post-deployment smoke test.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 5",
            title: "ML Pipelines & Automation",
            description: "Automate and orchestrate the ML lifecycle with Vertex AI Pipelines and CI/CD.",
            duration: "55 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Vertex AI Pipelines",
                "Building reusable components",
                "CI/CD for ML",
                "Orchestration & scheduling"
            ],
            detailedDescription: "This module covers automating and orchestrating ML workflows: building pipelines with Vertex AI Pipelines, creating reusable components, adding CI/CD, and scheduling retraining.",
            detailedContent: [
                {
                    title: "Vertex AI Pipelines",
                    content: `<strong>Vertex AI Pipelines</strong> orchestrate ML steps into a repeatable, serverless workflow (built on Kubeflow Pipelines / TFX).

<strong>Typical steps:</strong>
• Data prep
• Training
• Evaluation (conditional)
• Model registration
• Deployment

<strong>Benefits:</strong> Reproducibility, lineage/metadata tracking, and step caching.`,
                    code: `# Define a minimal Vertex AI (KFP) pipeline
from kfp import dsl

@dsl.pipeline(name="churn-training-pipeline")
def pipeline(project: str):
    prep = preprocess_op()
    train = train_op(data=prep.outputs["dataset"])
    _ = deploy_op(model=train.outputs["model"])

# Compile & submit with google.cloud.aiplatform.PipelineJob(...)`
                },
                {
                    title: "Building Reusable Components",
                    content: `A <strong>component</strong> is a self-contained, versioned pipeline step (code + inputs + outputs + container).

<strong>Why components:</strong>
• Write once, reuse across pipelines
• Test and version steps independently
• Share via artifact registries

<strong>Tip:</strong> Treat components like functions for your ML workflows.`
                },
                {
                    title: "CI/CD for ML",
                    content: `<strong>CI/CD</strong> automates testing and releasing ML code and models.

<strong>Google Cloud tools:</strong>
• <strong>Cloud Build:</strong> Build, test, and trigger pipelines
• <strong>Artifact Registry:</strong> Store containers and components
• <strong>Model Registry:</strong> Version and approve models

<strong>Flow:</strong> commit → build/test → run pipeline → register → deploy, with approvals for production.`
                },
                {
                    title: "Orchestration & Scheduling",
                    content: `Automate when pipelines run.

<strong>Options:</strong>
• <strong>Scheduled pipelines</strong> (cron) for periodic retraining
• <strong>Event triggers</strong> (e.g., new data in Cloud Storage via Eventarc)
• <strong>Cloud Composer</strong> (managed Airflow) for complex DAGs

<strong>Use case:</strong> Retrain weekly on fresh data and auto-register the new model version.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 6",
            title: "Monitoring AI Solutions",
            description: "Monitor deployed models for drift and quality, and apply responsible AI.",
            duration: "45 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Model monitoring",
                "Detecting drift & retraining",
                "Explainability & responsible AI",
                "Logging, security & cost"
            ],
            detailedDescription: "The final PMLE domain covers keeping AI solutions healthy: monitoring models for drift and quality, retraining, applying explainability and responsible AI, and managing logging, security, and cost.",
            detailedContent: [
                {
                    title: "Model Monitoring",
                    content: `<strong>Vertex AI Model Monitoring</strong> watches deployed models for issues.

<strong>What it detects:</strong>
• <strong>Training/serving skew:</strong> Serving data differs from training data
• <strong>Prediction drift:</strong> Input distributions change over time
• Feature attribution shifts

Alerts fire when metrics breach thresholds, integrated with Cloud Monitoring.`
                },
                {
                    title: "Detecting Drift & Retraining",
                    content: `<strong>Drift</strong> degrades accuracy as live data diverges from training data.

<strong>Response loop:</strong>
1. Monitoring flags drift
2. Trigger a retraining pipeline (Vertex AI Pipelines)
3. Evaluate and register the new model
4. Deploy with traffic splitting

<strong>This closes the MLOps loop</strong>, keeping models accurate in production.`
                },
                {
                    title: "Explainability & Responsible AI",
                    content: `<strong>Vertex Explainable AI</strong> shows which features drove a prediction.

<strong>Techniques:</strong>
• Feature attributions (integrated gradients, sampled Shapley)
• The <strong>What-If Tool</strong> for interactive analysis

<strong>Responsible AI:</strong> Check for bias, document models, and keep humans accountable — aligned with Google's AI Principles.`,
                    code: `# Request predictions with feature attributions (Explainable AI)
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint("projects/.../endpoints/123")
response = endpoint.explain(instances=[[12, 79.9, "month-to-month"]])
print("Prediction:", response.predictions)
print("Attributions:", response.explanations)`
                },
                {
                    title: "Logging, Security & Cost",
                    content: `Operate AI solutions reliably and safely.

<strong>Observability:</strong>
• <strong>Cloud Logging</strong> and <strong>Cloud Monitoring</strong> for metrics, logs, and alerts
• <strong>Cloud Audit Logs</strong> for who-did-what auditing

<strong>Security:</strong> IAM least privilege, VPC Service Controls, and CMEK encryption.

<strong>Cost:</strong> Right-size compute, use auto-scaling, and prefer batch prediction for bulk scoring.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 5",
            title: "ML Pipelines, MLOps & Automation",
            description: "Operationalize ML with Vertex AI Pipelines, CI/CD, and automated retraining for reproducible, production-grade workflows.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Vertex AI Pipelines",
                "Reusable components",
                "CI/CD for ML",
                "Automated retraining",
                "Metadata & lineage"
            ],
            detailedDescription: "MLOps is heavily weighted on the Professional ML Engineer exam. This module covers building reproducible pipelines with Vertex AI Pipelines, packaging reusable components, automating with CI/CD, triggering retraining, and tracking lineage with Vertex ML Metadata.",
            detailedContent: [
                {
                    title: "Vertex AI Pipelines",
                    content: `<strong>Vertex AI Pipelines</strong> orchestrate ML workflows as reproducible, containerized steps (built on Kubeflow Pipelines or TFX).

<strong>Benefits:</strong>
• Reproducibility and versioning
• Step caching to skip unchanged work
• Parameterized, schedulable runs
• Clear lineage from data to model

<strong>Exam tip:</strong> Pipelines are the standard way to make training repeatable and production-ready on Google Cloud.`,
                    code: `# Define and run a simple Vertex AI pipeline (KFP v2)
from kfp import dsl, compiler
from google.cloud import aiplatform

@dsl.pipeline(name="train-churn")
def pipeline(data_uri: str):
    prep = preprocess_op(data_uri=data_uri)
    train_model_op(train_data=prep.outputs["out_data"])

compiler.Compiler().compile(pipeline, "pipeline.json")
aiplatform.PipelineJob(
    display_name="train-churn",
    template_path="pipeline.json",
    parameter_values={"data_uri": "gs://bucket/data.csv"},
).run()`
                },
                {
                    title: "Reusable Components",
                    content: `A <strong>component</strong> is a self-contained pipeline step (code + inputs + outputs + container).

<strong>Why components:</strong>
• Write a step once, reuse across pipelines
• Version and test steps independently
• Share via a components registry

<strong>Exam tip:</strong> Prebuilt Google Cloud Pipeline Components exist for common Vertex AI tasks (training, batch prediction, model upload).`
                },
                {
                    title: "CI/CD for ML",
                    content: `<strong>CI/CD</strong> automates testing and deployment of ML code, pipelines, and models.

<strong>Typical stages:</strong>
• CI: lint, unit test, validate pipeline on commit
• CD: run the pipeline, register the model, deploy to an endpoint

<strong>Tools:</strong> Cloud Build (or GitHub Actions) triggers pipelines; approvals gate production. This is the backbone of reliable MLOps.`
                },
                {
                    title: "Automated Retraining",
                    content: `Models degrade as data changes, so retraining should be automated.

<strong>Triggers:</strong>
• Schedule (e.g., weekly)
• New data arriving (event-based)
• Drift detected by Model Monitoring

<strong>Loop:</strong> monitor → detect drift → trigger pipeline → retrain → evaluate → deploy. Closing this loop is what "operationalizing ML" means on the exam.`
                },
                {
                    title: "Metadata & Lineage",
                    content: `<strong>Vertex ML Metadata</strong> records artifacts, executions, and their relationships.

<strong>Why it matters:</strong>
• Trace a deployed model back to its data and code
• Compare runs and reproduce results
• Support audits and governance

<strong>Exam tip:</strong> lineage and reproducibility are recurring themes - know that ML Metadata underpins them.`
                }
            ]
        },
        {
            number: "GCP PMLE · Module 6",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Vertex AI labs plus a domain-mapped study plan and exam-day strategy for the Professional ML Engineer exam.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: BigQuery ML model",
                "Lab: Custom training & endpoint",
                "Lab: Pipeline & monitoring",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn PMLE concepts into practice with guided labs in a Google Cloud project, then map your study to the official exam domains and apply an exam-day strategy for this scenario-heavy exam.",
            detailedContent: [
                {
                    title: "Lab: BigQuery ML Model",
                    content: `<strong>Goal:</strong> train and evaluate a model with SQL.

<strong>Steps:</strong>
1. Open BigQuery and load or select a public dataset.
2. Create a logistic regression model with CREATE MODEL.
3. Evaluate with ML.EVALUATE and inspect metrics.
4. Predict with ML.PREDICT.

<strong>Takeaway:</strong> BigQuery ML delivers fast baselines with no data movement.`,
                    code: `-- Train and evaluate a model in BigQuery ML
CREATE OR REPLACE MODEL \`ds.churn\`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['churned']) AS
SELECT tenure, monthly_charges, contract_type, churned
FROM \`ds.customers\`;

SELECT * FROM ML.EVALUATE(MODEL \`ds.churn\`);`
                },
                {
                    title: "Lab: Custom Training & Endpoint",
                    content: `<strong>Goal:</strong> train a custom model and serve it.

<strong>Steps:</strong>
1. Package a training script and run a Vertex AI custom training job.
2. Register the resulting model in the Model Registry.
3. Deploy the model to an endpoint.
4. Send a test prediction request.

<strong>Takeaway:</strong> the train → register → deploy → predict flow is the core of custom ML on Vertex AI.`
                },
                {
                    title: "Lab: Pipeline & Monitoring",
                    content: `<strong>Goal:</strong> automate and observe.

<strong>Steps:</strong>
1. Compile and run a simple Vertex AI Pipeline.
2. Schedule it to run on a recurrence.
3. Enable Model Monitoring on the deployed endpoint.
4. Review drift and prediction metrics.

<strong>Takeaway:</strong> pipelines + monitoring close the MLOps loop for reliable production models.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the official Professional ML Engineer domains.

<strong>Architecting low-code AI:</strong> Module 1 (APIs, BQML, AutoML).
<strong>Collaborating & managing data/models:</strong> Module 2 (datasets, Feature Store, registry).
<strong>Scaling prototypes:</strong> Module 3 (custom training, distributed, tuning).
<strong>Serving & scaling:</strong> Module 4 (endpoints, batch, monitoring, security, cost).
<strong>Automating & orchestrating (MLOps):</strong> Module 5 + Module 6 labs (pipelines, CI/CD, retraining).

<strong>Recommended plan:</strong>
• Weeks 1-2: Modules 1-3 + BQML and custom training labs
• Week 3: Modules 4-5 + pipeline/monitoring lab
• Week 4: Module 6 + Google Cloud Skills Boost labs + practice exam

<strong>Pair with:</strong> Google Cloud Skills Boost hands-on labs and the official sample questions.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `PMLE is scenario-heavy and technical.

<strong>During the exam:</strong>
• Read for constraints: cost, latency, scale, team skill, and data location
• Prefer the lowest-effort option that meets requirements (API > BQML > AutoML > custom)
• For production ML, expect pipelines, monitoring, and retraining in the answer
• Watch for training/serving skew, drift, and security (IAM, VPC-SC, CMEK) cues
• Eliminate options that ignore reproducibility or monitoring

<strong>Mindset:</strong> you are the ML engineer designing a reliable, cost-effective, production system.`
                }
            ]
        }
    ],

    // ==========================================================
    // Salesforce Certified Agentforce Specialist
    // ==========================================================
    salesforceAgentforce: [
        {
            number: "Agentforce · Module 1",
            title: "Agentforce Concepts",
            description: "Understand Agentforce agents and how topics and actions drive their behavior.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is Agentforce?",
                "Topics & classification",
                "Actions & the agent loop"
            ],
            detailedDescription: "This domain introduces Agentforce, Salesforce's platform for building autonomous AI agents. You learn what agents are, how topics organize an agent's jobs, and how actions let agents get work done on the Salesforce Platform.",
            detailedContent: [
                {
                    title: "What is Agentforce?",
                    content: `<strong>Agentforce</strong> is Salesforce's platform for building and deploying autonomous AI agents that reason and act on the Salesforce Platform.

<strong>Key ideas:</strong>
• Agents combine LLM reasoning with trusted Salesforce data and actions
• Built on the <strong>Atlas Reasoning Engine</strong>
• Deployed across Service, Sales, and custom experiences

<strong>Exam tip:</strong> An agent decides <em>what</em> to do; topics and actions define <em>how</em>.`
                },
                {
                    title: "Topics & Classification",
                    content: `A <strong>topic</strong> is a job an agent can handle (e.g., "Order Status").

<strong>Each topic includes:</strong>
• <strong>Scope & instructions:</strong> What it does and how to behave
• <strong>Actions:</strong> The tools it can use
• <strong>Example utterances:</strong> Help the agent classify user intent

<strong>Classification:</strong> The agent matches a user's request to the most relevant topic before choosing actions.`
                },
                {
                    title: "Actions & the Agent Loop",
                    content: `<strong>Actions</strong> are what an agent can actually do.

<strong>Action types:</strong>
• <strong>Flows</strong> and <strong>Apex</strong>
• <strong>Prompt templates</strong>
• <strong>Standard/custom</strong> Agentforce actions
• API/external calls

<strong>Reasoning loop:</strong> classify topic → select action → gather inputs → execute → respond, repeating as needed.`
                }
            ]
        },
        {
            number: "Agentforce · Module 2",
            title: "Prompt Engineering & Prompt Builder",
            description: "Design grounded, reusable prompts with Prompt Builder and dynamic grounding.",
            duration: "65 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prompt Builder & templates",
                "Grounding with data",
                "Prompt template types",
                "Testing & refining prompts"
            ],
            detailedDescription: "The largest Agentforce Specialist domain covers prompt engineering with Prompt Builder: creating reusable prompt templates, grounding them in Salesforce data, choosing the right template type, and testing prompts for quality and safety.",
            detailedContent: [
                {
                    title: "Prompt Builder & Templates",
                    content: `<strong>Prompt Builder</strong> is the low-code tool for creating reusable prompt templates in Salesforce.

<strong>A prompt template:</strong>
• Contains instructions plus merge fields
• Is grounded in live CRM data at runtime
• Can be reused across records and actions

<strong>Best practice:</strong> Be specific, give context, and state the desired output format.`
                },
                {
                    title: "Grounding with Data",
                    content: `<strong>Grounding</strong> injects trusted data into a prompt so responses are accurate and relevant.

<strong>Grounding sources:</strong>
• Record fields via merge fields
• Related lists and flows
• <strong>Data Cloud</strong> retrievers
• Apex-provided data

<strong>Why it matters:</strong> Grounding reduces hallucinations by keeping the model anchored to real CRM data.`
                },
                {
                    title: "Prompt Template Types",
                    content: `Choose the template type that fits the use case.

<strong>Common types:</strong>
• <strong>Sales Email:</strong> Personalized outreach
• <strong>Field Generation:</strong> Populate a record field
• <strong>Record Summary:</strong> Summarize a record
• <strong>Flex:</strong> Flexible, custom grounding for agents and more

<strong>Exam tip:</strong> Flex templates are the most versatile and commonly used with Agentforce actions.`
                },
                {
                    title: "Testing & Refining Prompts",
                    content: `Iterate on prompts before deploying them.

<strong>How:</strong>
• Preview with real records in Prompt Builder
• Inspect the resolved (grounded) prompt
• Check tone, accuracy, and format
• Watch the <strong>Einstein Trust Layer</strong> for masking and toxicity checks

<strong>Tip:</strong> Small wording changes can significantly change output quality.`
                }
            ]
        },
        {
            number: "Agentforce · Module 3",
            title: "Agentforce and Data Cloud",
            description: "Ground agents in enterprise knowledge using Data Cloud and retrievers (RAG).",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Data Cloud essentials",
                "Retrievers & RAG grounding",
                "Search indexes & knowledge"
            ],
            detailedDescription: "This domain covers grounding agents in enterprise data with Data Cloud: unifying data, building retrievers for Retrieval-Augmented Generation, and using search indexes and knowledge to return accurate, cited answers.",
            detailedContent: [
                {
                    title: "Data Cloud Essentials",
                    content: `<strong>Data Cloud</strong> unifies data from many sources into a single, real-time customer view.

<strong>Building blocks:</strong>
• <strong>Data streams</strong> and <strong>data lake objects</strong>
• <strong>Data model objects (DMOs)</strong>
• <strong>Identity resolution</strong> for unified profiles

<strong>Role in Agentforce:</strong> Provides trusted, grounded data for prompts and agent actions.`
                },
                {
                    title: "Retrievers & RAG Grounding",
                    content: `<strong>Retrieval-Augmented Generation (RAG)</strong> grounds answers in your own knowledge.

<strong>Retrievers:</strong>
• Search a vector/search index for relevant content
• Return the most relevant chunks to the prompt
• Keep responses accurate and current

<strong>Flow:</strong> user question → retriever finds relevant data → grounded prompt → trustworthy answer.`
                },
                {
                    title: "Search Indexes & Knowledge",
                    content: `Agents answer better when connected to curated knowledge.

<strong>Sources:</strong>
• <strong>Salesforce Knowledge</strong> articles
• Data Cloud <strong>search indexes</strong> (semantic search)
• Unstructured content (PDFs, docs) ingested into Data Cloud

<strong>Tip:</strong> Well-maintained knowledge + retrievers is the key to accurate, grounded agent responses.`
                }
            ]
        },
        {
            number: "Agentforce · Module 4",
            title: "Agentforce for Service and Sales",
            description: "Deploy agents and Einstein features across Service and Sales use cases.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Agentforce for Service",
                "Agentforce for Sales",
                "Channels & deployment"
            ],
            detailedDescription: "This domain covers applying Agentforce to real business functions: automating service interactions, accelerating sales work, and deploying agents across channels like websites and messaging.",
            detailedContent: [
                {
                    title: "Agentforce for Service",
                    content: `Agentforce automates and augments customer service.

<strong>Capabilities:</strong>
• <strong>Agentforce Service Agent</strong> resolves cases autonomously
• Grounded in Knowledge and case data
• Escalates to humans when needed

<strong>Related Einstein features:</strong> reply recommendations, case summaries, and knowledge generation.`
                },
                {
                    title: "Agentforce for Sales",
                    content: `Agentforce helps sellers move faster.

<strong>Examples:</strong>
• <strong>Sales Emails</strong> grounded in CRM data
• <strong>Sales Coach</strong> for practice and feedback
• Meeting and record summaries

<strong>Value:</strong> Automates busywork so reps focus on selling and relationships.`
                },
                {
                    title: "Channels & Deployment",
                    content: `Agents meet customers where they are.

<strong>Channels:</strong>
• Website and Experience Cloud
• Messaging (WhatsApp, SMS)
• Slack
• Service console for agents

<strong>Deployment:</strong> Build in Agent Builder, test in the preview, then activate and connect channels.`
                }
            ]
        },
        {
            number: "Agentforce · Module 5",
            title: "Einstein Trust Layer & Deployment",
            description: "Apply responsible AI with the Einstein Trust Layer and ship agents safely.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Einstein Trust Layer",
                "Testing & Agentforce Testing Center",
                "Monitoring & lifecycle"
            ],
            detailedDescription: "The final Agentforce Specialist domain covers trust and operations: how the Einstein Trust Layer protects data and enforces responsible AI, how to test agents before launch, and how to monitor and maintain them in production.",
            detailedContent: [
                {
                    title: "Einstein Trust Layer",
                    content: `The <strong>Einstein Trust Layer</strong> makes generative AI enterprise-safe.

<strong>Protections:</strong>
• <strong>Data masking</strong> of sensitive/PII fields
• <strong>Zero data retention</strong> with LLM providers
• <strong>Toxicity detection</strong> and scoring
• <strong>Prompt defense</strong> and audit trail

<strong>Exam tip:</strong> The Trust Layer secures prompts and responses without the model retaining your data.`
                },
                {
                    title: "Testing & Agentforce Testing Center",
                    content: `Validate agents before customers use them.

<strong>Approaches:</strong>
• Preview conversations in <strong>Agent Builder</strong>
• Use the <strong>Agentforce Testing Center</strong> to run test cases at scale
• Check topic classification and action selection

<strong>Goal:</strong> Confirm the agent picks the right topics and actions and responds safely.`
                },
                {
                    title: "Monitoring & Lifecycle",
                    content: `Operate agents responsibly after launch.

<strong>Practices:</strong>
• Review conversation transcripts and analytics
• Monitor deflection, accuracy, and escalations
• Refine topics, actions, and prompts over time
• Manage versions and permissions

<strong>Tip:</strong> Treat agents as living products — measure, learn, and iterate.`
                }
            ]
        },
        {
            number: "Agentforce · Module 6",
            title: "Building & Grounding Agents Deep Dive",
            description: "Go deeper on topics, actions, Data Cloud grounding, retrievers, and the Einstein Trust Layer that make Agentforce accurate and safe.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Topics & instructions",
                "Actions (Flow, Apex, prompt templates)",
                "Data Cloud grounding",
                "Retrievers & search indexes",
                "Prompt template types",
                "Einstein Trust Layer"
            ],
            detailedDescription: "The Agentforce Specialist exam is heavily weighted toward configuring agents and grounding them in trusted data. This module deepens topics and actions, Data Cloud grounding, retrievers, prompt template types, and the Einstein Trust Layer.",
            detailedContent: [
                {
                    title: "Topics & Instructions",
                    content: `A <strong>topic</strong> scopes a job an agent can handle and groups the instructions and actions that handle it.

<strong>Key ideas:</strong>
• Topics classify user requests into jobs
• Instructions guide how the agent behaves within a topic
• Clear, scoped topics improve accuracy and reduce wrong actions

<strong>Exam tip:</strong> Good topic design (clear scope + strong instructions) is the foundation of a reliable agent.`
                },
                {
                    title: "Actions (Flow, Apex, Prompt Templates)",
                    content: `<strong>Actions</strong> are what an agent can actually do.

<strong>Action sources:</strong>
• <strong>Flow</strong> - declarative automation
• <strong>Apex</strong> - custom code
• <strong>Prompt templates</strong> - generative responses
• <strong>Standard actions</strong> and external API calls

<strong>Exam tip:</strong> Actions can be backed by Flows, Apex, and prompt templates - know that an agent selects and runs actions to complete a topic's job.`
                },
                {
                    title: "Data Cloud Grounding",
                    content: `<strong>Data Cloud</strong> unifies data into a real-time customer view that grounds agents and prompts.

<strong>Building blocks:</strong>
• Data streams and data lake objects
• Data model objects (DMOs)
• Identity resolution for unified profiles

<strong>Role in Agentforce:</strong> grounding prompts and agents in Data Cloud provides trusted, current context so responses are accurate and specific.`
                },
                {
                    title: "Retrievers & Search Indexes",
                    content: `<strong>Retrievers</strong> find the most relevant content to ground a response (RAG).

<strong>How it works:</strong>
1. A retriever searches a vector/search index
2. It returns the most relevant chunks
3. Those chunks are added to the prompt as context

<strong>Sources:</strong> Salesforce Knowledge articles and Data Cloud search indexes over structured and unstructured content.

<strong>Exam tip:</strong> well-maintained knowledge plus retrievers is the key to accurate, grounded answers.`
                },
                {
                    title: "Prompt Template Types",
                    content: `<strong>Prompt Builder</strong> supports several template types for different jobs.

<strong>Common types:</strong>
• <strong>Sales/Field generation</strong> (e.g., sales emails)
• <strong>Record summaries</strong>
• <strong>Flex templates</strong> for flexible, multi-input prompts

<strong>Grounding:</strong> templates merge CRM and Data Cloud data into the prompt so output is specific and accurate.

<strong>Exam tip:</strong> choose the template type that matches the input and output you need, and always ground it in relevant data.`
                },
                {
                    title: "Einstein Trust Layer",
                    content: `The <strong>Einstein Trust Layer</strong> makes generative AI enterprise-safe.

<strong>Protections:</strong>
• Data masking of sensitive/PII fields
• Zero data retention with LLM providers
• Toxicity detection and scoring
• Prompt defense and an audit trail

<strong>Exam tip:</strong> the Trust Layer secures prompts and responses without the model retaining your data - a frequently tested concept.`
                }
            ]
        },
        {
            number: "Agentforce · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Trailhead and Developer Edition labs plus a domain-mapped study plan and exam-day strategy for the Agentforce Specialist exam.",
            duration: "55 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Build an agent in Agent Builder",
                "Lab: Create a grounded prompt template",
                "Lab: Ground with Data Cloud",
                "Lab: Test in Testing Center",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn Agentforce concepts into practice using a free Salesforce Developer Edition org and Trailhead, then map your study to the exam domains and apply a practical exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Build an Agent in Agent Builder",
                    content: `<strong>Goal:</strong> create a working agent end to end.

<strong>Steps:</strong>
1. Sign up for a free Salesforce Developer Edition org.
2. Enable Agentforce/Einstein features.
3. In Agent Builder, create an agent and add a topic.
4. Add an action to the topic and test it in the preview.

<strong>Takeaway:</strong> agents are assembled from topics and actions, then previewed before activation.`
                },
                {
                    title: "Lab: Create a Grounded Prompt Template",
                    content: `<strong>Goal:</strong> experience Prompt Builder and grounding.

<strong>Steps:</strong>
1. Open Prompt Builder and create a prompt template.
2. Add merge fields from a record (e.g., an Account).
3. Test the template and review the grounded output.

<strong>Takeaway:</strong> grounding merges CRM/Data Cloud data into the prompt so responses are specific and accurate.`
                },
                {
                    title: "Lab: Ground with Data Cloud",
                    content: `<strong>Goal:</strong> connect agents to unified data.

<strong>Steps (concept if Data Cloud is unavailable in your org):</strong>
1. Review how data streams and DMOs unify customer data.
2. Configure or inspect a search index/retriever.
3. Observe how retrieved content grounds an answer.

<strong>Takeaway:</strong> Data Cloud provides the trusted, unified context that makes agents reliable.`
                },
                {
                    title: "Lab: Test in Testing Center",
                    content: `<strong>Goal:</strong> validate agent behavior before launch.

<strong>Steps:</strong>
1. Open the Agentforce Testing Center.
2. Run test cases across your topics and actions.
3. Check topic classification and action selection.

<strong>Takeaway:</strong> testing confirms the agent picks the right topics and actions and responds safely.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the Agentforce Specialist exam areas.

<strong>Agentforce concepts:</strong> Modules 1 and 6 (topics, actions, agent types).
<strong>Prompt engineering:</strong> Module 2 plus Module 6 (Prompt Builder, template types, grounding).
<strong>Agentforce & Data:</strong> Module 3 plus Module 6 (Data Cloud, retrievers, knowledge).
<strong>Service & Sales applications:</strong> Module 4.
<strong>Trust & deployment:</strong> Module 5 plus Module 7 labs (Einstein Trust Layer, testing).

<strong>Recommended plan:</strong>
• Week 1: Modules 1-2 + build an agent lab
• Week 2: Modules 3 + 6 + prompt template and grounding labs
• Week 3: Modules 4-5 + 7 + Testing Center + Trailhead
• Week 4: Review + practice questions

<strong>Pair with:</strong> the official Trailhead Agentforce trailmix and hands-on practice in a Developer Edition org.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `The Agentforce Specialist exam is configuration- and scenario-based.

<strong>During the exam:</strong>
• Match scenarios to the right building block (topic, action, prompt template, retriever)
• When accuracy is the goal, think grounding in Data Cloud/Knowledge
• When safety or PII is mentioned, think the Einstein Trust Layer
• For "test before launch," think Testing Center
• Prefer declarative (Flow, prompt templates) before custom Apex when both fit

<strong>Mindset:</strong> you are configuring trusted, grounded agents on the Salesforce Platform.`
                }
            ]
        }
    ],
    sapGenAiDeveloper: [
        {
            number: "C_AIG · Module 1",
            title: "SAP Generative AI Foundations",
            description: "Understand generative AI on SAP Business Technology Platform and where the Generative AI Hub fits.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Generative AI & LLM basics",
                "SAP BTP AI portfolio",
                "Generative AI Hub overview"
            ],
            detailedDescription: "This domain introduces generative AI concepts and how SAP delivers them through SAP Business Technology Platform (BTP). You learn what large language models do, the SAP AI portfolio, and the role of the Generative AI Hub in SAP AI Core.",
            detailedContent: [
                {
                    title: "Generative AI & LLM Basics",
                    content: `<strong>Generative AI</strong> uses large language models (LLMs) to produce text, code, and structured output from natural-language prompts.

<strong>Key ideas:</strong>
• LLMs predict the next token from context
• <strong>Tokens</strong>, context windows, and temperature shape output
• Foundation models are adapted through prompting, RAG, and fine-tuning

<strong>Exam tip:</strong> Know the difference between prompting, retrieval augmentation, and fine-tuning as adaptation strategies.`
                },
                {
                    title: "SAP BTP AI Portfolio",
                    content: `<strong>SAP Business Technology Platform (BTP)</strong> is where SAP AI capabilities are built and consumed.

<strong>Core building blocks:</strong>
• <strong>SAP AI Core</strong> - runtime for training and serving AI
• <strong>SAP AI Launchpad</strong> - central operations UI
• <strong>Generative AI Hub</strong> - access to leading foundation models
• <strong>SAP HANA Cloud Vector Engine</strong> - embeddings and similarity search

<strong>Exam tip:</strong> AI Core runs the workloads; AI Launchpad manages them.`
                },
                {
                    title: "Generative AI Hub Overview",
                    content: `The <strong>Generative AI Hub</strong> in SAP AI Core gives trusted, governed access to multiple foundation models.

<strong>Capabilities:</strong>
• A single API and SDK across model providers
• Prompt experimentation in the hub
• Content filtering and data masking for trust
• Orchestration of prompts, grounding, and filtering

<strong>Exam tip:</strong> The Generative AI Hub abstracts model providers behind one consistent, governed interface.`
                }
            ]
        },
        {
            number: "C_AIG · Module 2",
            title: "SAP AI Core & AI Launchpad",
            description: "Provision, configure, and operate generative AI workloads with SAP AI Core and SAP AI Launchpad.",
            duration: "60 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Resource groups & configurations",
                "Deployments & model access",
                "SAP AI Launchpad operations",
                "Security & entitlements"
            ],
            detailedDescription: "This domain covers the operational backbone: setting up SAP AI Core, creating configurations and deployments for foundation models, and managing everything through SAP AI Launchpad with proper security.",
            detailedContent: [
                {
                    title: "Resource Groups & Configurations",
                    content: `<strong>SAP AI Core</strong> isolates work using <strong>resource groups</strong>.

<strong>Key objects:</strong>
• <strong>Resource group:</strong> tenant-level isolation of artifacts and secrets
• <strong>Configuration:</strong> a reusable definition of a workload (model, scenario, parameters)
• <strong>Executable/scenario:</strong> the template a configuration is based on

<strong>Exam tip:</strong> A configuration is the recipe; a deployment is the running instance.`
                },
                {
                    title: "Deployments & Model Access",
                    content: `A <strong>deployment</strong> serves a foundation model behind an inference endpoint.

<strong>Flow:</strong>
1. Create a configuration for the target model
2. Start a deployment from that configuration
3. Call the deployment URL via the harmonized inference API

<strong>Exam tip:</strong> Deployments expose a stable endpoint that apps and the orchestration service call.`
                },
                {
                    title: "SAP AI Launchpad Operations",
                    content: `<strong>SAP AI Launchpad</strong> is the central UI to operate AI across AI Core instances.

<strong>You can:</strong>
• Browse scenarios, configurations, and deployments
• Test prompts in the Generative AI Hub
• Monitor deployment status and logs
• Manage connections to multiple AI Core runtimes

<strong>Exam tip:</strong> Launchpad is for operations and experimentation, not for hosting your app code.`
                },
                {
                    title: "Security & Entitlements",
                    content: `Access is controlled through SAP BTP.

<strong>Essentials:</strong>
• <strong>Service keys</strong> and <strong>OAuth</strong> for API access
• BTP <strong>role collections</strong> for user permissions
• Resource-group scoping for isolation
• Content filtering and data masking for safe outputs

<strong>Exam tip:</strong> Least-privilege entitlements plus resource-group isolation keep workloads secure.`
                }
            ]
        },
        {
            number: "C_AIG · Module 3",
            title: "Prompt Engineering with the Generative AI Hub",
            description: "Design effective, reusable prompts and call foundation models through the harmonized API and SDK.",
            duration: "60 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Prompt design patterns",
                "System vs user messages",
                "Model parameters",
                "Harmonized API & SDK"
            ],
            detailedDescription: "The largest developer domain covers prompt engineering: writing clear instructions, using message roles, tuning model parameters, and calling models programmatically through the Generative AI Hub SDK.",
            detailedContent: [
                {
                    title: "Prompt Design Patterns",
                    content: `Good prompts are specific, contextual, and structured.

<strong>Patterns:</strong>
• <strong>Zero-shot</strong> and <strong>few-shot</strong> examples
• Role and format instructions
• Chain-of-thought for reasoning tasks
• Delimiters to separate instructions from data

<strong>Exam tip:</strong> Provide context, constraints, and the desired output format to improve reliability.`
                },
                {
                    title: "System vs User Messages",
                    content: `Chat-style models use message <strong>roles</strong>.

<strong>Roles:</strong>
• <strong>System:</strong> sets behavior, persona, and rules
• <strong>User:</strong> the request
• <strong>Assistant:</strong> prior model responses for context

<strong>Exam tip:</strong> Put durable rules and guardrails in the system message; put the task in the user message.`
                },
                {
                    title: "Model Parameters",
                    content: `Parameters control generation behavior.

<strong>Common parameters:</strong>
• <strong>temperature</strong> / <strong>top_p</strong> - randomness vs determinism
• <strong>max_tokens</strong> - output length
• <strong>frequency/presence penalties</strong> - repetition control

<strong>Exam tip:</strong> Lower temperature for factual, deterministic tasks; higher for creative generation.`
                },
                {
                    title: "Harmonized API & SDK",
                    content: `The Generative AI Hub exposes a <strong>harmonized API</strong> across model providers.

<strong>Developer options:</strong>
• Native REST inference endpoints
• The <strong>generative-ai-hub-sdk</strong> (Python)
• LangChain-compatible interfaces

<strong>Exam tip:</strong> The harmonized interface lets you switch models with minimal code changes.`
                }
            ]
        },
        {
            number: "C_AIG · Module 4",
            title: "Embeddings, Vector Search & RAG",
            description: "Ground responses in enterprise data using embeddings, SAP HANA Cloud Vector Engine, and Retrieval-Augmented Generation.",
            duration: "60 min",
            lessons: "4 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Embeddings & vectors",
                "SAP HANA Cloud Vector Engine",
                "RAG pipeline design",
                "Chunking & retrieval quality"
            ],
            detailedDescription: "This domain covers grounding: converting content to embeddings, storing and searching them in SAP HANA Cloud Vector Engine, and building Retrieval-Augmented Generation pipelines that keep answers accurate and current.",
            detailedContent: [
                {
                    title: "Embeddings & Vectors",
                    content: `<strong>Embeddings</strong> turn text into numeric vectors that capture meaning.

<strong>Key ideas:</strong>
• Similar content produces nearby vectors
• Similarity is measured with cosine or dot product
• Embedding models are called through the Generative AI Hub

<strong>Exam tip:</strong> Embeddings power semantic search - the retrieval half of RAG.`
                },
                {
                    title: "SAP HANA Cloud Vector Engine",
                    content: `<strong>SAP HANA Cloud</strong> stores embeddings with its <strong>Vector Engine</strong>.

<strong>Capabilities:</strong>
• A native <strong>REAL_VECTOR</strong> column type
• Similarity functions for nearest-neighbor search
• Combine vector search with SQL and business data

<strong>Exam tip:</strong> HANA Cloud lets you keep vectors alongside governed enterprise data for grounded retrieval.`
                },
                {
                    title: "RAG Pipeline Design",
                    content: `<strong>Retrieval-Augmented Generation (RAG)</strong> grounds an LLM in your own knowledge.

<strong>Pipeline:</strong>
1. Embed and index source documents
2. Retrieve relevant chunks for a query
3. Add them to the prompt as context
4. Generate a grounded, cited answer

<strong>Exam tip:</strong> RAG reduces hallucinations by anchoring generation to retrieved, trusted content.`
                },
                {
                    title: "Chunking & Retrieval Quality",
                    content: `Retrieval quality depends on how you prepare data.

<strong>Best practices:</strong>
• Chunk documents by semantic boundaries
• Balance chunk size vs context relevance
• Store useful metadata for filtering
• Evaluate retrieval with representative queries

<strong>Exam tip:</strong> Better chunking and metadata filtering usually improves answers more than a bigger model.`
                }
            ]
        },
        {
            number: "C_AIG · Module 5",
            title: "Orchestration, Grounding & Function Calling",
            description: "Compose prompts, grounding, filtering, and tools with the SAP AI Core orchestration service.",
            duration: "55 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Orchestration service",
                "Grounding & templating",
                "Function calling & agents"
            ],
            detailedDescription: "This domain covers the orchestration workflow in SAP AI Core that chains templating, model calls, grounding, and content filtering, plus function calling to let models trigger tools and actions.",
            detailedContent: [
                {
                    title: "Orchestration Service",
                    content: `The <strong>orchestration service</strong> in the Generative AI Hub composes generative AI steps into one call.

<strong>Modules include:</strong>
• <strong>Templating</strong> - reusable prompt templates
• <strong>Model configuration</strong> - which model and parameters
• <strong>Grounding</strong> - inject retrieved data
• <strong>Content filtering</strong> and <strong>data masking</strong>

<strong>Exam tip:</strong> Orchestration lets you combine templating, grounding, and filtering without wiring each step yourself.`
                },
                {
                    title: "Grounding & Templating",
                    content: `<strong>Templating</strong> defines a prompt with placeholders filled at runtime.

<strong>Grounding module:</strong>
• Retrieves relevant documents (e.g., from HANA Cloud)
• Injects them into the template
• Keeps responses specific and current

<strong>Exam tip:</strong> Templating + grounding in orchestration is the managed way to build RAG on SAP.`
                },
                {
                    title: "Function Calling & Agents",
                    content: `<strong>Function calling</strong> lets a model request a tool with structured arguments.

<strong>Flow:</strong>
1. Describe available functions/tools to the model
2. The model returns a structured call
3. Your app runs it and returns the result
4. The model uses the result to respond

<strong>Exam tip:</strong> Function calling is the foundation for agentic workflows that take real actions on SAP data.`
                }
            ]
        },
        {
            number: "C_AIG · Module 6",
            title: "Responsible AI, Trust & Operations",
            description: "Apply content filtering, data masking, and monitoring to ship generative AI responsibly on SAP BTP.",
            duration: "50 min",
            lessons: "3 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Content filtering & data masking",
                "Responsible & ethical AI",
                "Monitoring & lifecycle"
            ],
            detailedDescription: "This domain covers trust and operations: protecting sensitive data with masking, filtering harmful content, following SAP's responsible AI principles, and monitoring deployments through their lifecycle.",
            detailedContent: [
                {
                    title: "Content Filtering & Data Masking",
                    content: `SAP builds trust directly into the Generative AI Hub.

<strong>Guardrails:</strong>
• <strong>Content filtering</strong> on inputs and outputs
• <strong>Data masking</strong> of PII before it reaches the model
• Configurable within orchestration

<strong>Exam tip:</strong> Masking protects sensitive data; filtering blocks harmful or unwanted content.`
                },
                {
                    title: "Responsible & Ethical AI",
                    content: `SAP follows guiding principles for <strong>ethical AI</strong>.

<strong>Principles emphasize:</strong>
• Human agency and oversight
• Transparency and explainability
• Fairness and bias mitigation
• Data privacy and security

<strong>Exam tip:</strong> Responsible AI is a design requirement, not an afterthought - keep humans in the loop for high-impact decisions.`
                },
                {
                    title: "Monitoring & Lifecycle",
                    content: `Operate generative AI workloads reliably after launch.

<strong>Practices:</strong>
• Monitor deployment health and logs in AI Launchpad
• Track cost, latency, and token usage
• Version configurations and prompts
• Re-evaluate grounding sources over time

<strong>Exam tip:</strong> Treat deployments and prompts as versioned artifacts you monitor and iterate on.`
                }
            ]
        },
        {
            number: "C_AIG · Module 7",
            title: "Hands-On Labs & Exam Readiness",
            description: "Guided Generative AI Hub and SAP AI Core labs plus a domain-mapped study plan and exam-day strategy for the C_AIG exam.",
            duration: "55 min",
            lessons: "5 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Lab: Call a model with the SDK",
                "Lab: Build a RAG pipeline on HANA Cloud",
                "Lab: Orchestration with grounding",
                "Exam domain study plan",
                "Exam-day strategy"
            ],
            detailedDescription: "Turn C_AIG concepts into practice using SAP AI Core, the Generative AI Hub, and SAP HANA Cloud, then map your study to the exam domains and apply a practical exam-day strategy.",
            detailedContent: [
                {
                    title: "Lab: Call a Model with the SDK",
                    content: `<strong>Goal:</strong> run your first generative AI call on SAP.

<strong>Steps:</strong>
1. Provision SAP AI Core and bind a service key.
2. Create a configuration and deployment for a foundation model.
3. Install the <strong>generative-ai-hub-sdk</strong> and authenticate.
4. Send a prompt and read the response.

<strong>Takeaway:</strong> A deployment plus the harmonized SDK is all you need to call a model.`
                },
                {
                    title: "Lab: Build a RAG Pipeline on HANA Cloud",
                    content: `<strong>Goal:</strong> ground answers in your own documents.

<strong>Steps:</strong>
1. Chunk and embed source documents via the hub.
2. Store vectors in a HANA Cloud REAL_VECTOR column.
3. Retrieve top matches for a query with similarity search.
4. Add retrieved chunks to the prompt and generate an answer.

<strong>Takeaway:</strong> Embeddings + HANA Cloud Vector Engine + prompting = grounded, accurate responses.`
                },
                {
                    title: "Lab: Orchestration with Grounding",
                    content: `<strong>Goal:</strong> use the managed orchestration workflow.

<strong>Steps:</strong>
1. Define a prompt template with placeholders.
2. Configure the grounding module to retrieve context.
3. Enable content filtering and data masking.
4. Run the orchestrated call and inspect each module's output.

<strong>Takeaway:</strong> Orchestration composes templating, grounding, and filtering into a single trusted call.`
                },
                {
                    title: "Exam Domain Study Plan",
                    content: `Map your study to the C_AIG exam areas.

<strong>Foundations & platform:</strong> Modules 1-2 (BTP, AI Core, AI Launchpad).
<strong>Prompt engineering:</strong> Module 3 (design, roles, parameters, SDK).
<strong>Grounding & RAG:</strong> Module 4 (embeddings, HANA Cloud, retrieval).
<strong>Orchestration & tools:</strong> Module 5 (orchestration, function calling).
<strong>Trust & operations:</strong> Module 6 plus Module 7 labs.

<strong>Recommended plan:</strong>
• Week 1: Modules 1-2 + SDK lab
• Week 2: Modules 3-4 + RAG lab
• Week 3: Modules 5-6 + orchestration lab
• Week 4: Review + practice questions

<strong>Pair with:</strong> the official SAP Learning journey and hands-on time in an SAP BTP trial or free-tier account.`
                },
                {
                    title: "Exam-Day Strategy",
                    content: `The C_AIG exam is developer- and scenario-based.

<strong>During the exam:</strong>
• Match scenarios to the right building block (deployment, orchestration, grounding, function calling)
• When accuracy is the goal, think RAG with HANA Cloud embeddings
• When privacy or safety is mentioned, think data masking and content filtering
• When tools/actions are needed, think function calling
• Prefer the harmonized API and orchestration over hard-coding a single provider

<strong>Mindset:</strong> you are building trusted, grounded generative AI apps on SAP BTP.`
                }
            ]
        }
    ]
});
