// Course Data Structure
const courseData = {
    mlModels: [
        {
            number: "Module 0",
            title: "Introduction to ML",
            description: "A quick introduction to machine learning fundamentals, types of ML, and basic concepts to get you started.",
            duration: "30 min",
            lessons: "6 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "What is Machine Learning?",
                "Types of Machine Learning",
                "Supervised vs Unsupervised Learning",
                "Common ML Applications",
                "ML Workflow Overview",
                "Getting Started with ML"
            ],
            detailedDescription: "This introductory module covers the fundamentals of machine learning. You'll learn what machine learning is, explore different types of ML including supervised and unsupervised learning, understand common real-world applications, and get familiar with the basic ML workflow. Perfect for absolute beginners!",
            detailedContent: [
                {
                    title: "What is Machine Learning?",
                    content: `Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.

<strong>Traditional Programming vs Machine Learning:</strong>

<strong>Traditional Programming:</strong>
• Programmer writes explicit rules
• Input + Rules → Output
• Example: if temperature > 30°C, output "Hot"

<strong>Machine Learning:</strong>
• Computer learns rules from data
• Input + Output → Rules (Model)
• Example: Show many temperature-label pairs, model learns patterns

<strong>Key Components:</strong>
• <strong>Data:</strong> The fuel for ML (examples to learn from)
• <strong>Model:</strong> Mathematical representation of patterns
• <strong>Training:</strong> Process of learning from data
• <strong>Prediction:</strong> Using the learned model on new data

<strong>Why Machine Learning?</strong>
• Handles complex patterns humans can't easily code
• Adapts to new data automatically
• Scales to large datasets
• Improves over time with more data`,
                    code: `# Simple ML Example: Predicting House Prices
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: [size in sq ft]
X_train = np.array([[600], [800], [1000], [1200], [1400]])
y_train = np.array([150, 200, 250, 300, 350])  # prices in thousands

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict price for a 1100 sq ft house
house_size = [[1100]]
predicted_price = model.predict(house_size)
print("Predicted price: $" + str(int(predicted_price[0])) + "k")

# The model learned: Price ≈ Size × 0.25`
                },
                {
                    title: "Types of Machine Learning",
                    content: `Machine Learning can be categorized into three main types based on how the model learns:

<strong>1. Supervised Learning</strong>
• Learning with labeled data (input + correct output)
• Like learning with a teacher who provides answers
• <strong>Examples:</strong> Email spam detection, house price prediction
• <strong>Types:</strong>
  - Classification: Predicting categories (spam/not spam)
  - Regression: Predicting numbers (house prices)

<strong>2. Unsupervised Learning</strong>
• Learning from unlabeled data (only inputs)
• Discovers hidden patterns without guidance
• <strong>Examples:</strong> Customer segmentation, anomaly detection
• <strong>Types:</strong>
  - Clustering: Grouping similar items
  - Dimensionality Reduction: Simplifying data

<strong>3. Reinforcement Learning</strong>
• Learning through trial and error
• Receives rewards/penalties for actions
• <strong>Examples:</strong> Game playing AI, robotics, self-driving cars
• Agent learns optimal strategy over time`,
                    code: `# Supervised Learning Example
from sklearn.tree import DecisionTreeClassifier

# Training data: [hours_studied, hours_slept]
X = [[2, 8], [4, 7], [6, 6], [8, 5], [1, 9]]
y = [0, 0, 1, 1, 0]  # Labels: 0=Fail, 1=Pass

# Train model with labeled data
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for new student
new_student = [[5, 7]]
prediction = model.predict(new_student)
result = 'Pass' if prediction[0] else 'Fail'
print("Prediction:", result)

# -------------------
# Unsupervised Learning Example
from sklearn.cluster import KMeans

# Customer data: [age, spending_score]
customers = [[25, 70], [30, 80], [35, 90], 
             [22, 30], [28, 40], [32, 35]]

# Find patterns (no labels needed!)
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(customers)
print("Customer groups:", clusters)
# Output: [1 1 1 0 0 0] - Found 2 groups!`
                },
                {
                    title: "Supervised vs Unsupervised Learning",
                    content: `<strong>Supervised Learning - Learning with Labels</strong>

<strong>Characteristics:</strong>
• Requires labeled training data
• Goal: Learn mapping from input to output
• Measures accuracy against known answers
• More common in practical applications

<strong>When to Use:</strong>
• You have historical data with outcomes
• Clear prediction target exists
• Need to make specific predictions

<strong>Examples:</strong>
• <strong>Classification:</strong> Is this email spam? (Yes/No)
• <strong>Regression:</strong> What will stock price be tomorrow? (dollar amount)

<strong>Unsupervised Learning - Finding Hidden Patterns</strong>

<strong>Characteristics:</strong>
• Works with unlabeled data
• Goal: Discover structure in data
• No "correct answer" to measure against
• Explores data to find insights

<strong>When to Use:</strong>
• Exploring new datasets
• Finding natural groupings
• Reducing data complexity
• Anomaly detection

<strong>Examples:</strong>
• <strong>Clustering:</strong> Group customers by behavior
• <strong>Anomaly Detection:</strong> Find unusual transactions
• <strong>Dimensionality Reduction:</strong> Compress large datasets`,
                    code: `# Side-by-side comparison

# SUPERVISED: Predict if customer will buy
X_supervised = [[25, 50000], [35, 60000], [45, 80000]]
y_labels = [0, 0, 1]  # 0=No purchase, 1=Purchase

from sklearn.ensemble import RandomForestClassifier
supervised_model = RandomForestClassifier()
supervised_model.fit(X_supervised, y_labels)
print("Will buy?", supervised_model.predict([[30, 55000]]))

# UNSUPERVISED: Discover customer segments
X_unsupervised = [[25, 50000], [35, 60000], [45, 80000],
                  [50, 90000], [28, 52000], [48, 85000]]

from sklearn.cluster import KMeans
unsupervised_model = KMeans(n_clusters=2)
segments = unsupervised_model.fit_predict(X_unsupervised)
print("Customer segments:", segments)
# Discovers: [0 0 1 1 0 1] - 2 groups found!`
                },
                {
                    title: "Common ML Applications",
                    content: `Machine Learning powers many applications we use daily:

<strong>1. Computer Vision</strong>
• <strong>Face Recognition:</strong> Unlock phones, tag photos
• <strong>Object Detection:</strong> Self-driving cars, security
• <strong>Medical Imaging:</strong> Detect diseases in X-rays
• <strong>OCR:</strong> Convert images to text

<strong>2. Natural Language Processing</strong>
• <strong>Language Translation:</strong> Google Translate
• <strong>Chatbots:</strong> Customer service automation
• <strong>Sentiment Analysis:</strong> Analyze reviews, social media
• <strong>Text Generation:</strong> AI writing assistants

<strong>3. Recommendation Systems</strong>
• <strong>E-commerce:</strong> Product recommendations (Amazon)
• <strong>Streaming:</strong> Movie/music suggestions (Netflix, Spotify)
• <strong>Social Media:</strong> Friend suggestions, content feeds

<strong>4. Finance & Business</strong>
• <strong>Fraud Detection:</strong> Identify suspicious transactions
• <strong>Credit Scoring:</strong> Assess loan applications
• <strong>Stock Prediction:</strong> Trading algorithms
• <strong>Customer Churn:</strong> Predict who might leave

<strong>5. Healthcare</strong>
• <strong>Disease Diagnosis:</strong> Early detection
• <strong>Drug Discovery:</strong> Find new medicines
• <strong>Personalized Treatment:</strong> Tailored therapy plans

<strong>6. Other Applications</strong>
• <strong>Weather Forecasting:</strong> More accurate predictions
• <strong>Speech Recognition:</strong> Virtual assistants (Siri, Alexa)
• <strong>Spam Filtering:</strong> Email protection
• <strong>Search Engines:</strong> Better search results`,
                    code: `# Example: Simple Spam Detection Application
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training emails and labels
emails = [
    "Win FREE prize NOW! Click here!!!",
    "Meeting scheduled for tomorrow at 3pm",
    "URGENT: Your account needs verification!!!",
    "Project update: All tasks completed",
    "Claim your million dollar prize today!!!",
    "Can we discuss the report tomorrow?"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to numbers (ML only understands numbers!)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train spam detector
spam_detector = MultinomialNB()
spam_detector.fit(X, labels)

# Test on new email
new_email = ["Free money! Click now!"]
X_new = vectorizer.transform(new_email)
prediction = spam_detector.predict(X_new)
probability = spam_detector.predict_proba(X_new)[0][1]

print("Is spam:", prediction[0] == 1)
print("Confidence: {:.0%}".format(probability))
# Output: Is spam: True, Confidence: 95%`
                },
                {
                    title: "ML Workflow Overview",
                    content: `Every machine learning project follows a similar workflow:

<strong>1. Problem Definition</strong>
• What problem are we solving?
• Is ML the right approach?
• What data do we need?
• How will we measure success?

<strong>2. Data Collection & Preparation</strong>
• <strong>Gather data:</strong> From databases, APIs, files
• <strong>Clean data:</strong> Handle missing values, outliers
• <strong>Explore data:</strong> Understand patterns and distributions
• <strong>Feature engineering:</strong> Create useful input variables

<strong>3. Model Selection & Training</strong>
• <strong>Choose algorithm:</strong> Based on problem type
• <strong>Split data:</strong> Training set vs Testing set
• <strong>Train model:</strong> Learn patterns from training data
• <strong>Tune parameters:</strong> Optimize model performance

<strong>4. Model Evaluation</strong>
• <strong>Test performance:</strong> Use testing data (never seen before!)
• <strong>Calculate metrics:</strong> Accuracy, precision, recall, etc.
• <strong>Cross-validation:</strong> Ensure model generalizes well
• <strong>Compare models:</strong> Choose the best performer

<strong>5. Deployment & Monitoring</strong>
• <strong>Deploy model:</strong> Put in production environment
• <strong>Monitor performance:</strong> Track real-world accuracy
• <strong>Update model:</strong> Retrain with new data
• <strong>A/B testing:</strong> Compare against baseline

<strong>Data Split Best Practice:</strong>
• <strong>Training Set:</strong> 60-80% - Model learns from this
• <strong>Validation Set:</strong> 10-20% - Tune hyperparameters
• <strong>Test Set:</strong> 10-20% - Final evaluation (use only once!)`,
                    code: `# Complete ML Workflow Example
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Problem: Predict if a fruit is an apple or orange
# 2. Data Collection & Preparation
data = {
    'weight': [150, 170, 140, 130, 160, 180, 145, 155],
    'diameter': [7, 7.5, 6.8, 6.5, 7.2, 7.8, 6.9, 7.1],
    'fruit': ['apple', 'apple', 'orange', 'orange', 
              'apple', 'apple', 'orange', 'apple']
}
df = pd.DataFrame(data)

# Prepare features and labels
X = df[['weight', 'diameter']]
y = df['fruit']

# 3. Split data (training vs testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Model Selection & Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: {:.0%}".format(accuracy))

# 6. Make predictions on new data
new_fruit = [[165, 7.3]]
prediction = model.predict(new_fruit)
print("Predicted fruit:", prediction[0])

# 7. Ready for deployment!`
                },
                {
                    title: "Getting Started with ML",
                    content: `<strong>Essential Skills to Learn</strong>

<strong>1. Programming</strong>
• <strong>Python:</strong> Most popular language for ML
• <strong>Key Libraries:</strong>
  - NumPy: Numerical computing
  - pandas: Data manipulation
  - scikit-learn: ML algorithms
  - TensorFlow/PyTorch: Deep learning

<strong>2. Mathematics Foundations</strong>
• <strong>Statistics:</strong> Mean, standard deviation, probability
• <strong>Linear Algebra:</strong> Vectors, matrices, matrix operations
• <strong>Calculus:</strong> Derivatives, gradients (for optimization)

<strong>3. Data Skills</strong>
• Data cleaning and preprocessing
• Feature engineering and selection
• Data visualization (matplotlib, seaborn)
• Understanding data types and distributions

<strong>Learning Path Recommendation</strong>

<strong>Beginner (1-2 months):</strong>
1. Python basics
2. NumPy and pandas tutorials
3. Basic statistics
4. Simple ML algorithms (linear regression, decision trees)

<strong>Intermediate (3-6 months):</strong>
1. More ML algorithms
2. Cross-validation and model evaluation
3. Feature engineering techniques
4. Real datasets and Kaggle competitions

<strong>Advanced (6+ months):</strong>
1. Deep learning (neural networks)
2. Specialized domains (NLP, Computer Vision)
3. Model deployment and MLOps
4. Research papers and cutting-edge techniques

<strong>Resources to Get Started:</strong>
• This course! (Machine Learning Crash Course)
• Kaggle: Practice with real datasets
• Coursera/edX: Structured courses
• GitHub: Explore open-source projects
• ML blogs and papers

<strong>Tips for Success:</strong>
• Start with simple projects
• Practice consistently
• Learn by doing, not just watching
• Join ML communities
• Don't get overwhelmed - take it step by step!`,
                    code: `# Your First ML Project: Complete Example
# Problem: Predict student exam pass/fail

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
# Features: [hours_studied, previous_score, attendance%]
X = np.array([
    [2, 45, 60], [4, 55, 70], [6, 65, 80],
    [8, 75, 90], [3, 50, 65], [7, 70, 85],
    [5, 60, 75], [9, 85, 95], [1, 40, 50]
])
y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])  # 0=Fail, 1=Pass

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.0%}".format(accuracy))

# Step 5: Predict for new student
new_student = [[5, 60, 75]]  # 5 hrs, 60 score, 75% attendance
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)
result = 'Pass' if prediction[0] else 'Fail'
confidence = probability[0][1]

print("Prediction:", result)
print("Confidence: {:.0%}".format(confidence))

# Congratulations! You built your first ML model! 🎉`
                }
            ]
        },
        {
            number: "Module 1",
            title: "Linear Regression",
            description: "An introduction to linear regression, covering linear models, loss, gradient descent, and hyperparameter tuning.",
            duration: "45 min",
            lessons: "8 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Introduction to Linear Models",
                "Understanding Loss Functions",
                "Gradient Descent Explained",
                "Learning Rate and Optimization",
                "Hyperparameter Tuning",
                "Practical Implementation",
                "Model Evaluation",
                "Real-world Examples"
            ],
            detailedDescription: "Linear regression is one of the fundamental algorithms in machine learning. This module will teach you how to build predictive models that establish a linear relationship between input features and output predictions. You'll learn about loss functions, how gradient descent optimizes models, and the importance of choosing the right hyperparameters.",
            detailedContent: [
                {
                    title: "Introduction to Linear Models",
                    content: `Linear regression is the foundation of machine learning. It models the relationship between input variables (features) and output variable (target) using a linear equation.
                    
<strong>The Linear Equation:</strong>
y = mx + b

Where:
• y = predicted output (target variable)
• m = slope (weight/coefficient)
• x = input feature
• b = y-intercept (bias)

For multiple features:
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

<strong>Key Concepts:</strong>
• <strong>Features (X):</strong> Input variables used for prediction
• <strong>Target (y):</strong> Output variable we want to predict
• <strong>Weights (w):</strong> Parameters that determine feature importance
• <strong>Bias (b):</strong> Shifts the line up or down`,
                    code: `# Simple Linear Regression Example
import numpy as np
import matplotlib.pyplot as plt

# Sample data: Hours studied vs Test score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Hours studied
y = np.array([2, 4, 5, 4, 6, 7, 8, 9])  # Test scores

# Linear model: y = mx + b
m = 1.2  # slope (weight)
b = 0.5  # intercept (bias)

# Make predictions
y_pred = m * X + b

# Visualize
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.legend()
plt.show()

print(f"Prediction for 10 hours: " + str(m * 10 + b))`
                },
                {
                    title: "Understanding Loss Functions",
                    content: `A loss function measures how well our model's predictions match the actual values. It quantifies the error between predicted and actual values.

<strong>Mean Squared Error (MSE):</strong>
The most common loss function for regression problems.

MSE = (1/n) × Σ(y_actual - y_predicted)²

<strong>Why square the errors?</strong>
• Penalizes larger errors more heavily
• Always positive (no negative errors canceling positives)
• Mathematically convenient for optimization

<strong>Other Loss Functions:</strong>
• <strong>MAE (Mean Absolute Error):</strong> Less sensitive to outliers
• <strong>RMSE (Root Mean Squared Error):</strong> Same units as target variable
• <strong>Huber Loss:</strong> Combines MSE and MAE benefits`,
                    code: `import numpy as np

# Actual and predicted values
y_actual = np.array([3, 5, 7, 9, 11])
y_predicted = np.array([2.5, 5.5, 6.8, 9.2, 10.5])

# Calculate MSE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Calculate MAE
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Calculate RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

mse = mean_squared_error(y_actual, y_predicted)
mae = mean_absolute_error(y_actual, y_predicted)
rmse = root_mean_squared_error(y_actual, y_predicted)

print("MSE:", round(mse, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# Output:
# MSE: 0.2140
# MAE: 0.3600
# RMSE: 0.4626`
                },
                {
                    title: "Gradient Descent Explained",
                    content: `Gradient Descent is an optimization algorithm that finds the best parameters (weights and bias) by iteratively moving in the direction of steepest descent.

<strong>How it works:</strong>
1. Start with random weights
2. Calculate the loss (error)
3. Compute gradients (slopes)
4. Update weights in opposite direction of gradient
5. Repeat until convergence

<strong>The Update Rule:</strong>
w_new = w_old - α × ∂Loss/∂w

Where:
• α (alpha) = learning rate
• ∂Loss/∂w = gradient (derivative of loss)

<strong>Types of Gradient Descent:</strong>
• <strong>Batch GD:</strong> Uses entire dataset (slow but stable)
• <strong>Stochastic GD:</strong> Uses one sample (fast but noisy)
• <strong>Mini-batch GD:</strong> Uses small batches (best of both)`,
                    code: `import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initialize parameters
w = 0.0  # weight
b = 0.0  # bias
learning_rate = 0.01
epochs = 100

n = len(X)

# Gradient Descent
for epoch in range(epochs):
    # Forward pass: predictions
    y_pred = w * X + b
    
    # Calculate loss (MSE)
    loss = np.mean((y - y_pred) ** 2)
    
    # Calculate gradients
    dw = -(2/n) * np.sum(X * (y - y_pred))
    db = -(2/n) * np.sum(y - y_pred)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print("\\nFinal: w=" + str(round(w, 4)) + ", b=" + str(round(b, 4)))
# Expected: w ≈ 2.0, b ≈ 0.0`
                },
                {
                    title: "Learning Rate and Optimization",
                    content: `The learning rate (α) is one of the most important hyperparameters. It controls how big the steps are during gradient descent.

<strong>Learning Rate Impact:</strong>

• <strong>Too Small:</strong> Slow convergence, takes forever
• <strong>Too Large:</strong> Overshooting, never converges
• <strong>Just Right:</strong> Fast and stable convergence

<strong>Adaptive Learning Rates:</strong>
Modern optimizers automatically adjust the learning rate:

• <strong>Adam:</strong> Adapts learning rate per parameter
• <strong>RMSprop:</strong> Uses moving average of gradients
• <strong>AdaGrad:</strong> Adapts based on historical gradients
• <strong>SGD with Momentum:</strong> Accelerates in relevant direction

<strong>Learning Rate Schedules:</strong>
• Step Decay: Reduce by factor every N epochs
• Exponential Decay: Gradual reduction
• Cosine Annealing: Oscillating reduction`,
                    code: `import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

def train_with_lr(learning_rate, epochs=100):
    w, b = 0.0, 0.0
    n = len(X)
    
    for epoch in range(epochs):
        y_pred = w * X + b
        loss = np.mean((y - y_pred) ** 2)
        
        dw = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    return w, b, loss

# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]

print("Learning Rate Comparison:")
print("-" * 50)
for lr in learning_rates:
    w, b, loss = train_with_lr(lr)
    print(f"LR={lr:.3f}: w={w:.4f}, b={b:.4f}, Loss={loss:.6f}")

# Output shows optimal learning rate

# SGD with Momentum
def sgd_momentum(X, y, lr=0.01, momentum=0.9, epochs=100):
    w, b = 0.0, 0.0
    vw, vb = 0.0, 0.0  # velocity
    n = len(X)
    
    for epoch in range(epochs):
        y_pred = w * X + b
        dw = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        
        # Update velocity
        vw = momentum * vw + lr * dw
        vb = momentum * vb + lr * db
        
        # Update parameters
        w = w - vw
        b = b - vb
    
    return w, b

w, b = sgd_momentum(X, y)
print("\\nWith Momentum: w=" + str(round(w, 4)) + ", b=" + str(round(b, 4)))`
                },
                {
                    title: "Hyperparameter Tuning",
                    content: `Hyperparameters are settings that control the learning process. Unlike model parameters (weights), they must be set before training.

<strong>Key Hyperparameters:</strong>

1. <strong>Learning Rate (α):</strong>
   • Most critical hyperparameter
   • Typical range: 0.001 to 0.1
   • Use learning rate finder

2. <strong>Number of Epochs:</strong>
   • Too few: Underfitting
   • Too many: Overfitting
   • Use early stopping

3. <strong>Batch Size:</strong>
   • Small (32): Noisy but generalizes well
   • Large (256): Stable but may overfit
   • Typical: 32, 64, 128, 256

4. <strong>Regularization:</strong>
   • L1 (Lasso): Feature selection
   • L2 (Ridge): Weight decay
   • Elastic Net: Combination

<strong>Tuning Strategies:</strong>
• Grid Search: Try all combinations
• Random Search: Random sampling
• Bayesian Optimization: Smart search`,
                    code: `from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Define hyperparameter grid
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky']
}

# Create model
model = Ridge()

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_squared_error'
)

# Fit and find best parameters
grid_search.fit(X, y)

print("Best Hyperparameters:")
print(grid_search.best_params_)
print("\\nBest Score:", round(-grid_search.best_score_, 4))

# Manual hyperparameter tuning
def tune_manually(X, y):
    best_loss = float('inf')
    best_params = {}
    
    for lr in [0.001, 0.01, 0.1]:
        for epochs in [50, 100, 200]:
            # Train model (simplified)
            w, b = 0.0, 0.0
            for _ in range(epochs):
                y_pred = w * X.flatten() + b
                loss = np.mean((y - y_pred) ** 2)
                dw = -(2/len(X)) * np.sum(X.flatten() * (y - y_pred))
                db = -(2/len(X)) * np.sum(y - y_pred)
                w -= lr * dw
                b -= lr * db
            
            if loss < best_loss:
                best_loss = loss
                best_params = {'lr': lr, 'epochs': epochs}
    
    return best_params, best_loss

best_params, best_loss = tune_manually(X, y)
print("\\nManual Tuning - Best:", best_params)
print("Loss:", round(best_loss, 4))`
                },
                {
                    title: "Practical Implementation",
                    content: `Let's build a complete linear regression model from scratch and compare it with scikit-learn's implementation.

<strong>Implementation Steps:</strong>

1. <strong>Data Preparation:</strong>
   • Load and clean data
   • Handle missing values
   • Feature scaling/normalization

2. <strong>Model Training:</strong>
   • Initialize parameters
   • Run gradient descent
   • Monitor convergence

3. <strong>Prediction:</strong>
   • Use trained weights
   • Make predictions on new data

4. <strong>Evaluation:</strong>
   • Calculate metrics
   • Visualize results

<strong>Best Practices:</strong>
• Always normalize/standardize features
• Use train-test split
• Monitor training loss
• Validate on unseen data
• Check for overfitting`,
                    code: `import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Custom Linear Regression Class
class MyLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient Descent
        for epoch in range(self.epochs):
            # Predictions
            y_pred = np.dot(X, self.w) + self.b
            
            # Calculate loss
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # Gradients
            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Generate dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train custom model
my_model = MyLinearRegression(learning_rate=0.1, epochs=500)
my_model.fit(X_train_scaled, y_train)

# Train sklearn model
sk_model = LinearRegression()
sk_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_my = my_model.predict(X_test_scaled)
y_pred_sk = sk_model.predict(X_test_scaled)

# Compare
print("Custom Model:")
print("  Weights:", my_model.w)
print("  Bias:", round(my_model.b, 4))
print("  MSE:", round(np.mean((y_test - y_pred_my)**2), 4))

print("\\nSklearn Model:")
print("  Weights:", sk_model.coef_)
print("  Bias:", round(sk_model.intercept_, 4))
print("  MSE:", round(np.mean((y_test - y_pred_sk)**2), 4))

# Plot training loss
plt.plot(my_model.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()`
                },
                {
                    title: "Model Evaluation",
                    content: `Evaluating your model properly is crucial to understand its performance and limitations.

<strong>Key Metrics for Regression:</strong>

1. <strong>R² Score (Coefficient of Determination):</strong>
   • Range: -∞ to 1
   • 1 = Perfect predictions
   • 0 = Model as good as mean
   • < 0 = Worse than mean

2. <strong>Mean Absolute Error (MAE):</strong>
   • Average absolute difference
   • Easy to interpret
   • Less sensitive to outliers

3. <strong>Root Mean Squared Error (RMSE):</strong>
   • Penalizes large errors
   • Same units as target
   • Most common metric

4. <strong>Mean Absolute Percentage Error (MAPE):</strong>
   • Percentage-based
   • Easy to understand
   • Scale-independent

<strong>Visualization Techniques:</strong>
• Actual vs Predicted plots
• Residual plots
• Learning curves
• Feature importance`,
                    code: `import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Sample predictions
y_true = np.array([3, 5, 7, 9, 11, 13, 15])
y_pred = np.array([2.8, 5.2, 6.9, 9.1, 10.8, 13.2, 15.1])

# Calculate all metrics
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("Model Evaluation Metrics:")
    print("-" * 40)
    print("R² Score:   ", round(r2, 4))
    print("MSE:        ", round(mse, 4))
    print("RMSE:       ", round(rmse, 4))
    print("MAE:        ", round(mae, 4))
    print("MAPE:       ", round(mape, 2), "%")
    
    return {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}

metrics = evaluate_model(y_true, y_pred)

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 4))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], 
         [y_true.min(), y_true.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_true - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 3: Residual Distribution
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()

# Cross-validation for robust evaluation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([3, 5, 7, 9, 11, 13, 15])

model = LinearRegression()

# 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, 
                            cv=5, 
                            scoring='r2')

print("\\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", round(cv_scores.mean(), 4), "(+/-", round(cv_scores.std(), 4), ")")`
                },
                {
                    title: "Real-world Examples",
                    content: `Let's apply linear regression to real-world scenarios with complete end-to-end examples.

<strong>Example 1: House Price Prediction</strong>
Predict house prices based on size, bedrooms, and location.

<strong>Example 2: Sales Forecasting</strong>
Predict product sales based on advertising spend.

<strong>Example 3: Student Performance</strong>
Predict exam scores based on study hours and attendance.

<strong>Real-World Considerations:</strong>

• <strong>Feature Engineering:</strong>
  - Create polynomial features
  - Handle categorical variables
  - Deal with missing data

• <strong>Data Quality:</strong>
  - Remove outliers
  - Handle multicollinearity
  - Check assumptions

• <strong>Model Limitations:</strong>
  - Linear relationships only
  - Sensitive to outliers
  - Assumes independence

• <strong>Production Deployment:</strong>
  - Save model (pickle/joblib)
  - Version control
  - Monitor performance`,
                    code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Example 1: House Price Prediction
print("=" * 50)
print("EXAMPLE 1: House Price Prediction")
print("=" * 50)

# Create synthetic dataset
np.random.seed(42)
n_samples = 200

house_data = pd.DataFrame({
    'size_sqft': np.random.randint(1000, 3500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age_years': np.random.randint(0, 50, n_samples),
    'distance_to_city': np.random.uniform(1, 30, n_samples)
})

# Target: Price (with some noise)
house_data['price'] = (
    300 * house_data['size_sqft'] + 
    50000 * house_data['bedrooms'] - 
    1000 * house_data['age_years'] - 
    2000 * house_data['distance_to_city'] + 
    np.random.normal(0, 50000, n_samples)
)

# Prepare data
X = house_data[['size_sqft', 'bedrooms', 'age_years', 'distance_to_city']]
y = house_data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", round(r2, 4))
print("RMSE: $" + str(round(rmse, 2)))
print("\\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_):
    print("  " + feature + ":", round(coef, 2))

# Example prediction
new_house = np.array([[2500, 3, 10, 5]])  # 2500 sqft, 3 bed, 10 years, 5 miles
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)[0]
print("\\nPredicted price for new house: $" + str(round(predicted_price, 2)))

# Example 2: Sales Forecasting
print("\\n" + "=" * 50)
print("EXAMPLE 2: Sales Forecasting")
print("=" * 50)

# Create sales dataset
months = 24
sales_data = pd.DataFrame({
    'tv_ad_spend': np.random.randint(1000, 10000, months),
    'radio_ad_spend': np.random.randint(500, 5000, months),
    'social_media_spend': np.random.randint(300, 3000, months)
})

# Target: Sales
sales_data['sales'] = (
    0.05 * sales_data['tv_ad_spend'] + 
    0.08 * sales_data['radio_ad_spend'] + 
    0.12 * sales_data['social_media_spend'] + 
    np.random.normal(0, 100, months)
)

X_sales = sales_data[['tv_ad_spend', 'radio_ad_spend', 'social_media_spend']]
y_sales = sales_data['sales']

# Train model
sales_model = LinearRegression()
sales_model.fit(X_sales, y_sales)

# ROI Analysis
print("\\nROI per $1 spent:")
for channel, coef in zip(X_sales.columns, sales_model.coef_):
    print("  " + channel + ": $" + str(round(coef, 4)))

# Optimal budget allocation
total_budget = 10000
print("\\nFor $" + str(total_budget) + " budget:")
print("Recommendation: Allocate more to highest ROI channel")

# Save model for production
import joblib
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("\\n✅ Models saved for production use!")

# Load and use saved model
loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('feature_scaler.pkl')
print("✅ Models loaded successfully!")`
                }
            ]
        },
        {
            number: "Module 2",
            title: "Logistic Regression",
            description: "An introduction to logistic regression, where ML models are designed to predict the probability of a given outcome.",
            duration: "50 min",
            lessons: "9 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Binary Classification Basics",
                "The Sigmoid Function",
                "Log Loss Explained",
                "Decision Boundaries",
                "Probability Interpretation",
                "Regularization Techniques",
                "Multi-class Classification",
                "Practical Applications"
            ],
            detailedDescription: "Logistic regression extends linear regression to classification problems. Learn how to predict probabilities, understand the sigmoid function, and work with binary and multi-class classification problems. This module covers everything from theory to practical implementation.",
            detailedContent: [
                {
                    title: "Binary Classification Basics",
                    content: `Binary classification is the task of predicting one of two possible classes (e.g., yes/no, spam/not spam, fraud/legitimate).

<strong>Why Not Linear Regression?</strong>
• Linear regression outputs any real number (-∞ to +∞)
• Classification needs probabilities (0 to 1)
• Linear regression is sensitive to outliers for classification

<strong>The Logistic Regression Idea:</strong>
• Compute a linear score: z = w·x + b
• Squash it into [0, 1] using the sigmoid function
• Interpret the result as P(class = 1)

<strong>Making a Decision:</strong>
• If probability >= 0.5, predict class 1
• If probability < 0.5, predict class 0
• The 0.5 threshold can be tuned based on the problem

<strong>Common Use Cases:</strong>
• <strong>Email:</strong> Spam vs not spam
• <strong>Medicine:</strong> Disease present vs absent
• <strong>Finance:</strong> Loan default vs no default
• <strong>Marketing:</strong> Will click vs will not click`,
                    code: `from sklearn.linear_model import LogisticRegression
import numpy as np

# Features: [hours_studied, hours_slept]
X = np.array([[2, 8], [4, 7], [6, 6], [8, 5], [1, 9], [7, 6]])
y = np.array([0, 0, 1, 1, 0, 1])  # 0=Fail, 1=Pass

# Train a binary classifier
model = LogisticRegression()
model.fit(X, y)

# Predict class and probability for a new student
new_student = [[5, 7]]
predicted_class = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0][1]

print("Predicted class:", "Pass" if predicted_class else "Fail")
print("Probability of passing: {:.0%}".format(probability))`
                },
                {
                    title: "The Sigmoid Function",
                    content: `The sigmoid (logistic) function is the heart of logistic regression. It maps any real number to a value between 0 and 1.

<strong>The Formula:</strong>
sigmoid(z) = 1 / (1 + e^(-z))

<strong>Key Properties:</strong>
• Output always between 0 and 1 (a valid probability)
• sigmoid(0) = 0.5
• Large positive z → output near 1
• Large negative z → output near 0
• Smooth and differentiable (great for gradient descent)

<strong>S-Shaped Curve:</strong>
The function has a characteristic "S" shape. Near the center it changes quickly; at the extremes it saturates (flattens out).

<strong>From Score to Probability:</strong>
z = w₁x₁ + w₂x₂ + ... + b   (linear score)
p = sigmoid(z)              (probability of class 1)`,
                    code: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Try a range of inputs
values = [-5, -2, -1, 0, 1, 2, 5]
for z in values:
    print("sigmoid({:>2}) = {:.4f}".format(z, sigmoid(z)))

# Output:
# sigmoid(-5) = 0.0067
# sigmoid( 0) = 0.5000
# sigmoid( 5) = 0.9933

# Converting a linear score into a probability
w = np.array([0.5, -0.3])
x = np.array([4, 2])
b = 0.1
z = np.dot(w, x) + b
probability = sigmoid(z)
print("Probability of class 1: {:.2%}".format(probability))`
                },
                {
                    title: "Log Loss Explained",
                    content: `Log Loss (also called binary cross-entropy) is the loss function used to train logistic regression. It measures how far predicted probabilities are from the true labels.

<strong>The Formula:</strong>
LogLoss = -(1/n) Σ [ y·log(p) + (1-y)·log(1-p) ]

Where:
• y = actual label (0 or 1)
• p = predicted probability of class 1

<strong>Why Not Use MSE?</strong>
• MSE with sigmoid creates a non-convex loss surface (many local minima)
• Log loss is convex → gradient descent finds the global minimum
• Log loss heavily penalizes confident wrong predictions

<strong>Intuition:</strong>
• Correct and confident (p=0.99, y=1) → tiny loss
• Wrong and confident (p=0.99, y=0) → huge loss
• Uncertain (p=0.5) → moderate loss

This encourages the model to be both accurate and well-calibrated.`,
                    code: `import numpy as np

def log_loss(y_true, y_pred):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

y_true = np.array([1, 0, 1, 1, 0])

# Good predictions (close to truth)
good = np.array([0.9, 0.1, 0.8, 0.95, 0.05])
# Bad predictions (confidently wrong)
bad = np.array([0.1, 0.9, 0.2, 0.05, 0.95])

print("Good predictions log loss:", round(log_loss(y_true, good), 4))
print("Bad predictions log loss: ", round(log_loss(y_true, bad), 4))

# Bad predictions produce a much larger loss`
                },
                {
                    title: "Decision Boundaries",
                    content: `A decision boundary is the surface that separates the predicted classes. For logistic regression it is defined by where the probability equals the threshold (usually 0.5).

<strong>Where p = 0.5:</strong>
sigmoid(z) = 0.5  happens when  z = 0
So the boundary is:  w·x + b = 0

<strong>Shape of the Boundary:</strong>
• With raw features → a straight line (linear boundary)
• Logistic regression is a <strong>linear classifier</strong>
• Adding polynomial/interaction features → curved boundaries

<strong>Interpreting the Boundary:</strong>
• Points on one side → class 1
• Points on the other side → class 0
• Points near the boundary → uncertain (probability near 0.5)

<strong>Moving the Threshold:</strong>
Changing the decision threshold shifts the boundary, trading off false positives against false negatives.`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression

# Two clearly separated groups
X = np.array([[1, 1], [1, 2], [2, 1],
              [6, 6], [7, 5], [5, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# The boundary satisfies: w1*x1 + w2*x2 + b = 0
w1, w2 = model.coef_[0]
b = model.intercept_[0]
print("Boundary equation:")
print("  {:.2f}*x1 + {:.2f}*x2 + {:.2f} = 0".format(w1, w2, b))

# Classify a point near the middle
point = [[4, 4]]
print("Probability class 1:", round(model.predict_proba(point)[0][1], 3))
print("Predicted class:", model.predict(point)[0])`
                },
                {
                    title: "Probability Interpretation",
                    content: `A major advantage of logistic regression is that it outputs calibrated probabilities, not just hard class labels.

<strong>What the Output Means:</strong>
• p = 0.85 means "85% confident this is class 1"
• This confidence is useful for ranking and risk-based decisions

<strong>Odds and Log-Odds:</strong>
• Odds = p / (1 - p)
• Log-odds (logit) = log(odds) = z = w·x + b
• Logistic regression is linear in the log-odds

<strong>Interpreting Coefficients:</strong>
• Each weight tells how a feature changes the log-odds
• exp(weight) = how much the odds multiply per unit increase
• Positive weight → increases probability of class 1
• Negative weight → decreases probability of class 1

<strong>Why Probabilities Matter:</strong>
• Set custom thresholds for different costs
• Rank predictions (e.g., most likely buyers first)
• Combine with business rules and expected value`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression

# Predict loan default from [income_k, debt_ratio]
X = np.array([[80, 0.2], [40, 0.6], [100, 0.1],
              [30, 0.8], [60, 0.4], [25, 0.9]])
y = np.array([0, 1, 0, 1, 0, 1])  # 1 = default

model = LogisticRegression()
model.fit(X, y)

# Coefficients as odds multipliers
for feature, coef in zip(["income_k", "debt_ratio"], model.coef_[0]):
    print("{}: weight={:.3f}, odds x{:.3f} per unit".format(
        feature, coef, np.exp(coef)))

# Probability-based decision with a custom threshold
applicant = [[45, 0.5]]
prob_default = model.predict_proba(applicant)[0][1]
threshold = 0.3  # be cautious: flag anything above 30%
print("Default probability: {:.1%}".format(prob_default))
print("Decision:", "Reject" if prob_default > threshold else "Approve")`
                },
                {
                    title: "Regularization Techniques",
                    content: `Regularization prevents logistic regression from overfitting by discouraging overly large weights.

<strong>Why Regularize?</strong>
• Large weights → model too confident and fits noise
• Small weights → smoother, more generalizable model
• Especially important with many features

<strong>L2 Regularization (Ridge):</strong>
• Adds penalty proportional to sum of squared weights
• Shrinks all weights toward zero (but rarely to exactly zero)
• Default in scikit-learn

<strong>L1 Regularization (Lasso):</strong>
• Adds penalty proportional to sum of absolute weights
• Drives some weights to exactly zero → feature selection
• Produces sparse, interpretable models

<strong>The C Parameter (scikit-learn):</strong>
• C is the inverse of regularization strength
• Small C → strong regularization (simpler model)
• Large C → weak regularization (fits data closely)
• Tune C with cross-validation`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a dataset with some noise features
X, y = make_classification(n_samples=300, n_features=20,
                           n_informative=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42)

# Compare regularization strengths
for C in [0.01, 0.1, 1, 10]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    n_small = np.sum(np.abs(model.coef_[0]) < 0.01)
    print("C={:>5}: accuracy={:.3f}, near-zero weights={}".format(
        C, acc, n_small))

# L1 for feature selection
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
l1_model.fit(X_tr, y_tr)
kept = np.sum(l1_model.coef_[0] != 0)
print("\\nL1 kept {} of 20 features".format(kept))`
                },
                {
                    title: "Multi-class Classification",
                    content: `Logistic regression naturally handles two classes, but it can be extended to problems with more than two classes.

<strong>One-vs-Rest (OvR):</strong>
• Train one binary classifier per class ("this class vs everything else")
• For N classes, train N classifiers
• Predict the class with the highest probability
• Simple and widely used

<strong>Softmax (Multinomial) Regression:</strong>
• A single model that outputs a probability for every class at once
• Uses the softmax function instead of sigmoid
• Probabilities across all classes sum to 1
• Often more accurate than OvR

<strong>The Softmax Function:</strong>
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)

<strong>Choosing an Approach:</strong>
• OvR: simple, parallelizable, good baseline
• Softmax: preferred when classes are mutually exclusive
• scikit-learn picks a sensible default automatically`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Iris: 3 flower classes
data = load_iris()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42)

# Multinomial (softmax) logistic regression
model = LogisticRegression(multi_class='multinomial',
                           max_iter=500)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("Accuracy: {:.0%}".format(accuracy_score(y_te, pred)))

# Probabilities for each class sum to 1
sample = X_te[0].reshape(1, -1)
probs = model.predict_proba(sample)[0]
for name, p in zip(data.target_names, probs):
    print("  {}: {:.1%}".format(name, p))
print("  Sum:", round(probs.sum(), 4))`
                },
                {
                    title: "Practical Applications",
                    content: `Logistic regression remains one of the most widely used algorithms in industry because it is fast, interpretable, and produces probabilities.

<strong>Real-World Uses:</strong>
• <strong>Healthcare:</strong> Predict disease risk from patient data
• <strong>Finance:</strong> Credit scoring and fraud detection
• <strong>Marketing:</strong> Predict click-through and conversion
• <strong>HR:</strong> Predict employee churn
• <strong>Manufacturing:</strong> Predict equipment failure

<strong>Why It Is Popular:</strong>
• Trains quickly, even on large datasets
• Coefficients are interpretable
• Outputs probabilities for risk-based decisions
• A strong baseline before trying complex models

<strong>Best Practices:</strong>
• Scale/standardize numeric features
• Encode categorical features properly
• Use regularization to avoid overfitting
• Evaluate with precision/recall, not just accuracy
• Tune the decision threshold for your business cost

<strong>Limitations:</strong>
• Assumes a roughly linear decision boundary (in feature space)
• May underperform on complex, non-linear patterns`,
                    code: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Customer churn prediction
# Features: [tenure_months, monthly_charge, support_tickets]
X = np.array([
    [24, 50, 1], [3, 90, 5], [36, 40, 0], [1, 100, 6],
    [48, 30, 0], [2, 95, 4], [60, 25, 1], [5, 85, 3]
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # 1 = churned

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)

# Pipeline: scale + classify
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_tr, y_tr)

print(classification_report(y_te, pipe.predict(X_te),
                            zero_division=0))

# Score a new customer's churn risk
new_customer = [[4, 88, 4]]
risk = pipe.predict_proba(new_customer)[0][1]
print("Churn risk: {:.0%}".format(risk))
print("Action:", "Offer retention deal" if risk > 0.5 else "Monitor")`
                }
            ]
        },
        {
            number: "Module 3",
            title: "Classification",
            description: "An introduction to binary classification models, covering thresholding, confusion matrices, and metrics like accuracy, precision, recall, and AUC.",
            duration: "55 min",
            lessons: "10 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Classification Fundamentals",
                "Confusion Matrix Deep Dive",
                "Accuracy vs Precision vs Recall",
                "F1 Score and Trade-offs",
                "ROC Curves",
                "AUC Metric",
                "Class Imbalance Handling",
                "Model Evaluation Strategies"
            ],
            detailedDescription: "Master the art of classification by understanding key metrics and evaluation techniques. Learn when to use accuracy, precision, or recall, how to interpret confusion matrices, and work with ROC curves to evaluate your classification models effectively.",
            detailedContent: [
                {
                    title: "Classification Fundamentals",
                    content: `Classification is the task of assigning inputs to discrete categories. Choosing the right evaluation approach is as important as choosing the model.

<strong>Types of Classification:</strong>
• <strong>Binary:</strong> Two classes (spam / not spam)
• <strong>Multi-class:</strong> One label from many (cat / dog / bird)
• <strong>Multi-label:</strong> Multiple labels at once (tags on an article)

<strong>The Prediction Threshold:</strong>
• Models output a probability or score
• A threshold converts the score to a class
• Default is 0.5, but it should be chosen deliberately

<strong>Why Accuracy Is Not Enough:</strong>
• On imbalanced data, "always predict the majority" can look accurate
• Example: 99% legitimate transactions → 99% accuracy by never catching fraud
• We need metrics that reveal what the model misses

<strong>The Evaluation Toolkit:</strong>
• Confusion matrix (the foundation)
• Precision, recall, F1
• ROC curve and AUC
• Threshold analysis`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Imbalanced dataset: mostly class 0
np.random.seed(0)
X = np.random.randn(1000, 3)
y = (X[:, 0] + X[:, 1] > 2.5).astype(int)  # rare positive class
print("Class balance:", np.bincount(y))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)
model = LogisticRegression()
model.fit(X_tr, y_tr)

# A naive "always predict 0" baseline
baseline_acc = np.mean(y_te == 0)
model_acc = np.mean(model.predict(X_te) == y_te)
print("Always-predict-0 accuracy: {:.2%}".format(baseline_acc))
print("Model accuracy:            {:.2%}".format(model_acc))
# High accuracy alone can be misleading!`
                },
                {
                    title: "Confusion Matrix Deep Dive",
                    content: `The confusion matrix is the foundation of classification evaluation. It breaks predictions into four categories for binary problems.

<strong>The Four Outcomes:</strong>
• <strong>True Positive (TP):</strong> Predicted 1, actually 1 ✓
• <strong>True Negative (TN):</strong> Predicted 0, actually 0 ✓
• <strong>False Positive (FP):</strong> Predicted 1, actually 0 ✗ (false alarm)
• <strong>False Negative (FN):</strong> Predicted 0, actually 1 ✗ (missed detection)

<strong>Layout:</strong>
                Predicted 0    Predicted 1
Actual 0          TN             FP
Actual 1          FN             TP

<strong>Which Errors Matter More?</strong>
• <strong>Medical screening:</strong> False negatives are dangerous (missed disease)
• <strong>Spam filter:</strong> False positives are annoying (lost real email)
• The cost of each error type drives metric choice

<strong>Everything Derives From It:</strong>
Accuracy, precision, recall, and F1 are all computed from these four numbers.`,
                    code: `import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)
print()
print("True Negatives: ", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives: ", tp)

# Derived metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
print("\\nAccuracy: {:.2%}".format(accuracy))`
                },
                {
                    title: "Accuracy vs Precision vs Recall",
                    content: `These three metrics answer different questions. Knowing which one matters is key to real-world ML.

<strong>Accuracy:</strong>
• "What fraction of all predictions were correct?"
• Accuracy = (TP + TN) / Total
• Misleading on imbalanced datasets

<strong>Precision:</strong>
• "Of everything I flagged as positive, how much really was?"
• Precision = TP / (TP + FP)
• High precision → few false alarms
• Optimize when false positives are costly (spam filter)

<strong>Recall (Sensitivity):</strong>
• "Of all actual positives, how many did I catch?"
• Recall = TP / (TP + FN)
• High recall → few misses
• Optimize when false negatives are costly (disease detection)

<strong>The Trade-off:</strong>
• Raising the threshold → higher precision, lower recall
• Lowering the threshold → higher recall, lower precision
• You usually cannot maximize both at once`,
                    code: `import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)

print("Accuracy:  {:.2%}".format(acc))
print("Precision: {:.2%}  (of flagged positives, how many correct)".format(prec))
print("Recall:    {:.2%}  (of real positives, how many caught)".format(rec))

# Scenario guidance
print("\\nCancer screening -> maximize RECALL (don't miss cases)")
print("Spam filter      -> maximize PRECISION (don't block real mail)")`
                },
                {
                    title: "F1 Score and Trade-offs",
                    content: `The F1 score combines precision and recall into a single number, useful when you need to balance both.

<strong>The Formula:</strong>
F1 = 2 × (Precision × Recall) / (Precision + Recall)

<strong>Why the Harmonic Mean?</strong>
• The harmonic mean punishes imbalance
• High F1 requires BOTH precision and recall to be good
• If either is near zero, F1 is near zero

<strong>F1 vs Accuracy:</strong>
• F1 focuses on the positive class
• Better than accuracy for imbalanced problems
• Ignores true negatives (often what we want)

<strong>The Fβ Generalization:</strong>
• Fβ weights recall β times as much as precision
• F2 → recall matters more (medical)
• F0.5 → precision matters more (recommendations)

<strong>When to Use:</strong>
• Imbalanced datasets
• When both false positives and false negatives matter
• Comparing models with a single number`,
                    code: `import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision: {:.3f}".format(p))
print("Recall:    {:.3f}".format(r))
print("F1:        {:.3f}".format(f1))

# Emphasize recall (F2) vs precision (F0.5)
print("F2  (recall-weighted):    {:.3f}".format(
    fbeta_score(y_true, y_pred, beta=2)))
print("F0.5 (precision-weighted):{:.3f}".format(
    fbeta_score(y_true, y_pred, beta=0.5)))`
                },
                {
                    title: "ROC Curves",
                    content: `The ROC (Receiver Operating Characteristic) curve visualizes classifier performance across all possible thresholds.

<strong>The Axes:</strong>
• X-axis: False Positive Rate = FP / (FP + TN)
• Y-axis: True Positive Rate (Recall) = TP / (TP + FN)

<strong>How It Is Built:</strong>
• Sweep the threshold from 1 down to 0
• At each threshold compute TPR and FPR
• Plot the points to form the curve

<strong>Reading the Curve:</strong>
• Top-left corner = perfect classifier
• Diagonal line = random guessing
• The more the curve hugs the top-left, the better

<strong>Choosing a Threshold:</strong>
• Each point on the curve is one threshold
• Pick the point matching your tolerance for false alarms
• The ROC curve makes the trade-off visible

<strong>Advantage:</strong>
ROC is threshold-independent, so it evaluates the model's ranking ability overall rather than at one cutoff.`,
                    code: `import numpy as np
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, weights=[0.7, 0.3],
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42)
model = LogisticRegression().fit(X_tr, y_tr)

# Probability scores for the positive class
scores = model.predict_proba(X_te)[:, 1]
fpr, tpr, thresholds = roc_curve(y_te, scores)

print("Sample points on the ROC curve:")
for i in range(0, len(thresholds), max(1, len(thresholds)//5)):
    print("  threshold={:.2f} -> FPR={:.2f}, TPR={:.2f}".format(
        thresholds[i], fpr[i], tpr[i]))

# import matplotlib.pyplot as plt
# plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
# plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')`
                },
                {
                    title: "AUC Metric",
                    content: `AUC (Area Under the ROC Curve) summarizes the entire ROC curve as a single number between 0 and 1.

<strong>Interpretation:</strong>
• AUC = 1.0 → perfect classifier
• AUC = 0.5 → no better than random
• AUC < 0.5 → worse than random (predictions inverted)

<strong>Probabilistic Meaning:</strong>
AUC is the probability that the model ranks a random positive example higher than a random negative example.

<strong>Why AUC Is Useful:</strong>
• Threshold-independent (measures ranking quality)
• Works well on imbalanced data
• Single number to compare models

<strong>General Guidelines:</strong>
• 0.9 - 1.0 → excellent
• 0.8 - 0.9 → good
• 0.7 - 0.8 → fair
• 0.6 - 0.7 → poor

<strong>PR-AUC Alternative:</strong>
For highly imbalanced problems, the area under the Precision-Recall curve (PR-AUC) is often more informative than ROC-AUC.`,
                    code: `import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=800, weights=[0.8, 0.2],
                           random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=1)

for name, clf in [("LogReg", LogisticRegression()),
                  ("Forest", RandomForestClassifier(random_state=1))]:
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, scores)
    pr = average_precision_score(y_te, scores)  # PR-AUC
    print("{}: ROC-AUC={:.3f}, PR-AUC={:.3f}".format(name, roc, pr))`
                },
                {
                    title: "Class Imbalance Handling",
                    content: `Class imbalance occurs when one class vastly outnumbers another (fraud, disease, defects). Standard training tends to ignore the rare class.

<strong>The Problem:</strong>
• The model optimizes overall accuracy
• Predicting the majority class always is "accurate" but useless
• The rare (often important) class gets missed

<strong>Data-Level Techniques:</strong>
• <strong>Oversampling:</strong> Duplicate/synthesize minority examples (SMOTE)
• <strong>Undersampling:</strong> Remove majority examples
• <strong>Combined:</strong> Mix both approaches

<strong>Algorithm-Level Techniques:</strong>
• <strong>Class weights:</strong> Penalize minority mistakes more
• <strong>Threshold tuning:</strong> Lower the decision threshold
• <strong>Ensemble methods:</strong> Balanced bagging/boosting

<strong>Right Metrics:</strong>
• Use precision, recall, F1, PR-AUC — not accuracy
• Look at the confusion matrix for the minority class

<strong>Practical Tip:</strong>
Start with class_weight='balanced' — it is simple and often effective.`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

# Severe imbalance: 5% positive
X, y = make_classification(n_samples=2000, weights=[0.95, 0.05],
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42)

# Without vs with class weighting
plain = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
weighted = LogisticRegression(class_weight='balanced',
                              max_iter=500).fit(X_tr, y_tr)

for name, model in [("Default", plain), ("Balanced", weighted)]:
    pred = model.predict(X_te)
    print("{}: recall={:.2f}, F1={:.2f}".format(
        name, recall_score(y_te, pred), f1_score(y_te, pred)))
# Balanced weighting catches far more of the rare class`
                },
                {
                    title: "Model Evaluation Strategies",
                    content: `Reliable evaluation ensures your reported performance reflects real-world behavior on unseen data.

<strong>Train / Validation / Test Split:</strong>
• Train: fit the model
• Validation: tune hyperparameters and threshold
• Test: final, one-time performance estimate

<strong>Cross-Validation:</strong>
• Split data into k folds; train k times
• Each fold serves as validation once
• Averages out lucky/unlucky splits
• Use <strong>stratified</strong> k-fold for classification to preserve class balance

<strong>Avoiding Data Leakage:</strong>
• Fit scalers/encoders on training data only
• Do preprocessing inside a pipeline
• Never let test information influence training

<strong>Reporting Results:</strong>
• Show a full classification report (precision/recall/F1)
• Include the confusion matrix
• Report AUC for ranking quality
• State the threshold you used

<strong>Beyond Accuracy:</strong>
Always evaluate against a baseline and consider the business cost of each error type.`,
                    code: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, weights=[0.7, 0.3],
                           random_state=0)

# Pipeline avoids leakage: scaler fit only on each train fold
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500))
])

# Stratified 5-fold cross-validation, scored by F1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1')

print("F1 per fold:", np.round(scores, 3))
print("Mean F1: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))`
                }
            ]
        }
    ],
    data: [
        {
            number: "Module 4",
            title: "Working with Numerical Data",
            description: "Learn how to analyze and transform numerical data to help train ML models more effectively.",
            duration: "40 min",
            lessons: "7 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Feature Scaling and Normalization",
                "Handling Missing Values",
                "Outlier Detection",
                "Data Distribution Analysis",
                "Feature Engineering",
                "Binning and Discretization",
                "Best Practices for Numerical Features"
            ],
            detailedDescription: "Numerical data is the backbone of most ML models. This module teaches you how to properly prepare, transform, and engineer numerical features to improve model performance. Learn about normalization, standardization, and advanced preprocessing techniques.",
            detailedContent: [
                {
                    title: "Feature Scaling and Normalization",
                    content: `Many ML algorithms are sensitive to the scale of features. Scaling puts all features on comparable ranges so none dominates.

<strong>Why Scaling Matters:</strong>
• Gradient descent converges faster
• Distance-based models (KNN, SVM) treat features fairly
• Regularization penalizes weights evenly

<strong>Min-Max Normalization:</strong>
• Rescales to a fixed range, usually [0, 1]
• x' = (x - min) / (max - min)
• Good when you need bounded values
• Sensitive to outliers

<strong>Standardization (Z-score):</strong>
• Centers to mean 0, standard deviation 1
• x' = (x - mean) / std
• Works well when data is roughly normal
• Less affected by outliers than min-max

<strong>Robust Scaling:</strong>
• Uses median and interquartile range
• Best when outliers are present

<strong>Golden Rule:</strong>
Fit the scaler on training data only, then apply the same transform to validation and test data.`,
                    code: `import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Feature with very different scale and an outlier
X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])

for name, scaler in [("MinMax", MinMaxScaler()),
                     ("Standard", StandardScaler()),
                     ("Robust", RobustScaler())]:
    scaled = scaler.fit_transform(X)
    print("{:>8}: {}".format(name, np.round(scaled.ravel(), 2)))

# Correct workflow: fit on train, transform test
from sklearn.model_selection import train_test_split
data = np.random.rand(100, 1) * 50
X_tr, X_te = train_test_split(data, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)   # fit ONLY on training
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)       # reuse same scaler
print("\\nTrain mean ~0:", round(X_tr_s.mean(), 3))`
                },
                {
                    title: "Handling Missing Values",
                    content: `Real datasets are rarely complete. How you handle missing values can strongly affect model quality.

<strong>Why Values Go Missing:</strong>
• Data entry errors or sensor failures
• Optional fields not filled in
• Merging datasets with different coverage

<strong>Deletion Strategies:</strong>
• <strong>Drop rows:</strong> Simple, but loses data
• <strong>Drop columns:</strong> Only if mostly missing
• Risky when missingness is not random

<strong>Imputation Strategies:</strong>
• <strong>Mean/Median:</strong> Simple, median is outlier-robust
• <strong>Mode:</strong> For discrete numeric values
• <strong>Constant:</strong> A sentinel value (e.g., 0 or -1)
• <strong>KNN imputation:</strong> Use similar rows
• <strong>Model-based:</strong> Predict the missing value

<strong>Missingness as a Signal:</strong>
Add a binary "was_missing" indicator column — sometimes the fact that a value is missing is itself predictive.

<strong>Best Practice:</strong>
Impute inside a pipeline, fitting on training data only.`,
                    code: `import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.DataFrame({
    'age':    [25, np.nan, 35, 40, np.nan, 28],
    'income': [50, 60, np.nan, 80, 55, 62]
})
print("Missing per column:\\n", df.isna().sum(), "\\n")

# Median imputation (robust to outliers)
median_imp = SimpleImputer(strategy='median')
filled = median_imp.fit_transform(df)
print("Median-imputed:\\n", np.round(filled, 1), "\\n")

# Add a missingness indicator before imputing
df['age_missing'] = df['age'].isna().astype(int)

# KNN imputation uses similar rows
knn = KNNImputer(n_neighbors=2)
print("KNN-imputed:\\n", np.round(knn.fit_transform(df), 1))`
                },
                {
                    title: "Outlier Detection",
                    content: `Outliers are values far from the rest of the data. They can distort models and metrics, but sometimes they are the most important signal (fraud, defects).

<strong>Causes of Outliers:</strong>
• Measurement or data-entry errors
• Genuinely rare but valid events
• Different populations mixed together

<strong>Detection Methods:</strong>
• <strong>Z-score:</strong> Flag values more than ~3 std from the mean
• <strong>IQR rule:</strong> Outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
• <strong>Isolation Forest:</strong> Model-based anomaly detection
• <strong>Visualization:</strong> Box plots and scatter plots

<strong>What to Do With Them:</strong>
• <strong>Remove:</strong> If clearly an error
• <strong>Cap (winsorize):</strong> Clip to a threshold
• <strong>Transform:</strong> Log transform to reduce impact
• <strong>Keep:</strong> If they carry real signal

<strong>Caution:</strong>
Never remove outliers blindly — investigate why they exist first.`,
                    code: `import numpy as np

data = np.array([10, 12, 11, 13, 12, 11, 95, 10, 12, 11])

# Z-score method
mean, std = data.mean(), data.std()
z = np.abs((data - mean) / std)
print("Z-score outliers:", data[z > 2])

# IQR method
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("IQR bounds: [{:.1f}, {:.1f}]".format(low, high))
print("IQR outliers:", data[(data < low) | (data > high)])

# Capping (winsorization) instead of removal
capped = np.clip(data, low, high)
print("Capped data:", np.round(capped, 1))`
                },
                {
                    title: "Data Distribution Analysis",
                    content: `Understanding how a feature is distributed guides which transformations and models will work best.

<strong>Key Properties to Inspect:</strong>
• <strong>Central tendency:</strong> Mean, median, mode
• <strong>Spread:</strong> Variance, standard deviation, range
• <strong>Shape:</strong> Skewness and kurtosis
• <strong>Modality:</strong> One peak or several?

<strong>Skewness:</strong>
• Right-skewed (positive): long tail to the right (income)
• Left-skewed (negative): long tail to the left
• Many models assume roughly symmetric features

<strong>Fixing Skew:</strong>
• Log transform: compresses large values
• Square-root transform: milder effect
• Box-Cox / Yeo-Johnson: automatic power transforms

<strong>Why It Matters:</strong>
• Linear models like symmetric, normal-ish features
• Skewed targets can be log-transformed then reversed
• Reveals whether scaling or transformation is needed

<strong>Visual Tools:</strong>
Histograms, KDE plots, Q-Q plots, and box plots.`,
                    code: `import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# Highly right-skewed data (like income)
np.random.seed(0)
data = np.random.exponential(scale=1000, size=1000)

print("Mean:   {:.1f}".format(data.mean()))
print("Median: {:.1f}".format(np.median(data)))
print("Skew:   {:.2f}".format(stats.skew(data)))

# Log transform reduces skew
log_data = np.log1p(data)
print("\\nAfter log transform:")
print("Skew:   {:.2f}".format(stats.skew(log_data)))

# Yeo-Johnson finds the best power transform automatically
pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(data.reshape(-1, 1))
print("After Yeo-Johnson skew: {:.2f}".format(
    stats.skew(transformed.ravel())))`
                },
                {
                    title: "Feature Engineering",
                    content: `Feature engineering creates new input variables that expose patterns to the model. It is often more impactful than the choice of algorithm.

<strong>Common Techniques:</strong>
• <strong>Interactions:</strong> Multiply or combine features (price × quantity)
• <strong>Ratios:</strong> Debt-to-income, clicks-per-view
• <strong>Polynomial features:</strong> x, x², x³ for curvature
• <strong>Aggregations:</strong> Sum, mean, count over groups
• <strong>Date/time parts:</strong> Hour, day-of-week, is_weekend

<strong>Domain Knowledge:</strong>
• The best features come from understanding the problem
• Example: BMI = weight / height² beats raw weight and height

<strong>Transformations:</strong>
• Log for skewed values
• Differences and rolling windows for time series

<strong>Guiding Principles:</strong>
• Create features that relate to the target
• Avoid leakage (do not use future information)
• Remove redundant, highly correlated features
• Validate that new features actually help`,
                    code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    'price': [10, 20, 30, 40],
    'quantity': [5, 3, 8, 2],
    'timestamp': pd.to_datetime(
        ['2024-01-01 09:00', '2024-01-06 14:00',
         '2024-01-03 22:00', '2024-01-07 11:00'])
})

# Interaction and ratio features
df['revenue'] = df['price'] * df['quantity']
df['price_per_unit'] = df['price'] / df['quantity']

# Date-based features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['hour'] = df['timestamp'].dt.hour

print(df[['revenue', 'price_per_unit', 'is_weekend', 'hour']])

# Polynomial features expose curvature
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['price', 'quantity']])
print("\\nPolynomial feature names:", poly.get_feature_names_out())`
                },
                {
                    title: "Binning and Discretization",
                    content: `Binning converts a continuous feature into discrete buckets. This can help models capture non-linear patterns and reduce the impact of noise.

<strong>Why Bin?</strong>
• Capture non-linear relationships with linear models
• Reduce sensitivity to small fluctuations and outliers
• Create interpretable categories (age groups)

<strong>Binning Strategies:</strong>
• <strong>Equal-width:</strong> Same range per bin (0-10, 10-20, ...)
• <strong>Equal-frequency (quantile):</strong> Same count per bin
• <strong>Custom:</strong> Domain-driven boundaries
• <strong>K-means:</strong> Cluster values into bins

<strong>Encoding Bins:</strong>
• Ordinal: bins as ordered integers
• One-hot: each bin becomes a column

<strong>Trade-offs:</strong>
• Pro: robustness, interpretability, non-linearity
• Con: loses fine-grained information
• Too many bins → overfitting; too few → underfitting

<strong>Example Use:</strong>
Turning age into "child / teen / adult / senior" for a marketing model.`,
                    code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

ages = np.array([[5], [15], [25], [35], [45], [65], [80]])

# Custom, domain-driven bins
labels = ['child', 'teen', 'young_adult', 'adult', 'senior']
groups = pd.cut(ages.ravel(), bins=[0, 12, 19, 35, 60, 120],
                labels=labels)
print("Custom age groups:", list(groups))

# Equal-frequency (quantile) binning into 3 bins
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal',
                         strategy='quantile')
binned = kbins.fit_transform(ages)
print("Quantile bin indices:", binned.ravel().astype(int))
print("Bin edges:", np.round(kbins.bin_edges_[0], 1))`
                },
                {
                    title: "Best Practices for Numerical Features",
                    content: `Bringing it all together: a reliable workflow for preparing numerical data.

<strong>Recommended Order:</strong>
1. Explore distributions and spot problems
2. Handle missing values (impute + indicator)
3. Treat outliers (cap or transform)
4. Engineer new features
5. Transform skewed features
6. Scale/standardize
7. Do all of this inside a pipeline

<strong>Prevent Data Leakage:</strong>
• Fit every transformer on training data only
• Use scikit-learn Pipeline and ColumnTransformer
• Never peek at the test set during preprocessing

<strong>Match Preprocessing to the Model:</strong>
• Tree-based models: scaling usually not needed
• Linear/distance/neural models: scaling important
• Some models handle missing values natively (XGBoost)

<strong>Validate Everything:</strong>
• Compare model performance with and without a step
• Keep preprocessing reproducible and versioned

<strong>Document Choices:</strong>
Record why each transformation was applied so results are explainable.`,
                    code: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# A leakage-free preprocessing + model pipeline
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(max_iter=500))
])

# Data with missing values
np.random.seed(0)
X = np.random.randn(300, 5)
X[np.random.rand(*X.shape) < 0.1] = np.nan   # 10% missing
y = (X[:, 0] > 0).astype(int)
# fill target's driver so labels are defined
X[:, 0] = np.nan_to_num(X[:, 0])

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print("CV accuracy: {:.3f} (+/- {:.3f})".format(
    scores.mean(), scores.std()))
print("All preprocessing fit per-fold -> no leakage")`
                }
            ]
        },
        {
            number: "Module 5",
            title: "Working with Categorical Data",
            description: "Learn the fundamentals of working with categorical data: one-hot encoding, feature hashing, mean encoding, and feature crosses.",
            duration: "45 min",
            lessons: "8 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Understanding Categorical Variables",
                "One-Hot Encoding",
                "Label Encoding",
                "Feature Hashing Techniques",
                "Mean/Target Encoding",
                "Feature Crosses",
                "Handling High Cardinality",
                "Best Practices"
            ],
            detailedDescription: "Categorical data requires special handling in machine learning. This comprehensive module covers various encoding techniques, from basic one-hot encoding to advanced methods like feature hashing and mean encoding. Learn how to handle high-cardinality features and create meaningful feature crosses.",
            detailedContent: [
                {
                    title: "Understanding Categorical Variables",
                    content: `Categorical variables represent discrete groups or labels rather than numeric quantities. Models need them converted into numbers.

<strong>Types of Categorical Data:</strong>
• <strong>Nominal:</strong> No inherent order (color: red, green, blue)
• <strong>Ordinal:</strong> Meaningful order (size: small, medium, large)
• <strong>Binary:</strong> Two categories (yes/no)

<strong>Why Encoding Is Needed:</strong>
• ML algorithms operate on numbers
• The encoding method must preserve the right information
• Wrong encoding introduces false relationships

<strong>The Core Question:</strong>
Does the category have an order? 
• Ordinal → integer encoding that respects order
• Nominal → one-hot or other order-free encoding

<strong>Cardinality:</strong>
• Low cardinality: few unique values (weekdays)
• High cardinality: many unique values (zip codes, user IDs)
• Cardinality drives which technique works best

<strong>Watch Out For:</strong>
• Unseen categories at prediction time
• Rare categories that add noise
• Typos creating spurious categories`,
                    code: `import pandas as pd

df = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'red', 'green'],
    'size':  ['S', 'L', 'M', 'M', 'L'],
    'zip':   ['90210', '10001', '60601', '90210', '73301']
})

# Inspect cardinality
for col in df.columns:
    print("{}: {} unique values".format(col, df[col].nunique()))

# Identify column types
print("\\ncolor -> nominal (no order)")
print("size  -> ordinal (S < M < L)")
print("zip   -> high-cardinality nominal")`
                },
                {
                    title: "One-Hot Encoding",
                    content: `One-hot encoding creates a separate binary column for each category. It is the standard approach for nominal features with low cardinality.

<strong>How It Works:</strong>
• Each category becomes its own 0/1 column
• Exactly one column is 1 per row (the "hot" one)
• No false ordering is introduced

<strong>Example:</strong>
color = [red, green, blue] becomes:
is_red | is_green | is_blue
  1    |    0     |   0

<strong>The Dummy Variable Trap:</strong>
• With N categories, N-1 columns are enough
• Drop one column to avoid perfect collinearity
• Important for linear models (use drop='first')

<strong>Pros:</strong>
• No artificial order
• Works with any model
• Simple and interpretable

<strong>Cons:</strong>
• Explodes dimensionality for high cardinality
• Sparse matrices for many categories
• Not ideal for hundreds of unique values

<strong>Handle Unknowns:</strong>
Use handle_unknown='ignore' so new categories at predict time do not crash the pipeline.`,
                    code: `import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})

# pandas convenience method
print(pd.get_dummies(df, columns=['color']))

# scikit-learn encoder (production-friendly)
enc = OneHotEncoder(sparse_output=False,
                    handle_unknown='ignore')
encoded = enc.fit_transform(df[['color']])
print("\\nCategories:", enc.categories_[0])
print(encoded)

# Drop first to avoid the dummy variable trap
enc2 = OneHotEncoder(drop='first', sparse_output=False)
print("\\nDrop-first shape:", enc2.fit_transform(df[['color']]).shape)`
                },
                {
                    title: "Label Encoding",
                    content: `Label encoding maps each category to an integer. It is appropriate for ordinal data and for tree-based models.

<strong>How It Works:</strong>
• Assign 0, 1, 2, ... to each category
• small=0, medium=1, large=2

<strong>When It Is Correct:</strong>
• <strong>Ordinal features:</strong> Order is meaningful
• <strong>Tree-based models:</strong> Trees split on thresholds, so arbitrary integers are fine
• The target column in classification

<strong>When It Is Wrong:</strong>
• Nominal features with linear/distance models
• The model wrongly assumes blue(2) > green(1) > red(0)
• This invents relationships that do not exist

<strong>Ordinal Encoding:</strong>
• Explicitly specify the order for ordinal data
• Guarantees the integers respect the ranking

<strong>Rule of Thumb:</strong>
• Ordinal data → label/ordinal encoding
• Nominal data + linear model → one-hot
• Nominal data + tree model → label encoding is acceptable`,
                    code: `import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.DataFrame({'size': ['S', 'L', 'M', 'M', 'S', 'L']})

# Plain label encoding (alphabetical, order NOT guaranteed)
le = LabelEncoder()
print("LabelEncoder:", le.fit_transform(df['size']))

# Ordinal encoding with explicit, correct order
oe = OrdinalEncoder(categories=[['S', 'M', 'L']])
ordered = oe.fit_transform(df[['size']])
print("OrdinalEncoder (S<M<L):", ordered.ravel().astype(int))

# For nominal data + trees this is fine;
# for nominal data + linear models, prefer one-hot.`
                },
                {
                    title: "Feature Hashing Techniques",
                    content: `Feature hashing (the "hashing trick") maps categories to a fixed number of columns using a hash function. It is designed for very high cardinality.

<strong>The Problem It Solves:</strong>
• One-hot encoding of millions of categories is infeasible
• Hashing bounds the output size regardless of cardinality

<strong>How It Works:</strong>
• Apply a hash function to each category
• Use the hash value (mod n) as the column index
• Increment that column

<strong>Pros:</strong>
• Fixed, controllable dimensionality
• Memory efficient and fast
• Handles unseen categories automatically
• No need to store a category vocabulary

<strong>Cons:</strong>
• <strong>Collisions:</strong> Different categories may share a column
• Not interpretable (cannot reverse the hash)
• Some information loss

<strong>Choosing the Number of Buckets:</strong>
• More buckets → fewer collisions, more memory
• Tune as a hyperparameter

<strong>Use Cases:</strong>
Text tokens, user IDs, URLs, and other massive categorical spaces.`,
                    code: `from sklearn.feature_extraction import FeatureHasher

# High-cardinality categorical values
data = [{'user_id': 'user_8231'},
        {'user_id': 'user_45'},
        {'user_id': 'user_99012'},
        {'user_id': 'user_8231'}]  # repeat -> same hash

# Map into a fixed 8-column space
hasher = FeatureHasher(n_features=8, input_type='dict')
hashed = hasher.transform(data).toarray()

print("Fixed output shape:", hashed.shape)
print(hashed)
# Identical inputs (user_8231) produce identical rows`
                },
                {
                    title: "Mean/Target Encoding",
                    content: `Target encoding replaces each category with a statistic of the target variable (usually the mean) for that category. It is powerful for high-cardinality features.

<strong>How It Works:</strong>
• For each category, compute the average target value
• Replace the category with that average
• Example: city → average purchase rate in that city

<strong>Advantages:</strong>
• Single column regardless of cardinality
• Directly encodes predictive information
• Often boosts performance on high-cardinality data

<strong>The Big Risk: Overfitting/Leakage</strong>
• Naively using the target leaks information
• Rare categories get memorized

<strong>Safeguards:</strong>
• <strong>Smoothing:</strong> Blend category mean with the global mean
• <strong>Cross-fold encoding:</strong> Compute encoding out-of-fold
• <strong>Add noise:</strong> Regularize the encoded values

<strong>Smoothing Formula (intuition):</strong>
encoded = (count·category_mean + m·global_mean) / (count + m)
Rare categories lean toward the global mean.`,
                    code: `import numpy as np
import pandas as pd

df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC', 'SF'],
    'bought': [1, 0, 1, 1, 0, 0, 1]
})

global_mean = df['bought'].mean()
m = 2  # smoothing strength

# Smoothed target encoding
stats = df.groupby('city')['bought'].agg(['mean', 'count'])
stats['encoded'] = (
    (stats['count'] * stats['mean'] + m * global_mean) /
    (stats['count'] + m)
)
print(stats.round(3))

df['city_encoded'] = df['city'].map(stats['encoded'])
print("\\nEncoded feature:\\n", df[['city', 'city_encoded']].round(3))
# In practice, compute this out-of-fold to avoid leakage.`
                },
                {
                    title: "Feature Crosses",
                    content: `A feature cross combines two or more categorical features into a new one, letting linear models learn interactions.

<strong>Why Cross Features?</strong>
• Linear models cannot learn interactions on their own
• Sometimes the combination matters more than either part
• Example: (country, language) together predict behavior better than separately

<strong>How It Works:</strong>
• Concatenate categories: country_x_language
• "US_x_English", "US_x_Spanish", "MX_x_Spanish", ...
• Then encode the crossed feature (often one-hot or hashed)

<strong>Classic Example:</strong>
• latitude bins × longitude bins → location grid cells
• Captures neighborhood-level effects

<strong>Watch the Cardinality:</strong>
• Crossing multiplies the number of categories
• 50 states × 100 products = 5000 combinations
• Combine with hashing to bound the size

<strong>When to Use:</strong>
• Linear models needing interaction terms
• When domain knowledge suggests combinations matter
• Tree models learn crosses automatically, so less needed there`,
                    code: `import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    'country': ['US', 'US', 'MX', 'MX'],
    'language': ['EN', 'ES', 'ES', 'EN']
})

# Create the feature cross
df['country_x_language'] = df['country'] + '_x_' + df['language']
print(df)

# Encode the crossed feature
enc = OneHotEncoder(sparse_output=False)
crossed = enc.fit_transform(df[['country_x_language']])
print("\\nCrossed categories:", enc.categories_[0])
print(crossed)`
                },
                {
                    title: "Handling High Cardinality",
                    content: `High-cardinality features (thousands of unique values) require special strategies to avoid dimensionality explosion.

<strong>The Challenge:</strong>
• One-hot encoding creates thousands of sparse columns
• Slows training and risks overfitting
• Examples: user IDs, product SKUs, zip codes, URLs

<strong>Strategy 1: Target Encoding</strong>
• Collapse to a single informative column
• Use smoothing and out-of-fold computation

<strong>Strategy 2: Feature Hashing</strong>
• Bound dimensionality with the hashing trick
• Accept some collisions for scalability

<strong>Strategy 3: Grouping Rare Categories</strong>
• Merge infrequent values into an "Other" bucket
• Reduces noise from rarely seen categories

<strong>Strategy 4: Embeddings</strong>
• Learn a dense vector per category (neural networks)
• Captures semantic similarity between categories

<strong>Strategy 5: Frequency Encoding</strong>
• Replace category with how often it appears
• Simple and sometimes surprisingly effective

<strong>Choosing:</strong>
Balance interpretability, memory, and performance for your model type.`,
                    code: `import pandas as pd

# High-cardinality column with rare values
df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'C', 'A', 'D', 'E', 'A', 'B', 'F']
})

# Frequency encoding
freq = df['product'].value_counts(normalize=True)
df['product_freq'] = df['product'].map(freq)

# Group rare categories (appear < 2 times) into 'Other'
counts = df['product'].value_counts()
rare = counts[counts < 2].index
df['product_grouped'] = df['product'].replace(
    dict.fromkeys(rare, 'Other'))

print(df[['product', 'product_freq', 'product_grouped']].round(2))
print("\\nUnique before:", df['product'].nunique(),
      "-> after grouping:", df['product_grouped'].nunique())`
                },
                {
                    title: "Best Practices",
                    content: `A practical decision guide for encoding categorical data reliably.

<strong>Choosing an Encoder:</strong>
• Low-cardinality nominal → one-hot encoding
• Ordinal → ordinal encoding (specify order)
• High-cardinality + linear model → target encoding
• Very high cardinality → hashing or embeddings
• Tree-based models → label encoding is fine

<strong>Prevent Data Leakage:</strong>
• Fit encoders on training data only
• Target encoding must be computed out-of-fold
• Use ColumnTransformer inside a pipeline

<strong>Handle Unseen Categories:</strong>
• Use handle_unknown='ignore'
• Have a fallback (Other / global mean)
• Test with categories not in training

<strong>Keep It Reproducible:</strong>
• Save fitted encoders with the model
• Version your preprocessing logic
• Document why each encoding was chosen

<strong>Validate:</strong>
Compare model performance across encoding choices — the best one is data-dependent.`,
                    code: `import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'red', 'green', 'blue'],
    'size':  ['S', 'L', 'M', 'M', 'S', 'L'],
    'price': [10, 20, 15, 12, 18, 22]
})
y = np.array([0, 1, 0, 0, 1, 1])

# Different encoders for different columns, leak-free
pre = ColumnTransformer([
    ('nominal', OneHotEncoder(handle_unknown='ignore'), ['color']),
    ('ordinal', OrdinalEncoder(categories=[['S', 'M', 'L']]), ['size']),
    ('numeric', 'passthrough', ['price'])
])

pipe = Pipeline([
    ('pre', pre),
    ('clf', RandomForestClassifier(random_state=0))
])
pipe.fit(df, y)
print("Pipeline trained with mixed encodings")
print("Prediction:", pipe.predict(df.iloc[[0]]))`
                }
            ]
        },
        {
            number: "Module 6",
            title: "Datasets, Generalization, and Overfitting",
            description: "An introduction to the characteristics of machine learning datasets, and how to prepare your data to ensure high-quality results.",
            duration: "50 min",
            lessons: "9 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Training, Validation, and Test Sets",
                "Understanding Overfitting",
                "Bias-Variance Trade-off",
                "Cross-Validation Techniques",
                "Regularization Methods",
                "Early Stopping",
                "Data Splitting Strategies",
                "Generalization Techniques"
            ],
            detailedDescription: "Learn the critical concepts of overfitting and generalization. Understand how to split your data properly, use cross-validation, and apply regularization techniques to ensure your models perform well on unseen data. This module is essential for building robust ML systems.",
            detailedContent: [
                {
                    title: "Training, Validation, and Test Sets",
                    content: `Splitting data correctly is the foundation of trustworthy machine learning. Each split has a distinct purpose.

<strong>The Three Sets:</strong>
• <strong>Training set:</strong> The model learns its parameters here
• <strong>Validation set:</strong> Tune hyperparameters and compare models
• <strong>Test set:</strong> Final, unbiased performance estimate (use once)

<strong>Typical Proportions:</strong>
• 60% train / 20% validation / 20% test
• Or 80/10/10 for larger datasets

<strong>Golden Rules:</strong>
• Never train on the test set
• Never tune on the test set
• Touch the test set only at the very end

<strong>Why Separate Validation and Test?</strong>
• Tuning on validation "uses up" its objectivity
• The test set stays pristine for a final honest number

<strong>Stratification:</strong>
For classification, keep class proportions equal across splits (stratified splitting).

<strong>Time-Series Caution:</strong>
For temporal data, split by time — never shuffle — to avoid using the future to predict the past.`,
                    code: `import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(1000).reshape(-1, 1)
y = (X.ravel() % 2)

# First split off the test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Then split the rest into train (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("Train:     ", len(X_train))
print("Validation:", len(X_val))
print("Test:      ", len(X_test))
# Test set is held out and never used until the final evaluation`
                },
                {
                    title: "Understanding Overfitting",
                    content: `Overfitting happens when a model memorizes the training data, including its noise, and fails to generalize to new data.

<strong>The Symptoms:</strong>
• Training accuracy very high
• Validation/test accuracy much lower
• Large gap between the two

<strong>Overfitting vs Underfitting:</strong>
• <strong>Underfitting:</strong> Model too simple, poor on both train and test
• <strong>Good fit:</strong> Captures the real pattern, generalizes
• <strong>Overfitting:</strong> Model too complex, great on train, poor on test

<strong>Common Causes:</strong>
• Model too complex for the amount of data
• Too many features relative to samples
• Training too long
• Noisy or unrepresentative data

<strong>How to Detect:</strong>
• Compare train vs validation performance
• Plot learning curves
• Use cross-validation

<strong>How to Reduce:</strong>
• Get more data
• Simplify the model
• Apply regularization
• Use early stopping and dropout`,
                    code: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Underlying pattern is a gentle curve + noise
np.random.seed(0)
X = np.linspace(0, 1, 40).reshape(-1, 1)
y = np.sin(2 * np.pi * X.ravel()) + np.random.normal(0, 0.2, 40)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

# Increasing polynomial degree -> increasing complexity
for degree in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(degree),
                          LinearRegression())
    model.fit(X_tr, y_tr)
    train_err = mean_squared_error(y_tr, model.predict(X_tr))
    test_err = mean_squared_error(y_te, model.predict(X_te))
    print("degree {:>2}: train MSE={:.3f}, test MSE={:.3f}".format(
        degree, train_err, test_err))
# High degree: tiny train error but large test error = overfitting`
                },
                {
                    title: "Bias-Variance Trade-off",
                    content: `The bias-variance trade-off explains the tension between a model that is too simple and one that is too complex.

<strong>Bias:</strong>
• Error from overly simplistic assumptions
• High bias → underfitting
• Model misses the true relationship

<strong>Variance:</strong>
• Error from sensitivity to training data
• High variance → overfitting
• Model changes a lot with different data

<strong>The Trade-off:</strong>
• Simple models: high bias, low variance
• Complex models: low bias, high variance
• Total error = bias² + variance + irreducible noise

<strong>Finding the Sweet Spot:</strong>
• Increase complexity until validation error stops improving
• The minimum of the validation curve is the ideal balance

<strong>Reducing Bias:</strong>
• More complex model, more features, less regularization

<strong>Reducing Variance:</strong>
• More data, simpler model, regularization, ensembling

<strong>Key Insight:</strong>
You cannot eliminate both — you manage the balance for the best generalization.`,
                    code: `import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

np.random.seed(0)
X = np.random.rand(200, 1)
y = (np.sin(6 * X.ravel()) + np.random.normal(0, 0.1, 200))

# Tree depth controls complexity (bias vs variance)
for depth in [1, 3, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=0)
    scores = cross_val_score(model, X, y, cv=5,
                             scoring='neg_mean_squared_error')
    print("max_depth={:>4}: CV MSE={:.4f}".format(
        str(depth), -scores.mean()))
# Shallow = high bias; very deep = high variance.
# The best depth minimizes cross-validated error.`
                },
                {
                    title: "Cross-Validation Techniques",
                    content: `Cross-validation gives a more reliable performance estimate by training and evaluating on multiple data splits.

<strong>K-Fold Cross-Validation:</strong>
• Split data into k equal folds
• Train on k-1 folds, validate on the remaining one
• Repeat k times, average the scores
• Every point is used for both training and validation

<strong>Benefits:</strong>
• More stable estimate than a single split
• Uses all data efficiently
• Reveals variance across folds

<strong>Variants:</strong>
• <strong>Stratified K-Fold:</strong> Preserves class balance (classification)
• <strong>Leave-One-Out:</strong> k = n samples (small datasets)
• <strong>Group K-Fold:</strong> Keep related samples together
• <strong>Time-Series Split:</strong> Respect temporal order

<strong>Choosing k:</strong>
• k=5 or k=10 are common defaults
• Larger k → less bias, more computation

<strong>Important:</strong>
Do all preprocessing inside the CV loop (via a pipeline) so no information leaks between folds.`,
                    code: `import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, weights=[0.7, 0.3],
                           random_state=0)
model = LogisticRegression(max_iter=500)

# Standard vs stratified k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

kf_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
skf_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

print("K-Fold F1:            {:.3f} +/- {:.3f}".format(
    kf_scores.mean(), kf_scores.std()))
print("Stratified K-Fold F1: {:.3f} +/- {:.3f}".format(
    skf_scores.mean(), skf_scores.std()))`
                },
                {
                    title: "Regularization Methods",
                    content: `Regularization discourages overly complex models by adding a penalty for large weights, directly combating overfitting.

<strong>L2 (Ridge):</strong>
• Penalty = λ · Σ(weight²)
• Shrinks weights smoothly toward zero
• Keeps all features but reduces their influence

<strong>L1 (Lasso):</strong>
• Penalty = λ · Σ|weight|
• Drives some weights exactly to zero
• Performs automatic feature selection

<strong>Elastic Net:</strong>
• Combines L1 and L2 penalties
• Balances feature selection and smooth shrinkage

<strong>The Strength Parameter (λ / alpha):</strong>
• Larger λ → stronger penalty → simpler model
• Smaller λ → weaker penalty → fits data more closely
• Tune with cross-validation

<strong>Beyond Linear Models:</strong>
• Neural nets: weight decay, dropout
• Trees: max depth, min samples per leaf
• Boosting: learning rate, number of estimators

<strong>Effect:</strong>
Regularization increases bias slightly to reduce variance a lot — usually improving generalization.`,
                    code: `import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data with many irrelevant features
X, y = make_regression(n_samples=200, n_features=30,
                       n_informative=5, noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

for alpha in [0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=alpha).fit(X_tr, y_tr)
    lasso = Lasso(alpha=alpha).fit(X_tr, y_tr)
    print("alpha={:>4}: Ridge R2={:.3f} | "
          "Lasso R2={:.3f}, features kept={}".format(
        alpha,
        r2_score(y_te, ridge.predict(X_te)),
        r2_score(y_te, lasso.predict(X_te)),
        int(np.sum(lasso.coef_ != 0))))
# Lasso zeroes out irrelevant features automatically`
                },
                {
                    title: "Early Stopping",
                    content: `Early stopping halts training when validation performance stops improving, preventing the model from overfitting to the training set.

<strong>The Idea:</strong>
• Monitor validation error during training
• Training error keeps dropping, but validation error eventually rises
• Stop at the point where validation error is lowest

<strong>How It Works:</strong>
1. Train for one iteration/epoch
2. Evaluate on the validation set
3. If validation improves, save the model
4. If it does not improve for "patience" rounds, stop
5. Restore the best saved model

<strong>Key Parameter — Patience:</strong>
• How many rounds to wait before stopping
• Too small → stop too early (underfit)
• Too large → wasted computation, mild overfit

<strong>Benefits:</strong>
• Automatic complexity control
• Saves training time
• Acts as implicit regularization

<strong>Where It Is Used:</strong>
Neural networks and gradient boosting (XGBoost, LightGBM) support early stopping directly.`,
                    code: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=800, n_features=20,
                           random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

# early_stopping monitors an internal validation split
model = MLPClassifier(hidden_layer_sizes=(64, 32),
                      early_stopping=True,
                      n_iter_no_change=10,   # patience
                      validation_fraction=0.2,
                      max_iter=1000,
                      random_state=0)
model.fit(X_tr, y_tr)

print("Stopped after {} iterations".format(model.n_iter_))
print("Best validation score: {:.3f}".format(
    model.best_validation_score_))
print("Test accuracy: {:.3f}".format(model.score(X_te, y_te)))`
                },
                {
                    title: "Data Splitting Strategies",
                    content: `Choosing the right splitting strategy depends on your data's structure. A wrong split silently inflates or deflates your results.

<strong>Random Split:</strong>
• Default for independent, identically distributed data
• Shuffle then split
• Not valid for time series or grouped data

<strong>Stratified Split:</strong>
• Preserves class proportions
• Essential for imbalanced classification

<strong>Time-Based Split:</strong>
• Train on the past, test on the future
• Never shuffle temporal data
• Reflects real deployment conditions

<strong>Group-Based Split:</strong>
• Keep related records together (same patient, same user)
• Prevents leakage from correlated samples

<strong>Common Pitfalls:</strong>
• Leakage from duplicates across splits
• Preprocessing fit on the whole dataset
• Ignoring temporal or group structure

<strong>Rule:</strong>
The split should mimic how the model will be used in production.`,
                    code: `import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# Time-series split: each fold trains on earlier data only
X = np.arange(12).reshape(-1, 1)
tscv = TimeSeriesSplit(n_splits=3)
print("Time-Series Split:")
for train_idx, test_idx in tscv.split(X):
    print("  train:", train_idx, "test:", test_idx)

# Group split: samples from the same group stay together
groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
Xg = np.arange(8).reshape(-1, 1)
yg = np.array([0, 1, 0, 1, 0, 1, 0, 1])
gkf = GroupKFold(n_splits=2)
print("\\nGroup K-Fold:")
for train_idx, test_idx in gkf.split(Xg, yg, groups):
    print("  train groups:", groups[train_idx],
          "test groups:", groups[test_idx])`
                },
                {
                    title: "Generalization Techniques",
                    content: `Generalization is the ultimate goal: strong performance on data the model has never seen. Here is a consolidated toolkit.

<strong>Get More/Better Data:</strong>
• More examples reduce variance
• Data augmentation creates useful variations
• Clean, representative data beats clever models

<strong>Control Model Complexity:</strong>
• Match model capacity to data size
• Use regularization and early stopping
• Prune trees, limit depth

<strong>Ensemble Methods:</strong>
• <strong>Bagging:</strong> Average many models (Random Forest) → less variance
• <strong>Boosting:</strong> Sequentially fix errors → less bias
• <strong>Stacking:</strong> Combine diverse models

<strong>Validation Discipline:</strong>
• Always evaluate on held-out data
• Use cross-validation for reliability
• Keep a truly untouched test set

<strong>Monitor in Production:</strong>
• Watch for data drift over time
• Retrain as distributions change

<strong>Summary:</strong>
Good generalization comes from the disciplined combination of enough data, appropriate complexity, and honest validation.`,
                    code: `import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=600, n_features=20,
                           random_state=0)

# A single deep tree (high variance) vs an ensemble
single_tree = DecisionTreeClassifier(random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)

for name, model in [("Single Tree", single_tree),
                    ("Random Forest", forest)]:
    scores = cross_val_score(model, X, y, cv=5)
    print("{:>13}: accuracy {:.3f} +/- {:.3f}".format(
        name, scores.mean(), scores.std()))
# The ensemble generalizes better and is more stable`
                }
            ]
        }
    ],
    advancedML: [
        {
            number: "Module 7",
            title: "Neural Networks",
            description: "An introduction to the fundamental principles of neural network architectures, including perceptrons, hidden layers, and activation functions.",
            duration: "60 min",
            lessons: "12 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Perceptrons and Basic Architecture",
                "Hidden Layers and Deep Learning",
                "Activation Functions (ReLU, Sigmoid, Tanh)",
                "Forward Propagation",
                "Backpropagation Explained",
                "Optimization Algorithms",
                "Batch Normalization",
                "Dropout and Regularization",
                "Building Your First Neural Network",
                "Common Architectures",
                "Training Best Practices",
                "Debugging Neural Networks"
            ],
            detailedDescription: "Dive into the world of neural networks! This comprehensive module covers everything from basic perceptrons to deep neural networks. Learn how neurons work, understand activation functions, and master the backpropagation algorithm. Build practical neural networks from scratch.",
            detailedContent: [
                {
                    title: "Perceptrons and Basic Architecture",
                    content: `The perceptron is the fundamental building block of neural networks — a single artificial neuron.

<strong>What a Neuron Does:</strong>
1. Multiply each input by a weight
2. Sum the weighted inputs and add a bias
3. Pass the result through an activation function
4. Output the activated value

<strong>The Math:</strong>
output = activation(w₁x₁ + w₂x₂ + ... + b)

<strong>Biological Inspiration:</strong>
• Loosely models a brain neuron
• Inputs = dendrites, weights = synapse strength
• Activation = whether the neuron "fires"

<strong>From Neuron to Network:</strong>
• A single neuron = a linear classifier
• Stacking neurons into layers → a network
• Layers of neurons can learn complex patterns

<strong>Network Structure:</strong>
• <strong>Input layer:</strong> Receives the features
• <strong>Hidden layers:</strong> Learn intermediate representations
• <strong>Output layer:</strong> Produces the prediction

<strong>The Key Insight:</strong>
Single perceptrons only solve linearly separable problems; multiple layers overcome this limitation.`,
                    code: `import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def activate(self, z):
        return 1 if z > 0 else 0   # step activation

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activate(z)

    def train(self, X, y, lr=0.1, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                self.weights += lr * error * xi
                self.bias += lr * error

# Learn the logical AND function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
p = Perceptron(2)
p.train(X, y)
print("AND predictions:", [p.predict(xi) for xi in X])`
                },
                {
                    title: "Hidden Layers and Deep Learning",
                    content: `Hidden layers between input and output let networks learn hierarchical, non-linear representations. "Deep" learning simply means many hidden layers.

<strong>Why Hidden Layers?</strong>
• A single layer can only draw linear boundaries
• Hidden layers compose features into higher-level concepts
• Enough neurons can approximate almost any function

<strong>Hierarchical Feature Learning:</strong>
• Early layers: simple patterns (edges, basic shapes)
• Middle layers: combinations (textures, parts)
• Deep layers: complex concepts (faces, objects)

<strong>Depth vs Width:</strong>
• <strong>Deeper:</strong> More layers → more abstraction
• <strong>Wider:</strong> More neurons per layer → more capacity
• Deep networks often generalize better than wide ones

<strong>The Universal Approximation Theorem:</strong>
A network with enough neurons can approximate any continuous function — but depth makes this practical and efficient.

<strong>Trade-offs:</strong>
• More layers → more power but harder to train
• Risk of vanishing gradients and overfitting
• Requires more data and compute`,
                    code: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Non-linearly separable data
X, y = make_moons(n_samples=500, noise=0.2, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

# Compare shallow vs deeper architectures
for layers in [(2,), (16,), (32, 16, 8)]:
    model = MLPClassifier(hidden_layer_sizes=layers,
                          max_iter=2000, random_state=0)
    model.fit(X_tr, y_tr)
    print("hidden layers {}: test accuracy {:.3f}".format(
        layers, model.score(X_te, y_te)))
# Deeper networks capture the curved boundary better`
                },
                {
                    title: "Activation Functions (ReLU, Sigmoid, Tanh)",
                    content: `Activation functions introduce non-linearity, which is what allows networks to learn complex patterns. Without them, stacked layers collapse into a single linear function.

<strong>ReLU (Rectified Linear Unit):</strong>
• f(x) = max(0, x)
• Most popular for hidden layers
• Fast, avoids vanishing gradients for positive values
• Risk: "dying ReLU" (neurons stuck at 0)

<strong>Sigmoid:</strong>
• f(x) = 1 / (1 + e^(-x))
• Output in (0, 1) → good for probabilities
• Used in output layer for binary classification
• Suffers from vanishing gradients in deep nets

<strong>Tanh:</strong>
• f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
• Output in (-1, 1), zero-centered
• Often better than sigmoid for hidden layers

<strong>Variants:</strong>
• <strong>Leaky ReLU:</strong> Small slope for negatives (fixes dying ReLU)
• <strong>Softmax:</strong> Multi-class output probabilities

<strong>Choosing:</strong>
• Hidden layers → ReLU (or variants)
• Binary output → sigmoid
• Multi-class output → softmax`,
                    code: `import numpy as np

def relu(x):       return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def sigmoid(x):    return 1 / (1 + np.exp(-x))
def tanh(x):       return np.tanh(x)

x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

print("input:     ", x)
print("relu:      ", np.round(relu(x), 3))
print("leaky_relu:", np.round(leaky_relu(x), 3))
print("sigmoid:   ", np.round(sigmoid(x), 3))
print("tanh:      ", np.round(tanh(x), 3))

# Derivatives matter for backprop
def relu_deriv(x): return (x > 0).astype(float)
print("\\nrelu gradient:", relu_deriv(x))`
                },
                {
                    title: "Forward Propagation",
                    content: `Forward propagation is the process of passing input data through the network to produce a prediction.

<strong>Step by Step:</strong>
1. Start with the input features
2. For each layer: compute z = W·a + b
3. Apply the activation: a = activation(z)
4. Feed the output to the next layer
5. The final layer produces the prediction

<strong>Matrix Form:</strong>
• Weights stored as matrices for efficiency
• One matrix multiply processes a whole layer
• Batches of inputs processed simultaneously

<strong>Layer-by-Layer Transformation:</strong>
• Each layer transforms the representation
• Data flows in one direction: input → output
• Intermediate activations are cached for backprop

<strong>Why "Forward"?</strong>
Information moves forward through the network. The reverse pass (backpropagation) uses these cached values to compute gradients.

<strong>Output Interpretation:</strong>
• Regression: raw value
• Binary classification: sigmoid probability
• Multi-class: softmax probabilities`,
                    code: `import numpy as np

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

# A 2-layer network: 3 inputs -> 4 hidden -> 1 output
np.random.seed(0)
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros(4)
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

def forward(x):
    z1 = x @ W1 + b1        # hidden pre-activation
    a1 = relu(z1)           # hidden activation
    z2 = a1 @ W2 + b2       # output pre-activation
    a2 = sigmoid(z2)        # final probability
    return a2, (z1, a1, z2)  # cache for backprop

# Process a batch of 2 samples at once
X = np.array([[0.5, 0.2, 0.1],
              [0.9, 0.7, 0.3]])
output, cache = forward(X)
print("Predictions:\\n", np.round(output, 4))`
                },
                {
                    title: "Backpropagation Explained",
                    content: `Backpropagation is the algorithm that computes how much each weight contributed to the error, enabling the network to learn.

<strong>The Core Idea:</strong>
• Compute the loss at the output
• Work backwards, layer by layer
• Use the chain rule to find each weight's gradient
• Update weights in the direction that reduces loss

<strong>The Chain Rule:</strong>
Gradients flow backward by multiplying local derivatives:
∂Loss/∂w = ∂Loss/∂output × ∂output/∂z × ∂z/∂w

<strong>Two Passes:</strong>
1. <strong>Forward pass:</strong> Compute prediction and loss
2. <strong>Backward pass:</strong> Compute gradients for all weights

<strong>Then Update:</strong>
w = w - learning_rate × gradient

<strong>Why It Is Efficient:</strong>
• Reuses cached forward-pass values
• Computes all gradients in one backward sweep
• Scales to millions of parameters

<strong>Intuition:</strong>
Backprop assigns "blame" for the error to each weight and nudges it to do better next time.`,
                    code: `import numpy as np

# Tiny network: 2 inputs -> 2 hidden -> 1 output, sigmoid
np.random.seed(1)
def sigmoid(x):  return 1 / (1 + np.exp(-x))
def dsigmoid(a): return a * (1 - a)

X = np.array([[0.5, 0.1]])
y = np.array([[1.0]])
W1 = np.random.randn(2, 2) * 0.5
W2 = np.random.randn(2, 1) * 0.5

for step in range(1000):
    # Forward
    a1 = sigmoid(X @ W1)
    a2 = sigmoid(a1 @ W2)
    # Backward (chain rule)
    d2 = (a2 - y) * dsigmoid(a2)
    d1 = (d2 @ W2.T) * dsigmoid(a1)
    # Update
    W2 -= 0.5 * a1.T @ d2
    W1 -= 0.5 * X.T @ d1

print("Target:", y.ravel(), " Prediction:", np.round(a2.ravel(), 4))`
                },
                {
                    title: "Optimization Algorithms",
                    content: `Optimizers determine how weights are updated using the gradients from backpropagation. The right optimizer speeds up and stabilizes training.

<strong>Gradient Descent Variants:</strong>
• <strong>Batch GD:</strong> Uses all data per step (stable, slow)
• <strong>Stochastic GD:</strong> One sample per step (fast, noisy)
• <strong>Mini-batch GD:</strong> Small batches (the practical standard)

<strong>Momentum:</strong>
• Accumulates a velocity from past gradients
• Accelerates in consistent directions
• Dampens oscillations

<strong>Adaptive Optimizers:</strong>
• <strong>AdaGrad:</strong> Per-parameter learning rates
• <strong>RMSprop:</strong> Moving average of squared gradients
• <strong>Adam:</strong> Combines momentum + RMSprop (most popular default)

<strong>Adam in Practice:</strong>
• Works well out of the box
• Good default learning rate ~0.001
• Adapts per-parameter, converges quickly

<strong>Learning Rate Schedules:</strong>
• Decay the learning rate over time
• Warmup then decay is common in deep learning

<strong>Recommendation:</strong>
Start with Adam; switch to SGD+momentum for final fine-tuning if needed.`,
                    code: `import numpy as np

# Compare plain SGD vs SGD with momentum on a simple loss
# Minimize f(w) = (w - 3)^2, gradient = 2(w - 3)
def grad(w): return 2 * (w - 3)

# Plain SGD
w = 0.0
for _ in range(50):
    w -= 0.1 * grad(w)
print("SGD result:      w = {:.4f}".format(w))

# SGD with momentum
w, v = 0.0, 0.0
for _ in range(50):
    v = 0.9 * v - 0.1 * grad(w)
    w += v
print("Momentum result: w = {:.4f}".format(w))

# Adam-style update
w, m, vv, t = 0.0, 0.0, 0.0, 0
for _ in range(50):
    t += 1
    g = grad(w)
    m = 0.9 * m + 0.1 * g
    vv = 0.999 * vv + 0.001 * g**2
    m_hat = m / (1 - 0.9**t)
    v_hat = vv / (1 - 0.999**t)
    w -= 0.5 * m_hat / (np.sqrt(v_hat) + 1e-8)
print("Adam result:     w = {:.4f}".format(w))`
                },
                {
                    title: "Batch Normalization",
                    content: `Batch normalization stabilizes and accelerates training by normalizing the inputs to each layer.

<strong>The Problem It Solves:</strong>
• As training progresses, layer input distributions shift ("internal covariate shift")
• This slows convergence and destabilizes deep networks

<strong>How It Works:</strong>
• For each mini-batch, normalize activations to mean 0, variance 1
• Then scale and shift with learnable parameters (γ, β)
• Applied between the linear step and activation

<strong>Benefits:</strong>
• Faster training (higher learning rates possible)
• Reduces sensitivity to weight initialization
• Acts as mild regularization
• Smooths the loss landscape

<strong>Training vs Inference:</strong>
• Training: use the current batch's statistics
• Inference: use running averages collected during training

<strong>Related Techniques:</strong>
• <strong>Layer Normalization:</strong> Normalizes across features (used in Transformers)
• <strong>Group Normalization:</strong> For small batch sizes

<strong>Placement:</strong>
Typically applied after the dense/conv layer and before (or after) the activation.`,
                    code: `import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta   # scale and shift

# A batch of activations (4 samples, 3 features)
x = np.array([[10.0, 2.0, 30.0],
              [12.0, 1.0, 28.0],
              [ 8.0, 3.0, 35.0],
              [11.0, 2.5, 25.0]])

gamma = np.ones(3)    # learnable scale
beta = np.zeros(3)    # learnable shift

out = batch_norm(x, gamma, beta)
print("Normalized output:\\n", np.round(out, 3))
print("\\nColumn means ~0:", np.round(out.mean(axis=0), 3))
print("Column stds  ~1:", np.round(out.std(axis=0), 3))`
                },
                {
                    title: "Dropout and Regularization",
                    content: `Dropout is a simple, powerful regularization technique that reduces overfitting in neural networks.

<strong>How Dropout Works:</strong>
• During training, randomly "drop" (zero out) a fraction of neurons
• Each forward pass uses a different random subset
• Forces the network not to rely on any single neuron

<strong>Why It Helps:</strong>
• Prevents co-adaptation of neurons
• Acts like training an ensemble of sub-networks
• Improves generalization

<strong>The Dropout Rate:</strong>
• Typical values: 0.2 to 0.5
• Higher rate → stronger regularization
• Too high → underfitting

<strong>Training vs Inference:</strong>
• Training: randomly drop neurons
• Inference: use all neurons, scale outputs accordingly

<strong>Other NN Regularization:</strong>
• <strong>Weight decay (L2):</strong> Penalize large weights
• <strong>Early stopping:</strong> Stop when validation worsens
• <strong>Data augmentation:</strong> Expand training variety
• <strong>Batch norm:</strong> Provides mild regularization

<strong>Combine Wisely:</strong>
Dropout + weight decay + early stopping is a robust, common recipe.`,
                    code: `import numpy as np

def dropout(x, rate, training=True):
    if not training or rate == 0:
        return x
    # Keep each neuron with probability (1 - rate)
    mask = (np.random.rand(*x.shape) > rate) / (1 - rate)
    return x * mask   # inverted dropout scales during training

np.random.seed(0)
activations = np.ones((1, 10))

train_out = dropout(activations, rate=0.4, training=True)
test_out = dropout(activations, rate=0.4, training=False)

print("Training (some dropped, rest scaled):")
print(np.round(train_out, 2))
print("\\nInference (all kept):")
print(test_out)`
                },
                {
                    title: "Building Your First Neural Network",
                    content: `Let's assemble the concepts into a complete, trainable neural network using a high-level library.

<strong>The Standard Workflow:</strong>
1. Prepare and scale the data
2. Define the architecture (layers, activations)
3. Choose a loss function and optimizer
4. Train over epochs with mini-batches
5. Evaluate on held-out data

<strong>Architecture Decisions:</strong>
• Input size = number of features
• Hidden layers = capacity (start small)
• Output size = number of classes/targets
• Activations = ReLU hidden, sigmoid/softmax output

<strong>Key Hyperparameters:</strong>
• Learning rate (most important)
• Batch size (32-256 common)
• Number of epochs (use early stopping)
• Network width and depth

<strong>Practical Tips:</strong>
• Always scale inputs
• Start simple, add complexity gradually
• Monitor training and validation loss
• Use dropout/regularization if overfitting

<strong>Frameworks:</strong>
scikit-learn (MLP), Keras/TensorFlow, and PyTorch are the common choices.`,
                    code: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Real dataset: breast cancer diagnosis
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42)

# Scale + neural network in one pipeline
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16),
                  activation='relu',
                  solver='adam',
                  alpha=1e-4,          # L2 regularization
                  early_stopping=True,
                  max_iter=500,
                  random_state=42)
)
model.fit(X_tr, y_tr)

print("Test accuracy: {:.3f}".format(model.score(X_te, y_te)))
sample = data.data[[0]]
print("Predicted class:", model.predict(sample)[0],
      "(", data.target_names[model.predict(sample)[0]], ")")`
                },
                {
                    title: "Common Architectures",
                    content: `Different problems call for different network architectures. Knowing the main families helps you pick the right tool.

<strong>Feedforward (Dense) Networks:</strong>
• Fully connected layers
• Good for tabular data
• The general-purpose baseline

<strong>Convolutional Neural Networks (CNNs):</strong>
• Use convolution filters to detect local patterns
• Excellent for images and spatial data
• Parameter-efficient via weight sharing

<strong>Recurrent Neural Networks (RNNs):</strong>
• Process sequences step by step, keeping a memory
• LSTM and GRU handle long dependencies
• Used for time series and text (historically)

<strong>Transformers:</strong>
• Use attention to relate all positions at once
• Dominate NLP and increasingly vision
• Power modern LLMs

<strong>Autoencoders:</strong>
• Learn compressed representations
• Used for denoising, anomaly detection

<strong>Choosing:</strong>
• Tabular → dense networks
• Images → CNNs
• Sequences/text → Transformers (or RNNs)
• Compression/anomaly → autoencoders`,
                    code: `# Conceptual Keras-style sketches (illustrative)

# 1. Dense network for tabular data
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(n_features,)),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# 2. CNN for images
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     MaxPooling2D((2,2)),
#     Conv2D(64, (3,3), activation='relu'),
#     Flatten(),
#     Dense(10, activation='softmax')
# ])

# 3. LSTM for sequences
# model = Sequential([
#     Embedding(vocab_size, 128),
#     LSTM(64),
#     Dense(1, activation='sigmoid')
# ])

print("Tabular -> Dense | Images -> CNN | Text -> Transformer/RNN")`
                },
                {
                    title: "Training Best Practices",
                    content: `Training neural networks reliably requires attention to several practical details.

<strong>Data Preparation:</strong>
• Always normalize/standardize inputs
• Shuffle training data each epoch
• Use a proper train/validation/test split

<strong>Weight Initialization:</strong>
• Use He initialization for ReLU
• Use Xavier/Glorot for tanh/sigmoid
• Poor initialization → slow or failed training

<strong>Learning Rate:</strong>
• The single most important hyperparameter
• Too high → diverges; too low → crawls
• Use schedules or adaptive optimizers (Adam)

<strong>Batch Size:</strong>
• Smaller batches → noisier but often generalize well
• Larger batches → faster but need tuning

<strong>Monitor Training:</strong>
• Plot training and validation loss
• Watch for overfitting (diverging curves)
• Use early stopping

<strong>Regularization:</strong>
• Combine dropout, weight decay, early stopping
• Add data augmentation where possible

<strong>Reproducibility:</strong>
Set random seeds and log hyperparameters.`,
                    code: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, n_features=20,
                           random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

# Best practices: scale, adaptive LR, early stopping, regularization
scaler = StandardScaler().fit(X_tr)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    learning_rate_init=0.001,     # sensible starting LR
    alpha=1e-4,                   # weight decay
    batch_size=64,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=0
)
model.fit(scaler.transform(X_tr), y_tr)

print("Iterations run:", model.n_iter_)
print("Test accuracy: {:.3f}".format(
    model.score(scaler.transform(X_te), y_te)))`
                },
                {
                    title: "Debugging Neural Networks",
                    content: `Neural networks fail in subtle ways. A systematic debugging approach saves hours of frustration.

<strong>Loss Not Decreasing:</strong>
• Learning rate too high or too low
• Data not normalized
• Bug in the loss or labels
• Try overfitting a tiny sample first (should reach ~0 loss)

<strong>Loss Is NaN:</strong>
• Learning rate too high (exploding gradients)
• Division by zero / log(0)
• Use gradient clipping and check for bad inputs

<strong>Overfitting:</strong>
• Add dropout, weight decay, or more data
• Reduce model size
• Use early stopping

<strong>Underfitting:</strong>
• Increase capacity (layers/neurons)
• Train longer, reduce regularization
• Improve features

<strong>Vanishing/Exploding Gradients:</strong>
• Use ReLU, batch norm, residual connections
• Better initialization
• Gradient clipping for explosions

<strong>Debugging Checklist:</strong>
1. Can it overfit a tiny dataset?
2. Are inputs/labels correct and scaled?
3. Is the learning rate reasonable?
4. Are gradients flowing (not zero/NaN)?

<strong>Golden Rule:</strong>
Start simple, verify each piece, then scale up.`,
                    code: `import numpy as np
from sklearn.neural_network import MLPClassifier

# Debugging technique: can the model overfit a tiny sample?
# If it CAN'T reach ~100% here, something is broken.
np.random.seed(0)
X_tiny = np.random.randn(20, 10)
y_tiny = np.random.randint(0, 2, 20)

model = MLPClassifier(hidden_layer_sizes=(64, 64),
                      max_iter=2000, random_state=0)
model.fit(X_tiny, y_tiny)
train_acc = model.score(X_tiny, y_tiny)

print("Overfit-a-tiny-sample accuracy: {:.2f}".format(train_acc))
if train_acc > 0.95:
    print("Good: model + training loop can learn.")
else:
    print("Warning: check data, LR, or architecture.")`
                }
            ]
        },
        {
            number: "Module 8",
            title: "Embeddings",
            description: "Learn how embeddings allow you to do machine learning on large feature vectors and capture semantic relationships.",
            duration: "45 min",
            lessons: "8 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Introduction to Embeddings",
                "Word Embeddings (Word2Vec)",
                "Vector Space Models",
                "Similarity and Distance Metrics",
                "Embedding Visualization",
                "Transfer Learning with Embeddings",
                "Practical Applications",
                "Building Custom Embeddings"
            ],
            detailedDescription: "Embeddings are powerful representations that capture semantic meaning in a dense vector space. Learn how to work with word embeddings, create your own embeddings, and leverage pre-trained embeddings for transfer learning. Essential for NLP and recommendation systems.",
            detailedContent: [
                {
                    title: "Introduction to Embeddings",
                    content: `Embeddings are dense, low-dimensional vector representations that capture the meaning and relationships of discrete items.

<strong>The Problem With One-Hot:</strong>
• One-hot vectors are huge and sparse
• Every item is equally distant from every other
• No notion of similarity (cat vs dog vs car all equally different)

<strong>What Embeddings Do:</strong>
• Map each item to a dense vector (e.g., 100 numbers)
• Similar items get similar vectors
• Meaning is captured by position in the space

<strong>Key Properties:</strong>
• <strong>Dense:</strong> Few dimensions, all informative
• <strong>Learned:</strong> Trained from data, not hand-crafted
• <strong>Semantic:</strong> Distance reflects similarity

<strong>Where They Are Used:</strong>
• Words (NLP)
• Users and items (recommendations)
• Categories (high-cardinality features)
• Images, graphs, and more

<strong>The Big Idea:</strong>
Embeddings turn discrete symbols into continuous vectors that machines can reason about geometrically.`,
                    code: `import numpy as np

# One-hot: sparse, no similarity information
vocab = ['cat', 'dog', 'car', 'truck']
one_hot = np.eye(len(vocab))
print("One-hot (sparse, 4 dims):")
print(one_hot)

# Embedding: dense, encodes similarity
# (illustrative hand-set vectors)
embeddings = {
    'cat':   [0.9, 0.1],
    'dog':   [0.8, 0.2],   # close to cat (both animals)
    'car':   [0.1, 0.9],
    'truck': [0.2, 0.8],   # close to car (both vehicles)
}
print("\\nEmbeddings (dense, 2 dims):")
for word, vec in embeddings.items():
    print("  {:>5}: {}".format(word, vec))`
                },
                {
                    title: "Word Embeddings (Word2Vec)",
                    content: `Word2Vec is a landmark technique that learns word embeddings from large text corpora based on the words' contexts.

<strong>The Core Idea:</strong>
"You shall know a word by the company it keeps." Words appearing in similar contexts get similar vectors.

<strong>Two Architectures:</strong>
• <strong>CBOW:</strong> Predict a word from its surrounding context
• <strong>Skip-gram:</strong> Predict the context from a word
• Skip-gram works better for rare words

<strong>How Training Works:</strong>
• Slide a window over text
• Learn to predict neighbors
• Words with shared neighbors converge in vector space

<strong>Famous Analogies:</strong>
• king - man + woman ≈ queen
• Paris - France + Italy ≈ Rome
• Vector arithmetic captures relationships!

<strong>Other Word Embeddings:</strong>
• <strong>GloVe:</strong> Uses global co-occurrence statistics
• <strong>FastText:</strong> Uses subword information (handles unknown words)

<strong>Impact:</strong>
Word2Vec showed that meaning could be learned unsupervised from raw text, launching modern NLP.`,
                    code: `import numpy as np

# Illustrative pre-trained-style vectors
vecs = {
    'king':  np.array([0.8, 0.7, 0.2]),
    'man':   np.array([0.7, 0.1, 0.2]),
    'woman': np.array([0.7, 0.1, 0.9]),
    'queen': np.array([0.8, 0.7, 0.9]),
}

# Famous analogy: king - man + woman  ~=  queen
result = vecs['king'] - vecs['man'] + vecs['woman']
print("king - man + woman =", np.round(result, 2))
print("queen              =", vecs['queen'])

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print("Similarity to queen: {:.3f}".format(cosine(result, vecs['queen'])))

# In practice: from gensim.models import Word2Vec
# model = Word2Vec(sentences, vector_size=100, window=5)`
                },
                {
                    title: "Vector Space Models",
                    content: `A vector space model represents items as points in a continuous multi-dimensional space where geometry encodes meaning.

<strong>The Space:</strong>
• Each dimension captures some latent feature
• Items are points/vectors in this space
• Relationships become geometric operations

<strong>What Dimensions Represent:</strong>
• Learned automatically, not labeled
• A dimension might loosely encode "animal-ness" or "formality"
• Usually not individually interpretable

<strong>Operations in the Space:</strong>
• <strong>Distance:</strong> How different two items are
• <strong>Direction:</strong> Relationships (gender, tense, plurality)
• <strong>Clusters:</strong> Groups of related items

<strong>Dimensionality:</strong>
• Too few dims → can't capture nuance
• Too many dims → sparse, harder to train
• Typical: 50-300 for words

<strong>Why It Works:</strong>
Geometry gives us math tools: similarity, arithmetic, and clustering all become computable on meaning.

<strong>Beyond Words:</strong>
The same idea powers recommendation systems, search, and retrieval.`,
                    code: `import numpy as np

# A small vector space of foods
space = {
    'apple':  np.array([0.9, 0.1, 0.8]),
    'banana': np.array([0.85, 0.15, 0.75]),
    'pizza':  np.array([0.2, 0.9, 0.3]),
    'burger': np.array([0.15, 0.95, 0.25]),
}

def euclidean(a, b):
    return np.linalg.norm(a - b)

# Distances reveal structure: fruits close, fast foods close
items = list(space.keys())
print("Pairwise distances:")
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        d = euclidean(space[items[i]], space[items[j]])
        print("  {:>6} <-> {:<6}: {:.3f}".format(
            items[i], items[j], d))`
                },
                {
                    title: "Similarity and Distance Metrics",
                    content: `Measuring similarity between embeddings is central to search, recommendations, and clustering.

<strong>Cosine Similarity:</strong>
• Measures the angle between vectors
• Range: -1 (opposite) to 1 (identical direction)
• Ignores magnitude, focuses on direction
• The most common choice for embeddings

<strong>Euclidean Distance:</strong>
• Straight-line distance between points
• Sensitive to vector magnitude
• Range: 0 (identical) to ∞

<strong>Dot Product:</strong>
• Combines direction and magnitude
• Fast to compute
• Used inside neural networks and attention

<strong>Manhattan Distance:</strong>
• Sum of absolute differences
• Less sensitive to outliers

<strong>Choosing a Metric:</strong>
• Text/semantic similarity → cosine
• Magnitude matters → Euclidean or dot product
• Normalized vectors → cosine and dot product agree

<strong>Practical Use:</strong>
"Find the k nearest neighbors" powers semantic search, recommendations, and retrieval-augmented systems.`,
                    code: `import numpy as np

def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

query = np.array([0.5, 0.8, 0.3])
database = {
    'doc1': np.array([0.5, 0.7, 0.4]),
    'doc2': np.array([0.1, 0.2, 0.9]),
    'doc3': np.array([0.6, 0.75, 0.35]),
}

# Rank documents by cosine similarity (semantic search)
scores = {name: cosine_similarity(query, vec)
          for name, vec in database.items()}
ranked = sorted(scores.items(), key=lambda x: -x[1])

print("Most similar to query:")
for name, score in ranked:
    print("  {}: cosine={:.3f}".format(name, score))`
                },
                {
                    title: "Embedding Visualization",
                    content: `Embeddings live in high-dimensional space, so we use dimensionality reduction to inspect them visually.

<strong>Why Visualize?</strong>
• Verify that similar items cluster together
• Discover structure and relationships
• Debug and build intuition

<strong>t-SNE:</strong>
• Preserves local neighborhoods
• Great for revealing clusters
• Non-deterministic; tune perplexity
• Distances between clusters are not meaningful

<strong>UMAP:</strong>
• Faster than t-SNE
• Preserves both local and some global structure
• Increasingly the default choice

<strong>PCA:</strong>
• Linear, fast, deterministic
• Preserves global variance
• Good first look, less good at clusters

<strong>Reading the Plots:</strong>
• Tight clusters → strongly related items
• Outliers → unusual items
• Smooth gradients → continuous relationships

<strong>Caution:</strong>
2D projections distort the real space — use them for intuition, not precise measurement.`,
                    code: `import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)
# 20 items in 50-dimensional embedding space, two hidden groups
group_a = np.random.randn(10, 50) + 2
group_b = np.random.randn(10, 50) - 2
embeddings = np.vstack([group_a, group_b])

# Reduce 50D -> 2D for plotting
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

print("Reduced shape:", coords.shape)
print("Variance explained:",
      np.round(pca.explained_variance_ratio_, 3))
print("\\nGroup A center:", np.round(coords[:10].mean(axis=0), 2))
print("Group B center:", np.round(coords[10:].mean(axis=0), 2))
# Plot with: plt.scatter(coords[:,0], coords[:,1])
# For clusters, prefer: from sklearn.manifold import TSNE`
                },
                {
                    title: "Transfer Learning with Embeddings",
                    content: `Transfer learning reuses embeddings learned on large datasets to boost performance on your own smaller tasks.

<strong>The Idea:</strong>
• Pre-trained embeddings already encode general knowledge
• Reuse them instead of learning from scratch
• Especially valuable with limited data

<strong>Pre-trained Options:</strong>
• <strong>Words:</strong> Word2Vec, GloVe, FastText
• <strong>Sentences:</strong> Sentence-BERT, Universal Sentence Encoder
• <strong>Contextual:</strong> BERT, GPT embeddings

<strong>How to Use Them:</strong>
1. <strong>Frozen:</strong> Use embeddings as fixed features
2. <strong>Fine-tuned:</strong> Start from pre-trained, adapt to your task
3. <strong>Feature extraction:</strong> Feed embeddings into a simpler model

<strong>Benefits:</strong>
• Faster training and convergence
• Better performance with little data
• Encodes knowledge from massive corpora

<strong>When to Fine-tune:</strong>
• Enough task-specific data → fine-tune
• Very little data → keep embeddings frozen
• Domain very different → fine-tune more layers

<strong>Impact:</strong>
Transfer learning is why small teams can build strong NLP systems today.`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulated pre-trained sentence embeddings (e.g., from BERT)
# In practice: from sentence_transformers import SentenceTransformer
#              model = SentenceTransformer('all-MiniLM-L6-v2')
#              X = model.encode(sentences)

np.random.seed(0)
# Positive reviews cluster one way, negative another
pos = np.random.randn(30, 16) + 1
neg = np.random.randn(30, 16) - 1
X = np.vstack([pos, neg])
y = np.array([1]*30 + [0]*30)

# Train a simple classifier ON TOP of frozen embeddings
clf = LogisticRegression().fit(X, y)
print("Accuracy on embeddings: {:.3f}".format(clf.score(X, y)))

# Classify a new "review embedding"
new_review = np.random.randn(1, 16) + 1
print("Predicted sentiment:",
      "Positive" if clf.predict(new_review)[0] else "Negative")`
                },
                {
                    title: "Practical Applications",
                    content: `Embeddings power many production systems you use every day.

<strong>Semantic Search:</strong>
• Embed queries and documents
• Retrieve by vector similarity, not just keywords
• Understands meaning, handles synonyms

<strong>Recommendation Systems:</strong>
• Embed users and items
• Recommend items whose vectors are close to the user's
• Netflix, Spotify, Amazon all use this

<strong>Natural Language Processing:</strong>
• Text classification, sentiment analysis
• Named entity recognition, translation
• Foundation for LLMs

<strong>Retrieval-Augmented Generation (RAG):</strong>
• Store document embeddings in a vector database
• Retrieve relevant chunks to ground LLM answers

<strong>Other Uses:</strong>
• Image search (embed images)
• Fraud/anomaly detection (unusual vectors)
• Duplicate detection and clustering

<strong>Vector Databases:</strong>
Tools like FAISS, Pinecone, and Milvus enable fast nearest-neighbor search over millions of embeddings.`,
                    code: `import numpy as np

# Mini recommendation engine using embeddings
item_embeddings = {
    'Action Movie A':  np.array([0.9, 0.1, 0.2]),
    'Action Movie B':  np.array([0.85, 0.15, 0.25]),
    'Romance Movie':   np.array([0.1, 0.9, 0.3]),
    'Documentary':     np.array([0.2, 0.3, 0.9]),
}

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

# User just watched an action movie -> build a taste vector
user_vector = item_embeddings['Action Movie A']

# Recommend the most similar unseen items
recs = [(name, cosine(user_vector, vec))
        for name, vec in item_embeddings.items()
        if name != 'Action Movie A']
recs.sort(key=lambda x: -x[1])

print("Recommended for you:")
for name, score in recs:
    print("  {}: match={:.2f}".format(name, score))`
                },
                {
                    title: "Building Custom Embeddings",
                    content: `Sometimes pre-trained embeddings are not enough — you can train embeddings tailored to your own data.

<strong>When to Build Custom:</strong>
• Specialized domain (medical, legal, gaming)
• Non-text items (products, users, songs)
• Vocabulary not covered by pre-trained models

<strong>Embedding Layers in Neural Networks:</strong>
• A trainable lookup table: index → vector
• Learned jointly with the main task
• Optimized end-to-end via backpropagation

<strong>Training Approaches:</strong>
• <strong>Task-supervised:</strong> Learn embeddings while training a classifier
• <strong>Self-supervised:</strong> Predict context (Word2Vec-style)
• <strong>Contrastive:</strong> Pull similar items together, push others apart

<strong>Design Choices:</strong>
• Embedding dimension (start with 50-300)
• Vocabulary size and rare-token handling
• Regularization to prevent overfitting

<strong>Evaluation:</strong>
• Check nearest neighbors make sense
• Measure downstream task performance
• Visualize clusters

<strong>Key Point:</strong>
An embedding layer is just weights the network learns — meaning emerges from the training objective.`,
                    code: `import numpy as np

# A trainable embedding layer = a lookup table
class EmbeddingLayer:
    def __init__(self, vocab_size, dim):
        # Each row is one item's embedding vector
        self.table = np.random.randn(vocab_size, dim) * 0.1

    def lookup(self, indices):
        return self.table[indices]

    def update(self, indices, grad, lr=0.01):
        # Backprop updates only the used rows
        self.table[indices] -= lr * grad

# Vocabulary of 5 items, 4-dimensional embeddings
emb = EmbeddingLayer(vocab_size=5, dim=4)

# Look up embeddings for items 0 and 3
vectors = emb.lookup([0, 3])
print("Embeddings for items [0, 3]:")
print(np.round(vectors, 3))

# During training, gradients flow back and adjust these rows
# In Keras: Embedding(input_dim=vocab, output_dim=dim)
# In PyTorch: nn.Embedding(vocab, dim)`
                }
            ]
        },
        {
            number: "Module 9",
            title: "Large Language Models",
            description: "An introduction to large language models, from tokens to Transformers. Learn the basics of how LLMs learn to predict text output.",
            duration: "70 min",
            lessons: "14 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Introduction to LLMs",
                "Tokenization Fundamentals",
                "Attention Mechanism",
                "Transformer Architecture",
                "Self-Attention Explained",
                "Multi-Head Attention",
                "Positional Encoding",
                "BERT and GPT Models",
                "Fine-tuning LLMs",
                "Prompt Engineering",
                "Transfer Learning",
                "Ethical Considerations",
                "Practical Applications",
                "Future of LLMs"
            ],
            detailedDescription: "Explore the cutting-edge world of Large Language Models! Understand how transformers revolutionized NLP, learn about attention mechanisms, and discover how models like GPT and BERT work. This new module covers the latest advances in AI and practical applications of LLMs.",
            detailedContent: [
                {
                    title: "Introduction to LLMs",
                    content: `Large Language Models (LLMs) are neural networks trained on massive amounts of text to understand and generate human-like language.

<strong>What Makes Them "Large"?</strong>
• Billions (or trillions) of parameters
• Trained on enormous text corpora (books, web, code)
• Require significant compute to train

<strong>Core Capability:</strong>
• Predict the next token given previous tokens
• This simple objective, at scale, produces remarkable abilities

<strong>Emergent Abilities:</strong>
• Translation, summarization, reasoning
• Code generation, question answering
• Few-shot learning (learn from examples in the prompt)

<strong>Famous Examples:</strong>
• GPT family (OpenAI)
• Gemini (Google)
• Claude (Anthropic)
• LLaMA (Meta)

<strong>The Foundation:</strong>
Almost all modern LLMs are based on the <strong>Transformer</strong> architecture and its attention mechanism.

<strong>Why They Matter:</strong>
LLMs power chatbots, coding assistants, search, and content generation — reshaping how we work with information.`,
                    code: `# Conceptual: an LLM predicts the next token
# In practice: from transformers import pipeline
#              generator = pipeline('text-generation', model='gpt2')

# The fundamental loop of text generation
def generate(prompt, model, max_tokens=20):
    tokens = list(prompt)
    for _ in range(max_tokens):
        # Model outputs a probability for every possible next token
        next_token = model.predict_next(tokens)  # pseudo-code
        tokens.append(next_token)
        if next_token == '<END>':
            break
    return tokens

# Key insight: "understanding" emerges from
# next-token prediction at massive scale.
print("LLM = next-token predictor trained on huge text data")`
                },
                {
                    title: "Tokenization Fundamentals",
                    content: `Before an LLM can process text, it must break it into tokens — the basic units the model reads.

<strong>What Is a Token?</strong>
• A chunk of text: a word, subword, or character
• "unhappiness" might become ["un", "happiness"]
• Roughly 1 token ≈ 4 characters ≈ 0.75 words in English

<strong>Why Subwords?</strong>
• Full words → vocabulary too large, misses rare words
• Characters → sequences too long
• Subwords → best balance, handle unknown words

<strong>Common Algorithms:</strong>
• <strong>BPE (Byte-Pair Encoding):</strong> Merges frequent pairs (GPT)
• <strong>WordPiece:</strong> Similar, used by BERT
• <strong>SentencePiece:</strong> Language-agnostic

<strong>The Vocabulary:</strong>
• A fixed set of tokens (e.g., 50,000)
• Each token maps to an integer ID
• IDs are then converted to embeddings

<strong>Why It Matters:</strong>
• Token count drives cost and context limits
• Affects how the model "sees" your text
• Different languages tokenize differently`,
                    code: `# Simple word-level tokenization for intuition
text = "Machine learning is powerful"
tokens = text.lower().split()
print("Tokens:", tokens)

# Build a vocabulary (token -> id)
vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
print("Vocabulary:", vocab)

# Encode text to token IDs
ids = [vocab[t] for t in tokens]
print("Token IDs:", ids)

# Real tokenizers use subwords:
# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained('gpt2')
# tok.encode("unhappiness")  -> subword IDs
print("\\nSubword example: 'unhappiness' -> ['un', 'happiness']")`
                },
                {
                    title: "Attention Mechanism",
                    content: `Attention is the breakthrough that lets models focus on the most relevant parts of the input when processing each token.

<strong>The Problem It Solves:</strong>
• Older models (RNNs) struggled with long-range dependencies
• Information from early tokens faded over long sequences
• Attention lets any token directly access any other

<strong>The Intuition:</strong>
When processing a word, attention asks: "Which other words should I focus on to understand this one?"

<strong>Query, Key, Value:</strong>
• <strong>Query (Q):</strong> What the current token is looking for
• <strong>Key (K):</strong> What each token offers
• <strong>Value (V):</strong> The actual information to retrieve
• Match queries to keys → weights → weighted sum of values

<strong>Attention Scores:</strong>
• Compute similarity between query and each key
• Softmax turns scores into weights (sum to 1)
• Higher weight → more focus on that token

<strong>Why It Is Powerful:</strong>
• Handles long-range dependencies effortlessly
• Fully parallelizable (unlike RNNs)
• Learns what to focus on, per context`,
                    code: `import numpy as np

def attention(Q, K, V):
    d_k = Q.shape[-1]
    # Similarity between queries and keys
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax -> attention weights
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    # Weighted sum of values
    return weights @ V, weights

# 3 tokens, each represented by a 4-dim vector
np.random.seed(0)
X = np.random.randn(3, 4)
Q = K = V = X   # self-attention: same source

output, weights = attention(Q, K, V)
print("Attention weights (who focuses on whom):")
print(np.round(weights, 2))
print("\\nEach row sums to 1:", np.round(weights.sum(axis=1), 2))`
                },
                {
                    title: "Transformer Architecture",
                    content: `The Transformer is the architecture behind all modern LLMs. Introduced in 2017 ("Attention Is All You Need"), it replaced recurrence with attention.

<strong>Key Innovation:</strong>
• No recurrence or convolution
• Pure attention + feedforward layers
• Fully parallelizable → trains fast at scale

<strong>Main Components:</strong>
• <strong>Embedding + positional encoding:</strong> Turn tokens into vectors with order info
• <strong>Multi-head self-attention:</strong> Relate all tokens
• <strong>Feedforward layers:</strong> Process each position
• <strong>Residual connections + layer norm:</strong> Stabilize training

<strong>Encoder vs Decoder:</strong>
• <strong>Encoder:</strong> Reads and understands (BERT)
• <strong>Decoder:</strong> Generates text (GPT)
• <strong>Encoder-Decoder:</strong> Translation, summarization (T5)

<strong>Stacking:</strong>
• Many identical layers stacked deep
• Each layer refines the representation
• GPT-3 has 96 layers

<strong>Why It Won:</strong>
Parallelism + attention made it possible to train enormous models on enormous data — enabling today's LLMs.`,
                    code: `# Conceptual Transformer block (illustrative pseudo-code)
def transformer_block(x):
    # 1. Multi-head self-attention with residual connection
    attn_out = multi_head_attention(x)
    x = layer_norm(x + attn_out)      # residual + norm

    # 2. Feedforward network with residual connection
    ff_out = feed_forward(x)
    x = layer_norm(x + ff_out)        # residual + norm
    return x

# A full model stacks many such blocks
def transformer(tokens):
    x = embed(tokens) + positional_encoding(tokens)
    for _ in range(12):               # e.g., 12 layers
        x = transformer_block(x)
    return output_projection(x)

print("Transformer = stacked (attention + feedforward) blocks")
print("Residual connections + layer norm keep it trainable")`
                },
                {
                    title: "Self-Attention Explained",
                    content: `Self-attention is attention applied within a single sequence — each token attends to all tokens in the same sequence, including itself.

<strong>Why "Self"?</strong>
• Query, Key, and Value all come from the same input
• Each token builds a context-aware representation
• Captures relationships within the sequence

<strong>Example:</strong>
In "The animal didn't cross the street because it was tired":
• Self-attention links "it" to "animal"
• The model resolves the reference through attention

<strong>Step by Step:</strong>
1. Create Q, K, V from each token's embedding
2. Each token's query compares against all keys
3. Softmax the scores into attention weights
4. Blend the values by those weights
5. Output = context-enriched representation

<strong>Contextual Meaning:</strong>
• "bank" (river) vs "bank" (money) get different representations
• Self-attention makes embeddings context-dependent

<strong>Computational Note:</strong>
• Cost grows with sequence length squared (O(n²))
• This limits context length; many efficiency variants exist`,
                    code: `import numpy as np

# Self-attention: Q, K, V derived from the SAME input
np.random.seed(1)
tokens = np.random.randn(4, 6)   # 4 tokens, 6-dim each

# Learned projection matrices (random here for illustration)
Wq = np.random.randn(6, 6)
Wk = np.random.randn(6, 6)
Wv = np.random.randn(6, 6)

Q = tokens @ Wq
K = tokens @ Wk
V = tokens @ Wv

scores = Q @ K.T / np.sqrt(6)
weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
context = weights @ V

print("Each token now encodes context from all tokens")
print("Output shape:", context.shape)
print("Token 3 attends most to token:",
      int(np.argmax(weights[2])))`
                },
                {
                    title: "Multi-Head Attention",
                    content: `Multi-head attention runs several attention operations in parallel, letting the model focus on different types of relationships simultaneously.

<strong>The Idea:</strong>
• Split the representation into multiple "heads"
• Each head learns its own Q, K, V projections
• Each head attends to different aspects
• Concatenate and combine the results

<strong>Why Multiple Heads?</strong>
• One head might track syntax
• Another might track subject-verb links
• Another might track long-range references
• Together they capture richer relationships

<strong>How It Works:</strong>
1. Project input into h sets of Q, K, V
2. Run attention independently in each head
3. Concatenate the h outputs
4. Apply a final linear projection

<strong>Typical Configuration:</strong>
• 8-16 heads is common
• Each head works in a smaller dimension
• Total compute stays manageable

<strong>Benefit:</strong>
Multi-head attention gives the model multiple "perspectives" on the same sequence, greatly improving expressiveness.`,
                    code: `import numpy as np

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    w = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return w @ V

# Multi-head attention with 2 heads
np.random.seed(0)
X = np.random.randn(4, 8)   # 4 tokens, model dim = 8
n_heads, head_dim = 2, 4

heads = []
for h in range(n_heads):
    # Each head has its own projections (random here)
    Wq = np.random.randn(8, head_dim)
    Wk = np.random.randn(8, head_dim)
    Wv = np.random.randn(8, head_dim)
    heads.append(attention(X @ Wq, X @ Wk, X @ Wv))

# Concatenate heads back to full dimension
multi_head = np.concatenate(heads, axis=-1)
print("Per-head output dim:", head_dim)
print("Concatenated shape:", multi_head.shape)  # (4, 8)`
                },
                {
                    title: "Positional Encoding",
                    content: `Attention has no built-in sense of order, so positional encoding injects information about where each token sits in the sequence.

<strong>The Problem:</strong>
• Self-attention treats input as a set, not a sequence
• "dog bites man" and "man bites dog" would look identical
• We must add position information

<strong>Sinusoidal Positional Encoding:</strong>
• Uses sine and cosine functions of different frequencies
• Each position gets a unique pattern
• Added to the token embeddings
• Generalizes to unseen sequence lengths

<strong>Learned Positional Embeddings:</strong>
• A trainable vector per position
• Simple and effective
• Used by BERT and GPT

<strong>Modern Approaches:</strong>
• <strong>RoPE (Rotary):</strong> Rotates embeddings by position (LLaMA)
• <strong>ALiBi:</strong> Biases attention by distance
• Better length generalization

<strong>Why It Matters:</strong>
Word order is essential to meaning — positional encoding is what lets Transformers understand sequence structure.`,
                    code: `import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])   # even dims: sine
    pe[:, 1::2] = np.cos(angle[:, 1::2])   # odd dims: cosine
    return pe

pe = positional_encoding(seq_len=6, d_model=8)
print("Positional encodings (each row = one position):")
print(np.round(pe, 2))

# These are ADDED to token embeddings so the model knows order
# token_representation = token_embedding + positional_encoding
print("\\nPosition 0 differs from position 5:",
      not np.allclose(pe[0], pe[5]))`
                },
                {
                    title: "BERT and GPT Models",
                    content: `BERT and GPT are two landmark Transformer models representing different design philosophies.

<strong>BERT (Bidirectional Encoder):</strong>
• Encoder-only architecture
• Reads text in both directions at once
• Trained with masked language modeling (fill in blanks)
• Great for understanding tasks: classification, NER, Q&A
• Not designed to generate text

<strong>GPT (Generative Pre-trained Transformer):</strong>
• Decoder-only architecture
• Reads left-to-right (autoregressive)
• Trained to predict the next token
• Excellent at generation: writing, chat, code
• Powers modern conversational AI

<strong>Key Difference:</strong>
• BERT: understands (bidirectional context)
• GPT: generates (unidirectional, predicts forward)

<strong>Pre-training + Fine-tuning:</strong>
• Both pre-train on huge unlabeled text
• Then adapt to specific tasks

<strong>Evolution:</strong>
• BERT → RoBERTa, DeBERTa
• GPT → GPT-2/3/4, ChatGPT
• Modern LLMs are mostly decoder-only (GPT-style)`,
                    code: `# BERT-style: understanding via masked language modeling
# Input:  "The [MASK] sat on the mat"
# Output: "cat" (uses context from BOTH sides)

# GPT-style: generation via next-token prediction
# Input:  "The cat sat on the"
# Output: "mat" (uses only LEFT context)

# Using pre-trained models (conceptual):
# from transformers import pipeline

# BERT for understanding:
# classifier = pipeline('sentiment-analysis')  # BERT-based
# classifier("I love this!")  -> POSITIVE

# GPT for generation:
# generator = pipeline('text-generation', model='gpt2')
# generator("The future of AI is")  -> continues the text

print("BERT -> understanding (bidirectional, encoder)")
print("GPT  -> generation (left-to-right, decoder)")`
                },
                {
                    title: "Fine-tuning LLMs",
                    content: `Fine-tuning adapts a pre-trained LLM to a specific task or domain using additional training on targeted data.

<strong>Why Fine-tune?</strong>
• Pre-trained models are general-purpose
• Fine-tuning specializes them (legal, medical, support)
• Achieves strong results with less data than training from scratch

<strong>Full Fine-tuning:</strong>
• Update all model parameters
• Most powerful but expensive
• Requires significant compute and memory

<strong>Parameter-Efficient Fine-tuning (PEFT):</strong>
• <strong>LoRA:</strong> Train small low-rank adapter matrices
• <strong>Prefix/Prompt tuning:</strong> Learn soft prompt vectors
• Update a tiny fraction of parameters
• Cheap, fast, and effective

<strong>Instruction Tuning:</strong>
• Fine-tune on instruction-response pairs
• Makes models follow directions better

<strong>RLHF:</strong>
• Reinforcement Learning from Human Feedback
• Aligns outputs with human preferences
• Key to ChatGPT-style helpfulness

<strong>When to Fine-tune vs Prompt:</strong>
• Simple task → prompt engineering may suffice
• Consistent, specialized behavior → fine-tune`,
                    code: `# Parameter-efficient fine-tuning with LoRA (conceptual)
# Instead of updating a huge weight matrix W,
# learn small matrices A and B where update = A @ B

import numpy as np

# Original frozen weight (large)
W = np.random.randn(1000, 1000)   # 1,000,000 params (frozen)

# LoRA: low-rank update (rank r=8)
r = 8
A = np.random.randn(1000, r)      # 8,000 params
B = np.zeros((r, 1000))           # 8,000 params
# Only 16,000 trainable params vs 1,000,000!

# Effective weight during forward pass
def effective_weight():
    return W + A @ B   # W frozen, A and B trained

trainable = A.size + B.size
print("Full params:      {:,}".format(W.size))
print("LoRA params:      {:,}".format(trainable))
print("Reduction:        {:.1f}x fewer".format(W.size / trainable))`
                },
                {
                    title: "Prompt Engineering",
                    content: `Prompt engineering is the practice of crafting inputs that guide an LLM to produce the desired output — without changing the model.

<strong>Why It Matters:</strong>
• The same model gives very different results based on the prompt
• Often faster and cheaper than fine-tuning
• A core skill for using LLMs effectively

<strong>Core Techniques:</strong>
• <strong>Zero-shot:</strong> Just ask directly
• <strong>Few-shot:</strong> Provide examples in the prompt
• <strong>Chain-of-thought:</strong> Ask the model to reason step by step
• <strong>Role prompting:</strong> "You are an expert..."

<strong>Best Practices:</strong>
• Be specific and clear
• Provide context and constraints
• Show the desired format with examples
• Break complex tasks into steps

<strong>Chain-of-Thought:</strong>
Adding "Let's think step by step" dramatically improves reasoning on math and logic problems.

<strong>Advanced Patterns:</strong>
• Self-consistency (sample multiple reasoning paths)
• ReAct (reason + act with tools)
• Retrieval-augmented prompts

<strong>Iterate:</strong>
Prompt engineering is experimental — test, observe, and refine.`,
                    code: `# Prompt patterns (conceptual examples)

# Zero-shot: direct instruction
zero_shot = "Classify the sentiment: 'This movie was amazing!'"

# Few-shot: teach by example
few_shot = '''Classify sentiment:
Text: "I hate waiting" -> Negative
Text: "Best day ever!" -> Positive
Text: "The food was okay" -> '''

# Chain-of-thought: elicit reasoning
cot = '''Q: A shop has 23 apples. It sells 8 and buys 12 more.
How many apples now?
A: Let's think step by step.
Start: 23. Sell 8 -> 15. Buy 12 -> 27. Answer: 27.'''

# Role prompting: set persona and expertise
role = "You are an expert Python tutor. Explain recursion simply."

for name, p in [("zero_shot", zero_shot), ("few_shot", few_shot)]:
    print("=== {} ===\\n{}\\n".format(name, p))`
                },
                {
                    title: "Transfer Learning",
                    content: `Transfer learning is the paradigm that makes LLMs practical: pre-train once on massive data, then adapt to many tasks.

<strong>The Two-Stage Process:</strong>
1. <strong>Pre-training:</strong> Learn general language on huge unlabeled text (expensive, done once)
2. <strong>Adaptation:</strong> Specialize for a task (cheap, done many times)

<strong>Why It Is Transformative:</strong>
• Knowledge learned once is reused everywhere
• Small teams leverage billion-dollar pre-training
• Strong results with limited task data

<strong>Ways to Adapt:</strong>
• <strong>Fine-tuning:</strong> Update weights on task data
• <strong>Prompting:</strong> Guide via input, no weight changes
• <strong>In-context learning:</strong> Learn from examples in the prompt
• <strong>Retrieval augmentation:</strong> Supply external knowledge

<strong>Foundation Models:</strong>
• Large models pre-trained broadly
• Serve as a base for countless applications
• GPT, BERT, and their descendants

<strong>The Payoff:</strong>
Transfer learning turned NLP from task-by-task engineering into "adapt a foundation model," accelerating the entire field.`,
                    code: `# In-context learning: the model "learns" from the prompt alone,
# with NO weight updates.

in_context = '''Translate English to French:
sea otter -> loutre de mer
cheese -> fromage
hello -> '''
# The model infers the pattern and outputs "bonjour"

# Transfer learning workflow (conceptual):
# 1. Load a foundation model (pre-trained on huge corpus)
#    from transformers import AutoModel
#    model = AutoModel.from_pretrained('bert-base-uncased')
#
# 2. Adapt it to your task:
#    - Add a classification head, OR
#    - Fine-tune with LoRA, OR
#    - Just prompt it well

print("Pre-train once (general) -> adapt many times (specific)")
print("This is why one model powers thousands of applications")`
                },
                {
                    title: "Ethical Considerations",
                    content: `LLMs are powerful but raise serious ethical concerns that responsible practitioners must address.

<strong>Bias and Fairness:</strong>
• Models learn biases present in training data
• Can produce stereotyped or unfair outputs
• Requires auditing and mitigation

<strong>Misinformation:</strong>
• LLMs can "hallucinate" — generate confident falsehoods
• May spread inaccurate information
• Outputs need verification for factual tasks

<strong>Privacy:</strong>
• Training data may contain personal information
• Models can memorize and leak sensitive data
• Careful data handling is essential

<strong>Misuse Potential:</strong>
• Generating spam, phishing, or malicious content
• Deepfakes and impersonation
• Academic dishonesty

<strong>Environmental Impact:</strong>
• Training large models consumes significant energy
• Motivates efficiency research

<strong>Responsible Practices:</strong>
• Test for bias and harmful outputs
• Add safety filters and guardrails
• Be transparent about limitations
• Keep humans in the loop for high-stakes decisions
• Cite sources and enable verification`,
                    code: `# Responsible LLM usage: validate and guardrail outputs

def safe_llm_response(prompt, model):
    response = model.generate(prompt)   # pseudo-code

    # 1. Content safety check
    if contains_harmful_content(response):
        return "I can't help with that request."

    # 2. Fact-check for factual claims
    if is_factual_query(prompt):
        response += "\\n(Please verify important facts independently.)"

    # 3. Flag low confidence / possible hallucination
    if model.confidence(response) < 0.5:
        response += "\\n(I'm uncertain about this answer.)"

    return response

# Key principles:
principles = [
    "Test for bias and harmful outputs",
    "Verify facts; LLMs can hallucinate",
    "Protect user privacy",
    "Keep humans in the loop for high-stakes use",
    "Be transparent about limitations"
]
for p in principles:
    print("-", p)`
                },
                {
                    title: "Practical Applications",
                    content: `LLMs have unlocked a wide range of real-world applications across industries.

<strong>Content and Writing:</strong>
• Drafting, editing, and summarizing text
• Marketing copy and creative writing
• Translation across languages

<strong>Software Development:</strong>
• Code generation and completion
• Bug fixing and explanation
• Documentation and test writing

<strong>Conversational AI:</strong>
• Customer support chatbots
• Virtual assistants
• Interactive tutoring

<strong>Knowledge Work:</strong>
• Question answering over documents
• Research assistance
• Data extraction and analysis

<strong>Retrieval-Augmented Generation (RAG):</strong>
• Combine LLMs with a knowledge base
• Ground answers in your own documents
• Reduce hallucinations, cite sources

<strong>Building LLM Apps:</strong>
• APIs (OpenAI, Anthropic, Google)
• Frameworks: LangChain, LlamaIndex
• Vector databases for retrieval

<strong>The Future:</strong>
• Multimodal models (text + image + audio)
• Autonomous agents that use tools
• More efficient, smaller, specialized models`,
                    code: `# Retrieval-Augmented Generation (RAG) pattern
import numpy as np

# 1. Knowledge base of document embeddings (precomputed)
docs = {
    "Refunds are processed within 5 business days.": np.array([0.9, 0.1]),
    "Our office hours are 9am to 5pm EST.":          np.array([0.1, 0.9]),
    "Shipping is free on orders over 50 dollars.":   np.array([0.5, 0.5]),
}

def embed(text):            # pseudo-embedding
    return np.array([len(text) % 10 / 10, 0.3])

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def rag_answer(question):
    q = embed(question)
    # Retrieve the most relevant document
    best = max(docs, key=lambda d: cosine(q, docs[d]))
    # LLM would generate an answer grounded in 'best'
    return "Based on our docs: " + best

print(rag_answer("How long do refunds take?"))
# RAG grounds the LLM in real data to reduce hallucination`
                },
                {
                    title: "Future of LLMs",
                    content: `The field of large language models is evolving rapidly. Here are the key directions shaping what comes next.

<strong>Multimodal Models:</strong>
• Understand and generate text, images, audio, and video together
• Examples: GPT-4V, Gemini
• Richer, more human-like interaction

<strong>Autonomous Agents:</strong>
• LLMs that plan, use tools, and act
• Break complex goals into steps
• Interact with software, APIs, and the web

<strong>Efficiency and Accessibility:</strong>
• Smaller models matching larger ones
• Quantization and distillation
• Running powerful models on local devices

<strong>Longer Context:</strong>
• Handling entire books or codebases at once
• New attention mechanisms for scale

<strong>Better Reasoning:</strong>
• Improved logical and mathematical reasoning
• Reduced hallucinations
• Verifiable, grounded outputs

<strong>Alignment and Safety:</strong>
• Making models more helpful, honest, and harmless
• Better control and interpretability

<strong>Specialization:</strong>
• Domain-specific models (science, medicine, law)
• Custom models fine-tuned for organizations

<strong>Staying Current:</strong>
The pace is fast — continuous learning is essential in this field.`,
                    code: `# The trajectory of LLM development (conceptual)

trends = {
    "Multimodal":   "Text + images + audio + video together",
    "Agents":       "Plan, use tools, and take actions autonomously",
    "Efficiency":   "Smaller, faster models via distillation/quantization",
    "Long context": "Process entire books or codebases at once",
    "Reasoning":    "Stronger logic, fewer hallucinations",
    "Alignment":    "More helpful, honest, and safe outputs",
}

print("Where LLMs are heading:\\n")
for area, description in trends.items():
    print("  {:<12}: {}".format(area, description))

print("\\nKey takeaway: keep learning - the field moves fast!")`
                }
            ]
        }
    ],
    realWorldML: [
        {
            number: "Module 10",
            title: "Production ML Systems",
            description: "Learn how a machine learning production system works across a breadth of components.",
            duration: "55 min",
            lessons: "10 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "ML System Architecture",
                "Model Deployment Strategies",
                "Serving Infrastructure",
                "Monitoring and Logging",
                "A/B Testing",
                "Model Versioning",
                "CI/CD for ML",
                "Performance Optimization",
                "Scalability Considerations",
                "Production Best Practices"
            ],
            detailedDescription: "Bridge the gap between development and production. Learn how to deploy ML models at scale, monitor their performance, and maintain them in production environments. Covers infrastructure, deployment strategies, and best practices for production ML systems.",
            detailedContent: [
                {
                    title: "ML System Architecture",
                    content: `A production ML system is far more than a model. The model is often a small part of a larger engineered system.

<strong>Components of an ML System:</strong>
• <strong>Data pipeline:</strong> Collect, clean, and transform data
• <strong>Feature store:</strong> Manage and serve features consistently
• <strong>Training pipeline:</strong> Train and validate models
• <strong>Model registry:</strong> Version and store models
• <strong>Serving layer:</strong> Deliver predictions
• <strong>Monitoring:</strong> Track health and performance

<strong>The Hidden Complexity:</strong>
• Model code is a tiny fraction of the system
• Data collection, infrastructure, and monitoring dominate
• "ML technical debt" accumulates quickly

<strong>Batch vs Online:</strong>
• <strong>Batch:</strong> Predict on schedules (nightly scoring)
• <strong>Online:</strong> Predict in real time (per request)

<strong>Design Considerations:</strong>
• Latency and throughput requirements
• Consistency between training and serving
• Scalability and reliability
• Reproducibility

<strong>Key Principle:</strong>
Design the whole system, not just the model — production success depends on the surrounding engineering.`,
                    code: `# Conceptual ML system pipeline
class MLSystem:
    def __init__(self):
        self.data_pipeline = None
        self.model = None
        self.monitor = None

    def data_flow(self):
        steps = [
            "1. Ingest raw data (databases, APIs, streams)",
            "2. Validate data quality",
            "3. Transform into features (feature store)",
            "4. Train / retrain model (training pipeline)",
            "5. Validate model (registry + tests)",
            "6. Deploy to serving layer",
            "7. Monitor predictions and data drift",
            "8. Trigger retraining when needed",
        ]
        return steps

for step in MLSystem().data_flow():
    print(step)
# The model itself is only ONE step in a larger system`
                },
                {
                    title: "Model Deployment Strategies",
                    content: `Deploying a model means making it available to serve predictions. The strategy affects risk, latency, and rollback ability.

<strong>Deployment Patterns:</strong>
• <strong>Shadow deployment:</strong> Run new model alongside old, compare silently
• <strong>Canary release:</strong> Route a small % of traffic to the new model
• <strong>Blue-green:</strong> Switch all traffic at once, keep old ready for rollback
• <strong>A/B testing:</strong> Split traffic to compare models

<strong>Serving Modes:</strong>
• <strong>REST/gRPC API:</strong> Real-time online predictions
• <strong>Batch scoring:</strong> Score large datasets on a schedule
• <strong>Edge deployment:</strong> On-device (mobile, IoT)
• <strong>Streaming:</strong> Predict on event streams

<strong>Packaging:</strong>
• Serialize the model (pickle, ONNX, SavedModel)
• Containerize with Docker
• Include preprocessing to avoid training-serving skew

<strong>Rollback Plan:</strong>
• Always keep the previous version deployable
• Automate rollback on failure

<strong>Goal:</strong>
Deploy safely, minimize risk, and be able to reverse quickly if something goes wrong.`,
                    code: `# Simple model serving with a REST API (Flask-style, conceptual)
import joblib

# Load the trained model once at startup
# model = joblib.load('model.pkl')

# from flask import Flask, request, jsonify
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
def predict(request_json, model):
    features = request_json['features']
    # Reuse the SAME preprocessing as training (avoid skew)
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0].max()
    return {
        'prediction': int(prediction),
        'confidence': float(proba),
        'model_version': 'v1.2.0'
    }

# Canary logic: route a fraction of traffic to the new model
import random
def route(request, old_model, new_model, canary_pct=0.1):
    model = new_model if random.random() < canary_pct else old_model
    return model
print("Serve via API + gradually shift traffic (canary)")`
                },
                {
                    title: "Serving Infrastructure",
                    content: `Serving infrastructure delivers predictions reliably at the required scale and speed.

<strong>Serving Options:</strong>
• <strong>Model servers:</strong> TensorFlow Serving, TorchServe, Triton
• <strong>Cloud services:</strong> SageMaker, Vertex AI, Azure ML
• <strong>Custom APIs:</strong> Flask/FastAPI + containers
• <strong>Serverless:</strong> Lambda/Cloud Functions for light loads

<strong>Key Requirements:</strong>
• <strong>Latency:</strong> How fast a prediction returns
• <strong>Throughput:</strong> Predictions per second
• <strong>Availability:</strong> Uptime and reliability

<strong>Optimization Techniques:</strong>
• <strong>Batching:</strong> Group requests for efficiency
• <strong>Caching:</strong> Reuse frequent predictions
• <strong>Model optimization:</strong> Quantization, pruning, ONNX
• <strong>Hardware:</strong> GPUs/TPUs for heavy models

<strong>Scaling:</strong>
• Horizontal scaling with load balancers
• Auto-scaling based on demand
• Container orchestration (Kubernetes)

<strong>Trade-offs:</strong>
Balance cost, latency, and complexity for your use case — not every model needs GPU real-time serving.`,
                    code: `# Request batching improves throughput on model servers
import numpy as np
import time

def predict_single(model, x):
    return model.predict(x.reshape(1, -1))

def predict_batch(model, batch):
    return model.predict(batch)   # one call, many inputs

# Simulated timing benefit of batching
class FakeModel:
    def predict(self, X):
        time.sleep(0.001)          # fixed per-call overhead
        return np.zeros(len(X))

model = FakeModel()
data = np.random.randn(100, 5)

t0 = time.time()
[predict_single(model, x) for x in data]     # 100 calls
single_time = time.time() - t0

t0 = time.time()
predict_batch(model, data)                    # 1 call
batch_time = time.time() - t0

print("Single (100 calls): {:.3f}s".format(single_time))
print("Batched (1 call):   {:.3f}s".format(batch_time))`
                },
                {
                    title: "Monitoring and Logging",
                    content: `Once deployed, models must be monitored continuously. Unlike traditional software, ML systems can silently degrade.

<strong>What to Monitor:</strong>
• <strong>System metrics:</strong> Latency, throughput, error rate, uptime
• <strong>Model metrics:</strong> Accuracy, precision, recall over time
• <strong>Data metrics:</strong> Input distributions, missing values
• <strong>Business metrics:</strong> Revenue, conversion, user impact

<strong>Data Drift:</strong>
• Input data distribution changes over time
• Model was trained on old patterns
• Performance degrades silently

<strong>Concept Drift:</strong>
• The relationship between inputs and target changes
• Example: shopping behavior shifts after an event
• Requires retraining

<strong>Logging Best Practices:</strong>
• Log inputs, predictions, and outcomes
• Enable debugging and auditing
• Respect privacy regulations

<strong>Alerting:</strong>
• Set thresholds on key metrics
• Alert on drift and performance drops
• Automate responses where possible

<strong>Why It Matters:</strong>
A model that was accurate at launch can become dangerously wrong months later without anyone noticing — monitoring catches this.`,
                    code: `import numpy as np
from scipy import stats

def detect_drift(reference, current, threshold=0.05):
    # Kolmogorov-Smirnov test: are the distributions different?
    statistic, p_value = stats.ks_2samp(reference, current)
    drift = p_value < threshold
    return drift, p_value

# Training-time feature distribution
np.random.seed(0)
reference = np.random.normal(50, 10, 1000)

# Production data this week (shifted -> drift!)
current_ok = np.random.normal(50, 10, 1000)
current_drift = np.random.normal(65, 10, 1000)

for name, data in [("Stable", current_ok), ("Shifted", current_drift)]:
    drifted, p = detect_drift(reference, data)
    print("{}: drift={}, p-value={:.4f}".format(name, drifted, p))
# Detected drift -> trigger investigation or retraining`
                },
                {
                    title: "A/B Testing",
                    content: `A/B testing compares two models (or a model vs baseline) on live traffic to measure real-world impact.

<strong>How It Works:</strong>
• Split users randomly into groups
• Group A gets the current model (control)
• Group B gets the new model (treatment)
• Compare outcomes statistically

<strong>Why Offline Metrics Are Not Enough:</strong>
• Higher accuracy does not guarantee better business results
• Real users behave differently than test data
• A/B tests measure actual impact

<strong>What to Measure:</strong>
• Primary business metric (conversion, revenue, engagement)
• Guardrail metrics (latency, error rate)
• User satisfaction

<strong>Statistical Rigor:</strong>
• Ensure sufficient sample size (statistical power)
• Run long enough to be significant
• Check for statistical significance (p-value, confidence intervals)
• Avoid peeking and stopping early

<strong>Common Pitfalls:</strong>
• Too small a sample
• Ignoring seasonality
• Multiple comparisons inflating false positives

<strong>Decision:</strong>
Roll out the new model only if it shows a significant, meaningful improvement without harming guardrails.`,
                    code: `import numpy as np
from scipy import stats

# A/B test: does model B convert better than model A?
np.random.seed(0)

# Control (A): 10% conversion, Treatment (B): 12% conversion
n = 5000
conversions_a = np.random.binomial(1, 0.10, n)
conversions_b = np.random.binomial(1, 0.12, n)

rate_a = conversions_a.mean()
rate_b = conversions_b.mean()

# Two-proportion z-test
count = [conversions_a.sum(), conversions_b.sum()]
_, p_value = stats.ttest_ind(conversions_a, conversions_b)

print("Model A conversion: {:.2%}".format(rate_a))
print("Model B conversion: {:.2%}".format(rate_b))
print("Lift: {:.2%}".format(rate_b - rate_a))
print("p-value: {:.4f}".format(p_value))
print("Decision:",
      "Ship B" if p_value < 0.05 and rate_b > rate_a else "Keep A")`
                },
                {
                    title: "Model Versioning",
                    content: `Model versioning tracks models, data, and code together so results are reproducible and rollbacks are safe.

<strong>What to Version:</strong>
• <strong>Model artifacts:</strong> The trained weights
• <strong>Code:</strong> Training and preprocessing scripts
• <strong>Data:</strong> The exact dataset used
• <strong>Config:</strong> Hyperparameters and environment
• <strong>Metrics:</strong> Performance at training time

<strong>Why It Matters:</strong>
• Reproduce any past result
• Roll back to a known-good model instantly
• Audit and compliance
• Collaborate across a team

<strong>Model Registry:</strong>
• Central store for model versions
• Tracks lineage and stage (staging/production)
• Tools: MLflow, DVC, Weights & Biases

<strong>Semantic Versioning:</strong>
• Major.Minor.Patch (e.g., v2.1.3)
• Communicate the scope of changes

<strong>Reproducibility Checklist:</strong>
• Fixed random seeds
• Pinned dependencies
• Versioned data snapshots
• Logged hyperparameters

<strong>Goal:</strong>
Any model in production should be fully traceable back to the exact data, code, and config that produced it.`,
                    code: `import json
import hashlib
from datetime import datetime

def register_model(model_path, dataset, hyperparams, metrics):
    # Create a reproducible version record
    data_hash = hashlib.md5(str(dataset).encode()).hexdigest()[:8]
    record = {
        'version': 'v2.1.0',
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'data_hash': data_hash,       # exact dataset fingerprint
        'hyperparameters': hyperparams,
        'metrics': metrics,
        'stage': 'staging'
    }
    return record

record = register_model(
    model_path='models/classifier_v2.1.0.pkl',
    dataset='customers_2024_q1',
    hyperparams={'lr': 0.01, 'depth': 6},
    metrics={'accuracy': 0.94, 'f1': 0.91}
)
print(json.dumps(record, indent=2))
# In practice: mlflow.log_model(), mlflow.log_params(), etc.`
                },
                {
                    title: "CI/CD for ML",
                    content: `CI/CD (Continuous Integration / Continuous Deployment) automates testing and deploying ML systems, often called MLOps.

<strong>Beyond Traditional CI/CD:</strong>
• Code changes AND data/model changes trigger pipelines
• Must test data quality and model performance
• Sometimes called CI/CD/CT (Continuous Training)

<strong>Continuous Integration:</strong>
• Run unit tests on code
• Validate data schemas
• Test preprocessing logic
• Check model quality on a validation set

<strong>Continuous Deployment:</strong>
• Automatically deploy validated models
• Use canary/blue-green strategies
• Automated rollback on failure

<strong>Continuous Training:</strong>
• Retrain on fresh data automatically
• Triggered by schedule or drift detection
• Validate before promoting

<strong>Testing ML Pipelines:</strong>
• Data validation tests
• Model performance gates (min accuracy)
• Integration tests for the serving path

<strong>Tools:</strong>
• GitHub Actions, GitLab CI, Jenkins
• Kubeflow, MLflow, Airflow

<strong>Benefit:</strong>
Automation makes ML deployment reliable, repeatable, and fast — reducing manual errors.`,
                    code: `# ML CI/CD pipeline with quality gates (conceptual)

def ml_pipeline(new_data, current_model):
    results = {}

    # 1. Data validation gate
    if not validate_schema(new_data):
        return "FAILED: data schema invalid"
    results['data'] = 'passed'

    # 2. Train candidate model
    candidate = train_model(new_data)

    # 3. Performance gate: must beat a minimum threshold
    accuracy = evaluate(candidate)
    if accuracy < 0.90:
        return "FAILED: accuracy {:.2f} below 0.90".format(accuracy)
    results['accuracy'] = accuracy

    # 4. Regression gate: must not be worse than current model
    if accuracy < evaluate(current_model):
        return "FAILED: worse than current model"

    # 5. Deploy (canary)
    deploy(candidate, strategy='canary')
    results['deployed'] = True
    return results

# Placeholder helpers
def validate_schema(d): return True
def train_model(d): return "model"
def evaluate(m): return 0.93
def deploy(m, strategy): pass

print(ml_pipeline("data", "old_model"))`
                },
                {
                    title: "Performance Optimization",
                    content: `Production models must be fast and cost-efficient. Several techniques reduce latency and resource usage.

<strong>Model Compression:</strong>
• <strong>Quantization:</strong> Use lower-precision numbers (int8 vs float32)
• <strong>Pruning:</strong> Remove unimportant weights
• <strong>Distillation:</strong> Train a small model to mimic a large one
• <strong>Result:</strong> Smaller, faster models with minimal accuracy loss

<strong>Inference Optimization:</strong>
• Convert to optimized formats (ONNX, TensorRT)
• Fuse operations
• Use hardware accelerators (GPU/TPU)

<strong>System-Level:</strong>
• <strong>Batching:</strong> Process multiple requests together
• <strong>Caching:</strong> Store frequent predictions
• <strong>Async processing:</strong> Non-blocking request handling

<strong>Measuring Performance:</strong>
• Latency percentiles (p50, p95, p99)
• Throughput (requests/second)
• Cost per prediction

<strong>Trade-offs:</strong>
• Speed vs accuracy
• Cost vs latency
• Complexity vs maintainability

<strong>Approach:</strong>
Profile first to find bottlenecks, optimize the biggest ones, and validate that accuracy remains acceptable.`,
                    code: `import numpy as np

# Quantization: reduce precision to shrink and speed up a model
def quantize(weights, bits=8):
    # Map float32 weights to int8 range
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / (2**bits - 1)
    quantized = np.round((weights - w_min) / scale).astype(np.uint8)
    return quantized, scale, w_min

def dequantize(quantized, scale, w_min):
    return quantized.astype(np.float32) * scale + w_min

np.random.seed(0)
weights = np.random.randn(1000).astype(np.float32)
q, scale, w_min = quantize(weights)

original_bytes = weights.nbytes
quantized_bytes = q.nbytes
print("Original size:  {} bytes (float32)".format(original_bytes))
print("Quantized size: {} bytes (int8)".format(quantized_bytes))
print("Compression:    {:.1f}x smaller".format(
    original_bytes / quantized_bytes))

restored = dequantize(q, scale, w_min)
print("Max error: {:.5f}".format(np.abs(weights - restored).max()))`
                },
                {
                    title: "Scalability Considerations",
                    content: `As usage grows, ML systems must scale to handle more data, more requests, and larger models.

<strong>Scaling Dimensions:</strong>
• <strong>Data volume:</strong> More training data
• <strong>Request load:</strong> More prediction traffic
• <strong>Model size:</strong> Larger, more complex models

<strong>Horizontal Scaling:</strong>
• Add more machines/replicas
• Load balance across them
• Auto-scale based on demand
• The standard approach for serving

<strong>Vertical Scaling:</strong>
• Use bigger machines (more CPU/RAM/GPU)
• Simpler but has limits

<strong>Distributed Training:</strong>
• <strong>Data parallelism:</strong> Split data across workers
• <strong>Model parallelism:</strong> Split the model across devices
• Needed for very large models

<strong>Infrastructure:</strong>
• Container orchestration (Kubernetes)
• Message queues for async work
• Distributed storage and feature stores

<strong>Cost Management:</strong>
• Auto-scale down during low demand
• Use spot/preemptible instances for training
• Right-size resources

<strong>Design Principle:</strong>
Build stateless, containerized services so you can scale horizontally with demand.`,
                    code: `# Horizontal auto-scaling logic (conceptual)

def autoscale(current_load, current_replicas,
              target_per_replica=100,
              min_replicas=2, max_replicas=20):
    # Desired replicas based on load
    desired = max(min_replicas,
                  -(-current_load // target_per_replica))  # ceil div
    desired = min(desired, max_replicas)

    if desired > current_replicas:
        action = "SCALE UP to {}".format(desired)
    elif desired < current_replicas:
        action = "SCALE DOWN to {}".format(desired)
    else:
        action = "NO CHANGE ({})".format(current_replicas)
    return desired, action

# Simulate varying load (requests/sec)
for load in [150, 500, 1200, 300]:
    replicas, action = autoscale(load, current_replicas=5)
    print("Load {:>4} req/s -> {}".format(load, action))
# Kubernetes HPA automates this in production`
                },
                {
                    title: "Production Best Practices",
                    content: `A consolidated set of practices for running ML systems reliably in production.

<strong>Before Deployment:</strong>
• Validate data quality and schemas
• Test the full pipeline end-to-end
• Establish baseline metrics
• Plan monitoring and rollback

<strong>Prevent Training-Serving Skew:</strong>
• Use the same preprocessing code in both
• Share a feature store
• Test that offline and online features match

<strong>Reliability:</strong>
• Handle failures gracefully (fallbacks, defaults)
• Set timeouts and retries
• Design for graceful degradation

<strong>Observability:</strong>
• Log inputs, outputs, and outcomes
• Monitor system, model, and business metrics
• Alert on drift and degradation

<strong>Governance:</strong>
• Version models, data, and code
• Document decisions and limitations
• Ensure privacy and compliance

<strong>Continuous Improvement:</strong>
• Retrain on fresh data
• A/B test changes
• Collect feedback loops

<strong>Team Practices:</strong>
• Treat ML like software engineering
• Automate everything you can
• Review models like you review code

<strong>Golden Rule:</strong>
The model is never "done" — production ML is a continuous lifecycle of monitoring, learning, and improving.`,
                    code: `# Production readiness checklist (as executable structure)

checklist = {
    "Data": [
        "Schema validation in place",
        "Data quality checks automated",
        "Feature store shared with training",
    ],
    "Model": [
        "Versioned in a registry",
        "Performance gates defined",
        "Rollback plan tested",
    ],
    "Serving": [
        "Same preprocessing as training",
        "Latency/throughput SLOs set",
        "Graceful failure handling",
    ],
    "Monitoring": [
        "Drift detection running",
        "Metric dashboards + alerts",
        "Prediction logging enabled",
    ],
}

for category, items in checklist.items():
    print("[{}]".format(category))
    for item in items:
        print("   [ ]", item)
    print()
print("Production ML = continuous lifecycle, not a one-time launch")`
                }
            ]
        },
        {
            number: "Module 11",
            title: "AutoML",
            description: "Learn principles and best practices for using automated machine learning to streamline model development.",
            duration: "40 min",
            lessons: "7 lessons",
            isNew: true,
            isLocked: false,
            topics: [
                "Introduction to AutoML",
                "Automated Feature Engineering",
                "Neural Architecture Search",
                "Hyperparameter Optimization",
                "Model Selection",
                "AutoML Tools and Platforms",
                "When to Use AutoML"
            ],
            detailedDescription: "Discover how AutoML can accelerate your machine learning workflow. Learn about automated feature engineering, hyperparameter tuning, and neural architecture search. Understand when AutoML is appropriate and how to integrate it into your development process.",
            detailedContent: [
                {
                    title: "Introduction to AutoML",
                    content: `AutoML (Automated Machine Learning) automates the time-consuming, iterative parts of building ML models.

<strong>What AutoML Automates:</strong>
• Data preprocessing and cleaning
• Feature engineering and selection
• Model selection
• Hyperparameter tuning
• Sometimes deployment

<strong>Why AutoML?</strong>
• Speeds up development
• Lowers the barrier to entry
• Frees experts for higher-value work
• Explores more options than manual tuning

<strong>Who Benefits:</strong>
• <strong>Non-experts:</strong> Build models without deep ML knowledge
• <strong>Experts:</strong> Automate tedious steps, focus on strategy
• <strong>Teams:</strong> Standardize and accelerate workflows

<strong>Popular AutoML Tools:</strong>
• <strong>Cloud:</strong> Google Vertex AI, Azure AutoML, AWS SageMaker Autopilot
• <strong>Open source:</strong> Auto-sklearn, TPOT, H2O AutoML, AutoKeras

<strong>The Reality:</strong>
AutoML is powerful but not magic — it works best combined with human judgment, domain knowledge, and good data.`,
                    code: `# AutoML with a high-level library (conceptual)
# Example uses a TPOT-style interface

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

# AutoML explores many pipelines automatically:
# from tpot import TPOTClassifier
# automl = TPOTClassifier(generations=5, population_size=20)
# automl.fit(X_tr, y_tr)
# print(automl.score(X_te, y_te))
# automl.export('best_pipeline.py')   # exports the winning pipeline

print("AutoML automates: preprocessing -> model -> hyperparameters")
print("You provide: the data and the objective")`
                },
                {
                    title: "Automated Feature Engineering",
                    content: `Automated feature engineering generates and selects useful features without manual crafting.

<strong>What It Does:</strong>
• Creates new features from existing ones
• Generates interactions, aggregations, transformations
• Selects the most predictive features
• Removes redundant or useless ones

<strong>Techniques:</strong>
• <strong>Transformations:</strong> Log, square root, polynomial
• <strong>Interactions:</strong> Products and ratios of features
• <strong>Aggregations:</strong> Group-based statistics
• <strong>Date/time expansion:</strong> Extract parts automatically

<strong>Deep Feature Synthesis:</strong>
• Automatically builds features across related tables
• Used by tools like Featuretools
• Great for relational data

<strong>Feature Selection Methods:</strong>
• Filter: statistical tests
• Wrapper: model-based evaluation
• Embedded: regularization (Lasso)

<strong>Benefits and Limits:</strong>
• Saves significant manual effort
• May miss domain-specific insights
• Can generate too many features (needs selection)

<strong>Best Practice:</strong>
Combine automated generation with domain knowledge for the strongest features.`,
                    code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Original features
df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [2, 4, 6, 8, 10],
    'c': [5, 3, 8, 1, 9]
})
y = np.array([0, 0, 1, 0, 1])

# 1. Automatically generate interactions and squares
poly = PolynomialFeatures(degree=2, include_bias=False)
generated = poly.fit_transform(df)
print("Generated {} features from {}".format(
    generated.shape[1], df.shape[1]))
print("Names:", poly.get_feature_names_out())

# 2. Automatically select the best features
selector = SelectKBest(f_classif, k=3)
selected = selector.fit_transform(generated, y)
print("\\nSelected top 3 features, shape:", selected.shape)`
                },
                {
                    title: "Neural Architecture Search",
                    content: `Neural Architecture Search (NAS) automates the design of neural network architectures.

<strong>The Problem:</strong>
• Designing networks is expert-intensive and slow
• How many layers? How many neurons? Which connections?
• NAS searches this space automatically

<strong>Components of NAS:</strong>
• <strong>Search space:</strong> Possible architectures to consider
• <strong>Search strategy:</strong> How to explore the space
• <strong>Evaluation:</strong> How to score each candidate

<strong>Search Strategies:</strong>
• <strong>Reinforcement learning:</strong> A controller proposes architectures
• <strong>Evolutionary:</strong> Mutate and select good architectures
• <strong>Gradient-based (DARTS):</strong> Make the search differentiable
• <strong>Random search:</strong> Surprisingly strong baseline

<strong>The Challenge:</strong>
• Training each candidate is expensive
• Full NAS can require enormous compute
• Efficiency techniques: weight sharing, early stopping, proxies

<strong>Notable Results:</strong>
• EfficientNet was designed with NAS
• Often finds architectures better than hand-designed ones

<strong>Practical Note:</strong>
NAS is powerful but compute-heavy — cloud AutoML services make it accessible without massive infrastructure.`,
                    code: `import random

# Simplified NAS: search over architecture configurations
search_space = {
    'n_layers': [2, 3, 4, 5],
    'units': [32, 64, 128, 256],
    'activation': ['relu', 'tanh', 'elu'],
    'dropout': [0.0, 0.2, 0.3, 0.5],
}

def sample_architecture(space):
    return {k: random.choice(v) for k, v in space.items()}

def evaluate(arch):
    # In reality: build, train, and validate the network.
    # Here we use a placeholder score.
    return random.uniform(0.7, 0.95)

# Random search over the architecture space
random.seed(0)
best_arch, best_score = None, 0
for _ in range(10):
    arch = sample_architecture(search_space)
    score = evaluate(arch)
    if score > best_score:
        best_arch, best_score = arch, score

print("Best architecture found:")
print(best_arch)
print("Validation score: {:.3f}".format(best_score))`
                },
                {
                    title: "Hyperparameter Optimization",
                    content: `Hyperparameter optimization (HPO) automatically finds the best model settings, one of the most valuable parts of AutoML.

<strong>Hyperparameters vs Parameters:</strong>
• Parameters: learned during training (weights)
• Hyperparameters: set before training (learning rate, depth)

<strong>Search Methods:</strong>
• <strong>Grid search:</strong> Try all combinations (thorough, slow)
• <strong>Random search:</strong> Sample randomly (efficient baseline)
• <strong>Bayesian optimization:</strong> Use past results to pick promising settings
• <strong>Hyperband:</strong> Allocate resources adaptively

<strong>Bayesian Optimization:</strong>
• Builds a probabilistic model of the objective
• Balances exploration and exploitation
• Much more efficient than grid/random for expensive models

<strong>Popular Tools:</strong>
• Optuna, Hyperopt, Ray Tune
• scikit-learn GridSearchCV / RandomizedSearchCV

<strong>Best Practices:</strong>
• Use cross-validation for reliable scores
• Define sensible ranges (log-scale for learning rates)
• Start broad, then narrow
• Set a compute budget

<strong>Impact:</strong>
Good HPO can turn a mediocre model into a strong one — often the highest-ROI automation.`,
                    code: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=15,
                           random_state=0)

# Define the hyperparameter search space
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
}

# Randomized search with cross-validation
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_dist,
    n_iter=15,          # try 15 random combinations
    cv=5,
    random_state=0
)
search.fit(X, y)

print("Best hyperparameters:")
print(search.best_params_)
print("Best CV score: {:.3f}".format(search.best_score_))
# For expensive models, use Optuna's Bayesian optimization instead`
                },
                {
                    title: "Model Selection",
                    content: `Automated model selection tries multiple algorithms and picks the best performer for your data.

<strong>Why Automate Selection?</strong>
• No single algorithm is best for all problems (No Free Lunch)
• Manually testing many models is tedious
• AutoML evaluates candidates systematically

<strong>Candidate Models:</strong>
• Linear/logistic regression
• Decision trees and random forests
• Gradient boosting (XGBoost, LightGBM)
• Support vector machines
• Neural networks

<strong>The Selection Process:</strong>
1. Train each candidate with cross-validation
2. Compare on a chosen metric
3. Optionally tune each one's hyperparameters
4. Select the best (or ensemble the top few)

<strong>Ensembling:</strong>
• Combine multiple strong models
• Often beats any single model
• AutoML tools frequently build ensembles automatically

<strong>Evaluation:</strong>
• Use appropriate metrics for the task
• Consider not just accuracy but latency and interpretability
• Validate on held-out data

<strong>Key Point:</strong>
AutoML explores the model space faster and more thoroughly than manual experimentation.`,
                    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=15,
                           random_state=0)

# Automatically compare several candidate models
candidates = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'SVM': SVC(),
}

results = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=5)
    results[name] = scores.mean()

best = max(results, key=results.get)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    marker = " <-- BEST" if name == best else ""
    print("{:<20}: {:.3f}{}".format(name, score, marker))`
                },
                {
                    title: "AutoML Tools and Platforms",
                    content: `A variety of tools bring AutoML capabilities to different users and use cases.

<strong>Cloud AutoML Platforms:</strong>
• <strong>Google Vertex AI:</strong> AutoML for tables, vision, text
• <strong>Azure AutoML:</strong> Integrated with Azure ML
• <strong>AWS SageMaker Autopilot:</strong> End-to-end automation
• Managed, scalable, minimal setup

<strong>Open-Source Libraries:</strong>
• <strong>Auto-sklearn:</strong> Automates scikit-learn pipelines
• <strong>TPOT:</strong> Uses genetic programming to build pipelines
• <strong>H2O AutoML:</strong> Fast, includes stacked ensembles
• <strong>AutoKeras:</strong> NAS for deep learning
• <strong>FLAML:</strong> Fast, lightweight (Microsoft)

<strong>Specialized Tools:</strong>
• <strong>Optuna:</strong> Hyperparameter optimization
• <strong>Featuretools:</strong> Automated feature engineering
• <strong>PyCaret:</strong> Low-code ML workflow

<strong>Choosing a Tool:</strong>
• Cloud platforms: ease, scale, integration
• Open source: control, cost, customization
• Consider your data type, budget, and expertise

<strong>Trade-offs:</strong>
Cloud tools are convenient but can be costly and less transparent; open-source tools offer flexibility but need more setup.`,
                    code: `# Comparison of AutoML tools by use case (reference)

tools = {
    "Auto-sklearn": "Tabular data, scikit-learn pipelines",
    "TPOT":         "Genetic search over full pipelines",
    "H2O AutoML":   "Fast, strong stacked ensembles",
    "AutoKeras":    "Deep learning / neural architecture search",
    "FLAML":        "Fast, cost-efficient, lightweight",
    "PyCaret":      "Low-code end-to-end workflow",
    "Vertex AI":    "Managed cloud AutoML (Google)",
    "Azure AutoML": "Managed cloud AutoML (Microsoft)",
}

print("AutoML tool reference:\\n")
for tool, use_case in tools.items():
    print("  {:<14}: {}".format(tool, use_case))

# Quick start example (PyCaret-style, conceptual):
# from pycaret.classification import setup, compare_models
# setup(data=df, target='label')
# best = compare_models()   # trains and ranks many models`
                },
                {
                    title: "When to Use AutoML",
                    content: `AutoML is powerful, but knowing when to use it — and when not to — is key to good outcomes.

<strong>Good Fits for AutoML:</strong>
• Standard, well-defined problems (classification, regression)
• Tabular data with clear targets
• Rapid prototyping and baselines
• Limited ML expertise on the team
• Need to explore many options quickly

<strong>When to Be Cautious:</strong>
• Highly specialized domains needing expert features
• Novel problems without established approaches
• Very large-scale custom systems
• When interpretability is critical
• Tight latency or resource constraints

<strong>AutoML Limitations:</strong>
• Can be a "black box"
• May miss domain-specific insights
• Compute cost can be high
• Still needs good, clean data
• Does not replace problem framing

<strong>Best of Both Worlds:</strong>
• Use AutoML for a strong baseline fast
• Apply domain knowledge to features
• Manually refine the top candidates
• Keep humans in the loop

<strong>Bottom Line:</strong>
AutoML accelerates and augments ML work — it is a powerful assistant, not a replacement for thoughtful engineering.`,
                    code: `# Decision helper: is AutoML a good fit here?

def should_use_automl(problem):
    score = 0
    reasons = []

    if problem['type'] in ('classification', 'regression'):
        score += 1; reasons.append("Standard problem type (+)")
    if problem['data'] == 'tabular':
        score += 1; reasons.append("Tabular data (+)")
    if problem['need_speed']:
        score += 1; reasons.append("Need a fast baseline (+)")
    if problem['ml_expertise'] == 'low':
        score += 1; reasons.append("Limited ML expertise (+)")
    if problem['interpretability_critical']:
        score -= 1; reasons.append("Interpretability critical (-)")
    if problem['highly_specialized']:
        score -= 1; reasons.append("Highly specialized domain (-)")

    return score, reasons

problem = {
    'type': 'classification', 'data': 'tabular',
    'need_speed': True, 'ml_expertise': 'low',
    'interpretability_critical': False,
    'highly_specialized': False,
}
score, reasons = should_use_automl(problem)
for r in reasons:
    print(r)
print("\\nRecommendation:",
      "Use AutoML" if score >= 2 else "Consider manual approach")`
                }
            ]
        },
        {
            number: "Module 12",
            title: "ML Fairness",
            description: "Learn principles and best practices for auditing ML models for fairness, including strategies for identifying and mitigating biases.",
            duration: "50 min",
            lessons: "9 lessons",
            isNew: false,
            isLocked: false,
            topics: [
                "Understanding Bias in ML",
                "Types of Bias",
                "Fairness Metrics",
                "Bias Detection Techniques",
                "Mitigation Strategies",
                "Fairness-Aware Algorithms",
                "Ethical Considerations",
                "Case Studies",
                "Best Practices for Fair ML"
            ],
            detailedDescription: "Build responsible AI systems by understanding fairness and bias in machine learning. Learn how to identify, measure, and mitigate bias in your models. Essential knowledge for creating ethical and equitable ML systems that benefit everyone.",
            detailedContent: [
                {
                    title: "Understanding Bias in ML",
                    content: `Bias in machine learning refers to systematic unfairness in a model's predictions that disadvantages certain groups.

<strong>What Is ML Bias?</strong>
• Models can produce unfair outcomes for different groups
• Often reflects biases present in the training data
• Can cause real harm (denied loans, jobs, healthcare)

<strong>Why It Happens:</strong>
• Historical data encodes past discrimination
• Unrepresentative or imbalanced datasets
• Proxy variables correlated with protected attributes
• Flawed problem framing or labels

<strong>Real-World Consequences:</strong>
• Hiring tools favoring certain demographics
• Facial recognition failing for some groups
• Credit scoring disadvantaging communities
• Healthcare models underserving populations

<strong>Protected Attributes:</strong>
• Race, gender, age, religion, disability
• Legally protected in many contexts
• Must be handled carefully

<strong>The Challenge:</strong>
• Removing a protected attribute is not enough (proxies remain)
• Fairness has multiple, sometimes conflicting definitions
• Requires deliberate measurement and mitigation

<strong>Why It Matters:</strong>
ML systems increasingly affect people's lives — fairness is both an ethical obligation and often a legal requirement.`,
                    code: `import numpy as np
import pandas as pd

# Illustrate how bias hides in data via proxy variables
np.random.seed(0)
df = pd.DataFrame({
    'group': np.random.choice(['A', 'B'], 200),
    'zip_code': np.random.randint(1, 100, 200),
})
# Historical bias: group B was approved less often
df['approved'] = np.where(
    df['group'] == 'A',
    np.random.binomial(1, 0.7, 200),
    np.random.binomial(1, 0.4, 200)
)

# Even if we DROP 'group', a correlated proxy can leak it
rates = df.groupby('group')['approved'].mean()
print("Approval rate by group:")
print(rates.round(3))
print("\\nGap:", round(abs(rates['A'] - rates['B']), 3))
print("Removing 'group' alone won't fix this - proxies remain")`
                },
                {
                    title: "Types of Bias",
                    content: `Bias enters ML systems at many stages. Understanding the types helps you find and address it.

<strong>Historical Bias:</strong>
• The world's existing inequalities are in the data
• Even perfect data collection captures past discrimination

<strong>Representation Bias:</strong>
• Some groups underrepresented in the data
• Model performs poorly for them
• Example: medical data skewed to one population

<strong>Measurement Bias:</strong>
• Features or labels measured differently across groups
• Proxy labels that do not mean the same thing

<strong>Aggregation Bias:</strong>
• One model forced onto distinct groups
• Ignores that groups may need different treatment

<strong>Sampling Bias:</strong>
• Data not collected representatively
• Skews toward certain populations

<strong>Evaluation Bias:</strong>
• Benchmarks not representative of all users
• Hides poor performance on subgroups

<strong>Deployment Bias:</strong>
• Model used in ways or contexts it was not designed for

<strong>Key Insight:</strong>
Bias can enter at data collection, labeling, modeling, evaluation, and deployment — audit every stage.`,
                    code: `# Detect representation bias: are all groups well-covered?
import pandas as pd
import numpy as np

np.random.seed(0)
# Training data heavily skewed toward group A
df = pd.DataFrame({
    'group': (['A'] * 900) + (['B'] * 100),
    'feature': np.random.randn(1000),
})

counts = df['group'].value_counts()
proportions = df['group'].value_counts(normalize=True)

print("Group representation:")
for g in counts.index:
    print("  Group {}: {} samples ({:.1%})".format(
        g, counts[g], proportions[g]))

# Warn if any group is severely underrepresented
for g, prop in proportions.items():
    if prop < 0.2:
        print("\\nWARNING: Group {} underrepresented ({:.1%})".format(
            g, prop))
        print("Model may perform poorly for this group")`
                },
                {
                    title: "Fairness Metrics",
                    content: `Fairness must be measured to be managed. Several metrics formalize different notions of fairness.

<strong>Demographic Parity:</strong>
• Positive prediction rate equal across groups
• P(ŷ=1 | group A) = P(ŷ=1 | group B)
• Ignores actual outcomes

<strong>Equal Opportunity:</strong>
• Equal true positive rate across groups
• Qualified people have equal chance of a positive prediction

<strong>Equalized Odds:</strong>
• Equal true positive AND false positive rates
• Stronger condition than equal opportunity

<strong>Predictive Parity:</strong>
• Equal precision across groups
• Predictions mean the same thing for everyone

<strong>Individual Fairness:</strong>
• Similar individuals get similar predictions

<strong>The Impossibility Result:</strong>
• Many fairness metrics cannot all hold at once
• Except in trivial cases, you must choose which to prioritize
• The right choice depends on context and values

<strong>Practical Approach:</strong>
Measure several metrics, understand the trade-offs, and pick based on the harm you most want to prevent.`,
                    code: `import numpy as np

# Compute fairness metrics across two groups
def fairness_metrics(y_true, y_pred, group):
    results = {}
    for g in np.unique(group):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        pos_rate = yp.mean()                       # demographic parity
        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0   # equal opportunity
        results[g] = {'positive_rate': pos_rate, 'tpr': tpr}
    return results

np.random.seed(0)
group = np.array(['A'] * 100 + ['B'] * 100)
y_true = np.random.binomial(1, 0.5, 200)
y_pred = np.where(group == 'A',
                  np.random.binomial(1, 0.6, 200),
                  np.random.binomial(1, 0.4, 200))

for g, m in fairness_metrics(y_true, y_pred, group).items():
    print("Group {}: positive_rate={:.2f}, TPR={:.2f}".format(
        g, m['positive_rate'], m['tpr']))`
                },
                {
                    title: "Bias Detection Techniques",
                    content: `Detecting bias is the first step toward fixing it. This requires systematic auditing across groups.

<strong>Disaggregated Evaluation:</strong>
• Break down performance by subgroup
• Overall accuracy can hide subgroup failures
• Report metrics per group, not just in aggregate

<strong>Confusion Matrix Per Group:</strong>
• Compare error types across groups
• Reveals if certain groups get more false negatives/positives

<strong>Fairness Audits:</strong>
• Systematically test against fairness metrics
• Use tools like Fairlearn, AIF360, What-If Tool

<strong>Slice Analysis:</strong>
• Examine performance on data "slices"
• Find where the model underperforms

<strong>Proxy Detection:</strong>
• Check if features correlate with protected attributes
• Zip code, name, or purchase history can be proxies

<strong>Counterfactual Testing:</strong>
• Change only the protected attribute
• See if the prediction changes (it should not, ideally)

<strong>Ongoing Monitoring:</strong>
• Bias can emerge over time as data shifts
• Audit continuously, not just once

<strong>Principle:</strong>
You cannot fix what you do not measure — make bias detection a standard part of evaluation.`,
                    code: `import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Disaggregated evaluation reveals hidden bias
np.random.seed(0)
df = pd.DataFrame({
    'group': np.random.choice(['A', 'B'], 400),
})
df['y_true'] = np.random.binomial(1, 0.5, 400)
# Model is accurate for A but poor for B
df['y_pred'] = np.where(
    df['group'] == 'A',
    df['y_true'],                                  # perfect for A
    np.random.binomial(1, 0.5, 400)               # random for B
)

overall = accuracy_score(df['y_true'], df['y_pred'])
print("Overall accuracy: {:.2f}".format(overall))
print("\\nPer-group accuracy (the real story):")
for g in ['A', 'B']:
    sub = df[df['group'] == g]
    acc = accuracy_score(sub['y_true'], sub['y_pred'])
    print("  Group {}: {:.2f}".format(g, acc))
# Overall looks OK, but group B is failing!`
                },
                {
                    title: "Mitigation Strategies",
                    content: `Once bias is detected, mitigation techniques can reduce it. They apply at different stages of the ML pipeline.

<strong>Pre-processing (fix the data):</strong>
• Reweight or resample to balance groups
• Remove or transform biased features
• Augment underrepresented groups
• Relabel to correct biased labels

<strong>In-processing (fix the training):</strong>
• Add fairness constraints to the objective
• Adversarial debiasing (prevent predicting the protected attribute)
• Regularization that penalizes unfairness

<strong>Post-processing (fix the outputs):</strong>
• Adjust decision thresholds per group
• Calibrate predictions for fairness
• Modify outputs to satisfy fairness metrics

<strong>Trade-offs:</strong>
• Fairness often trades off with accuracy
• Different mitigation suits different constraints
• No one-size-fits-all solution

<strong>Choosing a Strategy:</strong>
• Pre-processing: when you control the data
• In-processing: when you control training
• Post-processing: when you only control outputs

<strong>Validate:</strong>
Always re-measure fairness after mitigation to confirm it actually helped without introducing new harms.`,
                    code: `import numpy as np

# Post-processing: group-specific thresholds for equal opportunity
def find_fair_thresholds(scores, y_true, group, target_tpr=0.7):
    thresholds = {}
    for g in np.unique(group):
        mask = group == g
        g_scores, g_true = scores[mask], y_true[mask]
        best_t = 0.5
        # Find a threshold that achieves the target TPR for this group
        for t in np.linspace(0.1, 0.9, 17):
            preds = (g_scores >= t).astype(int)
            tp = ((preds == 1) & (g_true == 1)).sum()
            pos = (g_true == 1).sum()
            tpr = tp / pos if pos > 0 else 0
            if tpr >= target_tpr:
                best_t = t
        thresholds[g] = best_t
    return thresholds

np.random.seed(0)
group = np.array(['A'] * 100 + ['B'] * 100)
y_true = np.random.binomial(1, 0.5, 200)
# Group B tends to get lower scores -> needs a lower threshold
scores = np.where(group == 'A',
                  np.random.uniform(0.3, 0.9, 200),
                  np.random.uniform(0.1, 0.7, 200))

thresholds = find_fair_thresholds(scores, y_true, group)
print("Group-specific thresholds for equal opportunity:")
print(thresholds)`
                },
                {
                    title: "Fairness-Aware Algorithms",
                    content: `Fairness-aware algorithms build fairness directly into the learning process rather than treating it as an afterthought.

<strong>Constrained Optimization:</strong>
• Optimize accuracy subject to fairness constraints
• Explicitly bound the disparity between groups
• Tools: Fairlearn's ExponentiatedGradient

<strong>Adversarial Debiasing:</strong>
• Train the main model to predict the target
• Train an adversary to predict the protected attribute from predictions
• The main model learns to hide protected info
• Result: predictions independent of the protected attribute

<strong>Fair Representation Learning:</strong>
• Learn representations that remove protected information
• Downstream models built on fair features

<strong>Regularization for Fairness:</strong>
• Add a penalty term for unfairness to the loss
• Balances accuracy and fairness via a tunable weight

<strong>Prejudice Remover:</strong>
• Penalizes mutual information with protected attributes

<strong>Trade-off Control:</strong>
• A hyperparameter tunes the accuracy-fairness balance
• Choose based on requirements and context

<strong>Toolkits:</strong>
• <strong>Fairlearn:</strong> Constraints and mitigation
• <strong>AIF360:</strong> Comprehensive fairness algorithms
• Make fairness-aware ML practical and accessible.`,
                    code: `# Fairness-aware training with Fairlearn (conceptual)

# from fairlearn.reductions import ExponentiatedGradient, DemographicParity
# from sklearn.linear_model import LogisticRegression

# Base model
# base = LogisticRegression()

# Wrap it with a fairness constraint (demographic parity)
# fair_model = ExponentiatedGradient(
#     base, constraints=DemographicParity()
# )
# fair_model.fit(X_train, y_train, sensitive_features=group_train)

# Adversarial debiasing intuition:
def adversarial_debiasing_idea():
    steps = [
        "1. Predictor learns the target task",
        "2. Adversary tries to guess the protected attribute",
        "   from the predictor's outputs",
        "3. Predictor is penalized when the adversary succeeds",
        "4. Result: predictions carry no protected-group info",
    ]
    return steps

for s in adversarial_debiasing_idea():
    print(s)
print("\\nFairness is built INTO training, not bolted on after")`
                },
                {
                    title: "Ethical Considerations",
                    content: `Fairness is part of a broader responsibility to build ethical AI. Technical fixes alone are not enough.

<strong>Core Ethical Principles:</strong>
• <strong>Fairness:</strong> Avoid unjust discrimination
• <strong>Transparency:</strong> Explain how decisions are made
• <strong>Accountability:</strong> Someone is responsible for outcomes
• <strong>Privacy:</strong> Protect personal data
• <strong>Beneficence:</strong> Do good, avoid harm

<strong>Beyond Metrics:</strong>
• Fairness metrics are tools, not the whole answer
• Context and values matter
• Involve affected communities

<strong>Transparency and Explainability:</strong>
• People deserve to understand decisions affecting them
• Use interpretable models where stakes are high
• Provide meaningful explanations

<strong>Accountability:</strong>
• Clear ownership of model decisions
• Recourse for those harmed
• Human oversight for high-stakes decisions

<strong>Stakeholder Involvement:</strong>
• Engage domain experts and affected groups
• Diverse teams catch more problems

<strong>Legal and Regulatory:</strong>
• GDPR, anti-discrimination laws, AI regulations
• Compliance is a baseline, not the ceiling

<strong>Ongoing Responsibility:</strong>
Ethics is not a checkbox — it requires continuous attention throughout the system's life.`,
                    code: `# A responsible AI review checklist as structured data

ethics_checklist = {
    "Fairness": [
        "Measured fairness across protected groups?",
        "Tested for proxy variables?",
        "Applied mitigation where needed?",
    ],
    "Transparency": [
        "Can we explain individual decisions?",
        "Documented model limitations?",
    ],
    "Accountability": [
        "Clear owner for model outcomes?",
        "Recourse process for those harmed?",
        "Human oversight for high-stakes cases?",
    ],
    "Privacy": [
        "Personal data protected?",
        "Compliant with regulations (GDPR, etc.)?",
    ],
}

for principle, questions in ethics_checklist.items():
    print("[{}]".format(principle))
    for q in questions:
        print("   [ ]", q)
    print()
print("Ethical AI is a continuous practice, not a one-time check")`
                },
                {
                    title: "Case Studies",
                    content: `Learning from real-world fairness failures helps prevent repeating them.

<strong>Case 1: Biased Hiring Tool</strong>
• A resume-screening model favored male candidates
• Cause: trained on historical hires (mostly men)
• Lesson: historical data encodes historical bias
• Fix: audit for gender disparity, remove biased signals

<strong>Case 2: Facial Recognition Disparities</strong>
• Accuracy much lower for darker-skinned women
• Cause: unrepresentative training data
• Lesson: representation bias causes unequal performance
• Fix: diverse, balanced datasets; disaggregated evaluation

<strong>Case 3: Credit Scoring</strong>
• Models disadvantaged certain neighborhoods
• Cause: zip code acted as a proxy for race
• Lesson: proxy variables leak protected attributes
• Fix: detect and remove proxies; fairness constraints

<strong>Case 4: Healthcare Risk Algorithm</strong>
• Underestimated illness severity for one group
• Cause: used healthcare cost as a proxy for need
• Lesson: label choice can embed bias
• Fix: choose labels that truly reflect the goal

<strong>Common Threads:</strong>
• Biased data → biased models
• Proxies are sneaky
• Aggregate metrics hide subgroup harm
• Audit early, continuously, and per-group

<strong>Takeaway:</strong>
Most fairness failures were preventable with careful data review and disaggregated testing.`,
                    code: `# Lessons from case studies encoded as guardrails

def fairness_review(model_context):
    warnings = []

    # Lesson from hiring tool: check historical bias
    if model_context.get('trained_on_historical_decisions'):
        warnings.append("Historical data may encode past bias")

    # Lesson from facial recognition: check representation
    if model_context.get('min_group_representation', 1.0) < 0.2:
        warnings.append("Some groups underrepresented in data")

    # Lesson from credit scoring: check for proxies
    if model_context.get('has_proxy_features'):
        warnings.append("Features may proxy protected attributes")

    # Lesson from healthcare: check the label
    if model_context.get('label_is_proxy'):
        warnings.append("Label may not reflect the true objective")

    return warnings

context = {
    'trained_on_historical_decisions': True,
    'min_group_representation': 0.1,
    'has_proxy_features': True,
    'label_is_proxy': False,
}
print("Fairness review warnings:")
for w in fairness_review(context):
    print("  -", w)`
                },
                {
                    title: "Best Practices for Fair ML",
                    content: `A consolidated set of practices for building fair and responsible ML systems.

<strong>1. Start With the Problem:</strong>
• Ask if ML is appropriate at all
• Consider who is affected and how
• Define what fairness means in this context

<strong>2. Audit Your Data:</strong>
• Check representation across groups
• Look for historical bias in labels
• Identify potential proxy variables

<strong>3. Measure Fairness:</strong>
• Choose metrics that match the harm to prevent
• Evaluate disaggregated by subgroup
• Do not rely on aggregate metrics alone

<strong>4. Mitigate Thoughtfully:</strong>
• Apply pre-, in-, or post-processing as appropriate
• Understand the accuracy-fairness trade-off
• Re-measure after mitigation

<strong>5. Be Transparent:</strong>
• Document data, decisions, and limitations
• Provide explanations for decisions
• Use model cards and datasheets

<strong>6. Involve People:</strong>
• Diverse teams and affected communities
• Domain experts and ethicists

<strong>7. Monitor Continuously:</strong>
• Bias can emerge over time
• Audit in production, not just at build time

<strong>8. Keep Humans in the Loop:</strong>
• Human oversight for high-stakes decisions
• Provide recourse for those affected

<strong>Final Thought:</strong>
Fairness is an ongoing commitment. Responsible ML combines good engineering, careful measurement, and genuine care for human impact.`,
                    code: `# A fair ML workflow bringing the practices together

def fair_ml_workflow():
    workflow = [
        ("Frame", "Define fairness for this context; identify stakeholders"),
        ("Audit data", "Check representation, labels, and proxies"),
        ("Measure", "Evaluate fairness metrics, disaggregated by group"),
        ("Mitigate", "Apply pre/in/post-processing; mind trade-offs"),
        ("Validate", "Re-measure fairness after mitigation"),
        ("Document", "Model cards, limitations, decisions"),
        ("Deploy", "With human oversight for high-stakes use"),
        ("Monitor", "Audit continuously for emerging bias"),
    ]
    for step, description in workflow:
        print("  {:<12}: {}".format(step, description))

print("Fair ML lifecycle:\\n")
fair_ml_workflow()
print("\\nResponsible AI = good engineering + care for human impact")`
                }
            ]
        }
    ]
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadModules();
    setupNavigation();
    setupScrollAnimations();
});

// Load all modules into their respective grids
function loadModules() {
    loadModulesIntoGrid('ml-models-grid', courseData.mlModels);
    loadModulesIntoGrid('data-grid', courseData.data);
    loadModulesIntoGrid('advanced-ml-grid', courseData.advancedML);
    loadModulesIntoGrid('realworld-ml-grid', courseData.realWorldML);
}

// Load modules into a specific grid
function loadModulesIntoGrid(gridId, modules) {
    const grid = document.getElementById(gridId);
    
    modules.forEach(module => {
        const card = createModuleCard(module);
        grid.appendChild(card);
    });
}

// Create a module card element
function createModuleCard(module) {
    const card = document.createElement('div');
    card.className = `module-card ${module.isNew ? 'new' : ''} ${module.isLocked ? 'locked' : ''}`;
    card.onclick = () => openModuleModal(module);
    
    const lockIcon = module.isLocked ? '<div class="lock-icon">🔒</div>' : '';
    
    card.innerHTML = `
        ${lockIcon}
        <div class="module-number">${module.number}</div>
        <h4 class="module-title">${module.title}</h4>
        <p class="module-description">${module.description}</p>
        <div class="module-meta">
            <span class="meta-item"><i class="fas fa-book-open"></i> ${module.lessons}</span>
        </div>
    `;
    
    return card;
}

// Open module details in modal
function openModuleModal(module) {
    const modal = document.getElementById('moduleModal');
    const modalBody = document.getElementById('modal-body');
    
    // Check if module is locked
    if (module.isLocked) {
        modalBody.innerHTML = `
            <div class="modal-header" style="text-align: center;">
                <div class="lock-icon-large">🔒</div>
                <h2 class="modal-title">Module Locked</h2>
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
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        return;
    }
    
    // Check if module has detailed content
    if (module.detailedContent && module.detailedContent.length > 0) {
        // Show detailed content with expandable sections
        const contentSections = module.detailedContent.map((section, index) => `
            <div class="content-section">
                <div class="content-header" onclick="toggleContent(${index})">
                    <h3 class="content-title">
                        <span class="content-number">${index + 1}</span>
                        ${section.title}
                    </h3>
                    <span class="expand-icon" id="icon-${index}">▼</span>
                </div>
                <div class="content-body" id="content-${index}" style="display: none;">
                    <div class="content-text">${section.content.replace(/\n/g, '<br>')}</div>
                    ${section.code ? `
                        <div class="code-section">
                            <div class="code-header">
                                <span>💻 Code Example</span>
                                <button class="copy-btn" onclick="copyCode(${index}, event)">📋 Copy</button>
                            </div>
                            <pre><code id="code-${index}">${escapeHtml(section.code)}</code></pre>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
        modalBody.innerHTML = `
            <div class="modal-header">
                <div class="module-number">${module.number}</div>
                <h2 class="modal-title">${module.title}</h2>
                <p class="modal-description">${module.detailedDescription}</p>
                <div class="module-meta" style="justify-content: center; margin-top: 1rem;">
                    <span class="meta-item"><i class="fas fa-book-open"></i> ${module.lessons}</span>
                </div>
            </div>
            <div class="detailed-content">
                ${contentSections}
            </div>
            <div style="margin-top: 2rem; text-align: center;">
                <button class="btn btn-primary" disabled style="cursor: not-allowed; opacity: 0.6;">
                    📺 Coming Soon on YouTube
                </button>
                <button class="btn btn-secondary" onclick="closeModal()" style="margin-left: 1rem;">
                    Close
                </button>
            </div>
        `;
    } else {
        // Original simple view for modules without detailed content
        modalBody.innerHTML = `
            <div class="modal-header">
                <div class="module-number">${module.number}</div>
                <h2 class="modal-title">${module.title}</h2>
                <p class="modal-description">${module.detailedDescription}</p>
            </div>
            <div class="modal-stats">
                <div class="module-meta">
                    <span class="meta-item"><i class="fas fa-book-open"></i> ${module.lessons}</span>
                </div>
            </div>
            <div class="topics-section">
                <h3>What You'll Learn</h3>
                <ul class="topics-list">
                    ${module.topics.map(topic => `<li>${topic}</li>`).join('')}
                </ul>
            </div>
            <div style="margin-top: 2rem; text-align: center;">
                <button class="btn btn-primary" disabled style="cursor: not-allowed; opacity: 0.6;">
                    📺 Coming Soon on YouTube
                </button>
                <button class="btn btn-secondary" onclick="closeModal()" style="margin-left: 1rem;">
                    Close
                </button>
            </div>
        `;
    }
    
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

// Close modal
function closeModal() {
    const modal = document.getElementById('moduleModal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('moduleModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Setup navigation
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Scroll to section
            const targetId = this.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
}

// Smooth scroll to section
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const offsetTop = section.offsetTop - 80; // Account for fixed navbar
        window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
        });
    }
}

// Setup scroll animations
function setupScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all module cards
    document.querySelectorAll('.module-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(card);
    });
    
    // Observe category sections
    document.querySelectorAll('.category-section').forEach(section => {
        observer.observe(section);
    });
}

// Update active nav link on scroll
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (window.pageYOffset >= sectionTop - 100) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Add keyboard support for modal
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Search functionality (can be extended later)
function searchModules(query) {
    const allModules = [
        ...courseData.mlModels,
        ...courseData.data,
        ...courseData.advancedML,
        ...courseData.realWorldML
    ];
    
    return allModules.filter(module => 
        module.title.toLowerCase().includes(query.toLowerCase()) ||
        module.description.toLowerCase().includes(query.toLowerCase())
    );
}

// Console welcome message
console.log('%c🧠 ML Course Website', 'color: #4285f4; font-size: 20px; font-weight: bold;');
console.log('%cWelcome to the ML Course! Happy Learning! 🚀', 'color: #34a853; font-size: 14px;');

// Toggle content section
function toggleContent(index) {
    const content = document.getElementById(`content-${index}`);
    const icon = document.getElementById(`icon-${index}`);
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▲';
        icon.style.transform = 'rotate(180deg)';
    } else {
        content.style.display = 'none';
        icon.textContent = '▼';
        icon.style.transform = 'rotate(0deg)';
    }
}

// Copy code to clipboard
function copyCode(index, event) {
    const codeElement = document.getElementById(`code-${index}`);
    const text = codeElement.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        // Show success message
        const copyBtn = event.target;
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        copyBtn.style.background = '#34a853';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
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
