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
    ],

    // ==========================================================
    // AI-901: Microsoft Azure AI Fundamentals
    // ==========================================================
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
        }
    ],

    // ==========================================================
    // Salesforce Certified Agentforce Specialist (AI-201)
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
            detailedDescription: "The largest AI-201 domain covers prompt engineering with Prompt Builder: creating reusable prompt templates, grounding them in Salesforce data, choosing the right template type, and testing prompts for quality and safety.",
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
            detailedDescription: "The final AI-201 domain covers trust and operations: how the Einstein Trust Layer protects data and enforces responsible AI, how to test agents before launch, and how to monitor and maintain them in production.",
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
        }
    ]
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadModules();
    setupNavigation();
    setupScrollAnimations();
    setupTabs();
    setupCertFilter();
    setupDropdown();
    setupProviderMarquee();
    setupPathFinder();
});

// Duplicate provider logos so the marquee scrolls in a seamless, gapless loop
function setupProviderMarquee() {
    const track = document.getElementById('providersTrack');
    if (!track) return;
    const marquee = track.parentElement;

    // Remember the original logo set once so we can rebuild on resize
    if (!track._originals) {
        track._originals = Array.from(track.children).map(node => node.cloneNode(true));
    }

    const build = () => {
        const originals = track._originals;
        const speed = 70; // pixels per second

        // Reset to a single clean set
        track.innerHTML = '';
        originals.forEach(node => track.appendChild(node.cloneNode(true)));

        const oneSetWidth = track.scrollWidth;
        const containerWidth = marquee.offsetWidth || window.innerWidth;

        // Repeat the set until one half of the track comfortably exceeds the viewport,
        // keeping an even number of sets so translateX(-50%) loops seamlessly.
        let sets = 1;
        let guard = 0;
        while (((oneSetWidth * sets) / 2 < containerWidth + oneSetWidth) && guard < 20) {
            originals.forEach(node => {
                const clone = node.cloneNode(true);
                clone.setAttribute('aria-hidden', 'true');
                track.appendChild(clone);
            });
            sets++;
            guard++;
        }
        if (sets % 2 !== 0) {
            originals.forEach(node => {
                const clone = node.cloneNode(true);
                clone.setAttribute('aria-hidden', 'true');
                track.appendChild(clone);
            });
            sets++;
        }

        // Keep the scroll speed constant regardless of how many copies exist
        const halfWidth = track.scrollWidth / 2;
        track.style.animationDuration = (halfWidth / speed) + 's';
    };

    build();

    // Rebuild on resize so the loop stays gapless across screen sizes
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(build, 200);
    });
}

// Filter certification sections by cloud provider
function applyProviderFilter(provider) {
    document.querySelectorAll('.filter-btn').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.provider === provider));

    document.querySelectorAll('section[data-provider]').forEach(section => {
        const show = provider === 'all' || section.dataset.provider === provider;
        section.style.display = show ? '' : 'none';
    });
}

function setupCertFilter() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => applyProviderFilter(btn.dataset.provider));
    });

    // Keep provider nav links in sync with the filter
    const navProviderMap = {
        'certifications': 'azure',
        'aws-certifications': 'aws',
        'gcp-certifications': 'gcp',
    };
    document.querySelectorAll('.nav-dropdown-menu .nav-link').forEach(link => {
        const targetId = link.getAttribute('href').substring(1);
        if (navProviderMap[targetId]) {
            link.addEventListener('click', () => applyProviderFilter(navProviderMap[targetId]));
        }
    });

    // The "Certifications" toggle shows all providers
    const toggle = document.querySelector('.nav-dropdown-toggle');
    if (toggle) toggle.addEventListener('click', () => applyProviderFilter('all'));
}

// Certifications nav dropdown (hover on desktop, tap on mobile)
function setupDropdown() {
    const dropdown = document.querySelector('.nav-dropdown');
    if (!dropdown) return;
    const toggle = dropdown.querySelector('.nav-dropdown-toggle');

    toggle.addEventListener('click', function(e) {
        e.preventDefault();
        if (window.innerWidth <= 768) {
            dropdown.classList.toggle('open');
        } else {
            scrollToSection('certifications');
            dropdown.classList.remove('open');
        }
    });

    // Close the menu after picking a provider
    dropdown.querySelectorAll('.nav-dropdown-menu .nav-link').forEach(link => {
        link.addEventListener('click', () => dropdown.classList.remove('open'));
    });

    // Close when clicking outside
    document.addEventListener('click', function(e) {
        if (!dropdown.contains(e.target)) dropdown.classList.remove('open');
    });
}

// Setup tab navigation for course/certification sections
function setupTabs() {
    document.querySelectorAll('.section-tabs').forEach(tabBar => {
        const section = tabBar.closest('section');
        const tabs = tabBar.querySelectorAll('.section-tab');

        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                const targetId = this.dataset.target;

                tabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');

                section.querySelectorAll('.tab-panel').forEach(panel => {
                    const show = panel.id === targetId;
                    panel.classList.toggle('active', show);
                    if (show) {
                        // Ensure cards in the newly shown panel are visible
                        panel.querySelectorAll('.module-card').forEach(card => {
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        });
                    }
                });
            });
        });
    });
}

// Load all modules into their respective grids
function loadModules() {
    loadModulesIntoGrid('ml-models-grid', courseData.mlModels);
    loadModulesIntoGrid('data-grid', courseData.data);
    loadModulesIntoGrid('advanced-ml-grid', courseData.advancedML);
    loadModulesIntoGrid('realworld-ml-grid', courseData.realWorldML);
    loadModulesIntoGrid('ai-fundamentals-grid', courseData.aiFundamentals);
    loadModulesIntoGrid('ai-apps-agents-grid', courseData.aiAppsAgents);
    loadModulesIntoGrid('azure-data-scientist-grid', courseData.azureDataScientist);
    loadModulesIntoGrid('azure-mlops-grid', courseData.azureMlOps);
    loadModulesIntoGrid('azure-ai-clouddev-grid', courseData.azureAiCloudDev);
    loadModulesIntoGrid('aws-ai-practitioner-grid', courseData.awsAiPractitioner);
    loadModulesIntoGrid('aws-ml-engineer-grid', courseData.awsMlEngineer);
    loadModulesIntoGrid('aws-ml-specialty-grid', courseData.awsMlSpecialty);
    loadModulesIntoGrid('aws-genai-developer-grid', courseData.awsGenAiDeveloper);
    loadModulesIntoGrid('gcp-genai-leader-grid', courseData.gcpGenAiLeader);
    loadModulesIntoGrid('gcp-mle-grid', courseData.gcpMlEngineer);
    loadModulesIntoGrid('salesforce-agentforce-grid', courseData.salesforceAgentforce);
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
                <button class="btn btn-primary" onclick="closeModal()">
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
                <button class="btn btn-primary" onclick="closeModal()">
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

// Navigate to a specific certification: show its provider, open its tab, and scroll to it
function goToCert(sectionId, tabTarget, provider) {
    if (provider) applyProviderFilter(provider);
    if (tabTarget) {
        const tab = document.querySelector('.section-tab[data-target="' + tabTarget + '"]');
        if (tab) tab.click();
    }
    // Let layout settle (filter may unhide the section) before scrolling
    setTimeout(function () { scrollToSection(sectionId); }, 60);
}

// "Find Your Path" recommender: pick ecosystem + profile, get a recommended certification
function setupPathFinder() {
    const ecoGroup = document.getElementById('path-eco');
    const levelGroup = document.getElementById('path-level');
    const result = document.getElementById('path-result');
    if (!ecoGroup || !levelGroup || !result) return;

    const state = { eco: null, level: null };

    // Recommendation map keyed by "ecosystem|profile"
    const recos = {
        'azure|beginner':       { name: 'AI-901 · Azure AI Fundamentals', why: 'The best starting point on Azure — no prior AI experience needed.', section: 'certifications', tab: 'panel-ai-901', provider: 'azure' },
        'azure|developer':      { name: 'AI-103 · Azure AI Engineer', why: 'Build AI apps and agents on Azure. Do AI-901 first if you are new to AI.', section: 'certifications', tab: 'panel-ai-103', provider: 'azure' },
        'azure|datascientist':  { name: 'DP-100 · Azure Data Scientist', why: 'Train, tune, and deploy ML models with Azure Machine Learning.', section: 'certifications', tab: 'panel-dp-100', provider: 'azure' },
        'aws|beginner':         { name: 'AIF-C01 · AWS AI Practitioner', why: 'The foundational AWS AI cert — start here before the associate exams.', section: 'aws-certifications', tab: 'panel-aif', provider: 'aws' },
        'aws|developer':        { name: 'MLA-C01 · AWS ML Engineer', why: 'Operationalize ML on AWS. Try the Generative AI Developer cert next for app-building.', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'aws|datascientist':    { name: 'MLA-C01 · AWS ML Engineer', why: 'The current associate ML path on AWS (the MLS-C01 Specialty retired in 2026).', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'gcp|beginner':         { name: 'Generative AI Leader', why: 'A foundational, business-friendly intro to generative AI on Google Cloud.', section: 'gcp-certifications', tab: 'panel-genai-leader', provider: 'gcp' },
        'gcp|developer':        { name: 'Professional ML Engineer', why: 'Design and productionize ML on Google Cloud with Vertex AI.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' },
        'gcp|datascientist':    { name: 'Professional ML Engineer', why: 'The core ML certification on Google Cloud for data-focused roles.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' },
        'salesforce|beginner':      { name: 'Agentforce Specialist', why: 'Learn to build and deploy Salesforce Agentforce AI agents.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'salesforce|developer':     { name: 'Agentforce Specialist', why: 'Build agents, prompts, and actions on the Salesforce Platform.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'salesforce|datascientist': { name: 'Agentforce Specialist', why: 'Ground agents in Data Cloud and design effective prompts.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'unsure|beginner':      { name: 'AWS AI Practitioner or Azure AI Fundamentals', why: 'Start with a foundational cert. AWS has the biggest market; Azure is great if your workplace uses Microsoft.', section: 'aws-certifications', tab: 'panel-aif', provider: 'aws' },
        'unsure|developer':     { name: 'AWS ML Engineer (MLA-C01)', why: 'AWS offers the broadest opportunities for developers. Prefer Azure or GCP if your company already uses them.', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'unsure|datascientist': { name: 'Google Cloud Professional ML Engineer', why: 'Google Cloud shines for data science. Azure DP-100 is a strong alternative in Microsoft shops.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' }
    };

    function selectIn(group, attr, value) {
        group.querySelectorAll('.path-opt').forEach(function (btn) {
            btn.classList.toggle('active', btn.dataset[attr] === value);
        });
    }

    function render() {
        if (!state.eco || !state.level) return;
        const reco = recos[state.eco + '|' + state.level];
        if (!reco) return;
        result.hidden = false;
        result.innerHTML =
            '<div class="path-result-inner">' +
                '<span class="path-result-label">Recommended for you</span>' +
                '<h4 class="path-result-name">' + reco.name + '</h4>' +
                '<p class="path-result-why">' + reco.why + '</p>' +
                '<button type="button" class="btn btn-primary path-result-btn">Go to this certification &rarr;</button>' +
            '</div>';
        result.querySelector('.path-result-btn').addEventListener('click', function () {
            goToCert(reco.section, reco.tab, reco.provider);
        });
    }

    ecoGroup.querySelectorAll('.path-opt').forEach(function (btn) {
        btn.addEventListener('click', function () {
            state.eco = btn.dataset.eco;
            selectIn(ecoGroup, 'eco', state.eco);
            render();
        });
    });
    levelGroup.querySelectorAll('.path-opt').forEach(function (btn) {
        btn.addEventListener('click', function () {
            state.level = btn.dataset.level;
            selectIn(levelGroup, 'level', state.level);
            render();
        });
    });
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
