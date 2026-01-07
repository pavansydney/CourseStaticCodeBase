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
            isLocked: true,
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
            isLocked: true,
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
            detailedDescription: "Logistic regression extends linear regression to classification problems. Learn how to predict probabilities, understand the sigmoid function, and work with binary and multi-class classification problems. This module covers everything from theory to practical implementation."
        },
        {
            number: "Module 3",
            title: "Classification",
            description: "An introduction to binary classification models, covering thresholding, confusion matrices, and metrics like accuracy, precision, recall, and AUC.",
            duration: "55 min",
            lessons: "10 lessons",
            isNew: false,
            isLocked: true,
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
            detailedDescription: "Master the art of classification by understanding key metrics and evaluation techniques. Learn when to use accuracy, precision, or recall, how to interpret confusion matrices, and work with ROC curves to evaluate your classification models effectively."
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
            isLocked: true,
            topics: [
                "Feature Scaling and Normalization",
                "Handling Missing Values",
                "Outlier Detection",
                "Data Distribution Analysis",
                "Feature Engineering",
                "Binning and Discretization",
                "Best Practices for Numerical Features"
            ],
            detailedDescription: "Numerical data is the backbone of most ML models. This module teaches you how to properly prepare, transform, and engineer numerical features to improve model performance. Learn about normalization, standardization, and advanced preprocessing techniques."
        },
        {
            number: "Module 5",
            title: "Working with Categorical Data",
            description: "Learn the fundamentals of working with categorical data: one-hot encoding, feature hashing, mean encoding, and feature crosses.",
            duration: "45 min",
            lessons: "8 lessons",
            isNew: false,
            isLocked: true,
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
            detailedDescription: "Categorical data requires special handling in machine learning. This comprehensive module covers various encoding techniques, from basic one-hot encoding to advanced methods like feature hashing and mean encoding. Learn how to handle high-cardinality features and create meaningful feature crosses."
        },
        {
            number: "Module 6",
            title: "Datasets, Generalization, and Overfitting",
            description: "An introduction to the characteristics of machine learning datasets, and how to prepare your data to ensure high-quality results.",
            duration: "50 min",
            lessons: "9 lessons",
            isNew: false,
            isLocked: true,
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
            detailedDescription: "Learn the critical concepts of overfitting and generalization. Understand how to split your data properly, use cross-validation, and apply regularization techniques to ensure your models perform well on unseen data. This module is essential for building robust ML systems."
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
            isLocked: true,
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
            detailedDescription: "Dive into the world of neural networks! This comprehensive module covers everything from basic perceptrons to deep neural networks. Learn how neurons work, understand activation functions, and master the backpropagation algorithm. Build practical neural networks from scratch."
        },
        {
            number: "Module 8",
            title: "Embeddings",
            description: "Learn how embeddings allow you to do machine learning on large feature vectors and capture semantic relationships.",
            duration: "45 min",
            lessons: "8 lessons",
            isNew: false,
            isLocked: true,
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
            detailedDescription: "Embeddings are powerful representations that capture semantic meaning in a dense vector space. Learn how to work with word embeddings, create your own embeddings, and leverage pre-trained embeddings for transfer learning. Essential for NLP and recommendation systems."
        },
        {
            number: "Module 9",
            title: "Large Language Models",
            description: "An introduction to large language models, from tokens to Transformers. Learn the basics of how LLMs learn to predict text output.",
            duration: "70 min",
            lessons: "14 lessons",
            isNew: true,
            isLocked: true,
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
            detailedDescription: "Explore the cutting-edge world of Large Language Models! Understand how transformers revolutionized NLP, learn about attention mechanisms, and discover how models like GPT and BERT work. This new module covers the latest advances in AI and practical applications of LLMs."
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
            isLocked: true,
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
            detailedDescription: "Bridge the gap between development and production. Learn how to deploy ML models at scale, monitor their performance, and maintain them in production environments. Covers infrastructure, deployment strategies, and best practices for production ML systems."
        },
        {
            number: "Module 11",
            title: "AutoML",
            description: "Learn principles and best practices for using automated machine learning to streamline model development.",
            duration: "40 min",
            lessons: "7 lessons",
            isNew: true,
            isLocked: true,
            topics: [
                "Introduction to AutoML",
                "Automated Feature Engineering",
                "Neural Architecture Search",
                "Hyperparameter Optimization",
                "Model Selection",
                "AutoML Tools and Platforms",
                "When to Use AutoML"
            ],
            detailedDescription: "Discover how AutoML can accelerate your machine learning workflow. Learn about automated feature engineering, hyperparameter tuning, and neural architecture search. Understand when AutoML is appropriate and how to integrate it into your development process."
        },
        {
            number: "Module 12",
            title: "ML Fairness",
            description: "Learn principles and best practices for auditing ML models for fairness, including strategies for identifying and mitigating biases.",
            duration: "50 min",
            lessons: "9 lessons",
            isNew: false,
            isLocked: true,
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
            detailedDescription: "Build responsible AI systems by understanding fairness and bias in machine learning. Learn how to identify, measure, and mitigate bias in your models. Essential knowledge for creating ethical and equitable ML systems that benefit everyone."
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
                                <button class="copy-btn" onclick="copyCode(${index})">📋 Copy</button>
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
function copyCode(index) {
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
