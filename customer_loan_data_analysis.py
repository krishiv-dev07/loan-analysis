import numpy as np

def compute_expected_loss(borrower_data: dict, model, scaler) -> dict:
    """
    borrower_data: dict with keys matching feature names
    model: trained logistic regression model
    scaler: fitted StandardScaler
    Returns a dictionary with PD and Expected Loss
    """
    feature_order = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding',
                     'income', 'years_employed', 'fico_score']
    
    # Convert input into the proper format and scale
    input_features = np.array([[borrower_data[feat] for feat in feature_order]])
    input_scaled = scaler.transform(input_features)
    
    # Predict Probability of Default (PD)
    pd = model.predict_proba(input_scaled)[0, 1]
    
    # Compute Expected Loss
    recovery_rate = 0.10
    loan_amount = borrower_data['loan_amt_outstanding']
    expected_loss = pd * (1 - recovery_rate) * loan_amount
    
    return {
        "probability_of_default": pd,
        "expected_loss": expected_loss
    }

# Example borrower
example_borrower = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 3000,
    'income': 40000,
    'years_employed': 3,
    'fico_score': 620
}

# Usage
result = compute_expected_loss(example_borrower, lr_model, scaler)
print(result)
