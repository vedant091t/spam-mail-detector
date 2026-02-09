from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_test, y_pred):
    """
    Evaluate the model performance using standard classification metrics.
    
    Args:
        y_test: True labels from test set
        y_pred: Predicted labels from the model
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Calculate evaluation metrics
    # Accuracy: Overall correctness
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision: Of all predicted spam, how many were actually spam?
    precision = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
    
    # Recall: Of all actual spam, how many did we correctly identify?
    recall = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)
    
    # F1-Score: Harmonic mean of precision and recall
    f1 = f1_score(y_test, y_pred, pos_label='spam', zero_division=0)
    
    # Print results
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nFormat: [[True Ham, False Spam]")
    print("         [False Ham, True Spam]]")
    print("="*60 + "\n")
    
# TODO: Add confusion matrix visualization using matplotlib/seaborn
