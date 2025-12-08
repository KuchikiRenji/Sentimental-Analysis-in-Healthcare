import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_curve, auc
)
import numpy as np
import os

# Function to create output directory if it doesn't exist
def create_output_directory(directory='output'):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save confusion matrix as an image
def plot_confusion_matrix(y_test, y_pred, directory='output'):
    create_output_directory(directory)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique classes and sort them to match labels
    classes = np.unique(np.concatenate([y_test, y_pred]))
    classes_sorted = sorted(classes)
    class_names = []
    for cls in classes_sorted:
        if cls == -1:
            class_names.append('Negative')
        elif cls == 0:
            class_names.append('Neutral')
        elif cls == 1:
            class_names.append('Positive')
        else:
            class_names.append(f'Class {cls}')
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(directory, 'confusion_matrix.png'))
    plt.close()

# Function to plot and save the ROC curve (multi-class)
def plot_roc_curve(y_test, y_proba, model_classes, directory='output'):
    create_output_directory(directory)
    # For multi-class, use one-vs-rest approach
    # model_classes is the order of classes as used by the model (from model.classes_)
    
    # Map class labels to their indices in model_classes
    class_to_index = {cls: idx for idx, cls in enumerate(model_classes)}
    
    # Calculate ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Map class labels to names
    class_name_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    for class_label in model_classes:
        # Convert to binary: current class vs rest
        y_test_binary = (y_test == class_label).astype(int)
        proba_idx = class_to_index[class_label]
        y_proba_binary = y_proba[:, proba_idx]
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binary, y_proba_binary)
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
    
    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for class_label in model_classes:
        class_name = class_name_map.get(class_label, f'Class {class_label}')
        color_idx = list(model_classes).index(class_label) % len(colors)
        plt.plot(fpr[class_label], tpr[class_label], color=colors[color_idx], lw=2,
                label=f'ROC curve for {class_name} (area = {roc_auc[class_label]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, 'roc_curve.png'))
    plt.close()
    
    # Calculate macro-average ROC-AUC
    macro_auc = np.mean(list(roc_auc.values()))
    return macro_auc

# Function to plot and save precision-recall curve (multi-class)
def plot_precision_recall_curve(y_test, y_proba, model_classes, directory='output'):
    create_output_directory(directory)
    # For multi-class, use one-vs-rest approach
    # model_classes is the order of classes as used by the model (from model.classes_)
    
    # Map class labels to their indices in model_classes
    class_to_index = {cls: idx for idx, cls in enumerate(model_classes)}
    
    # Calculate Precision-Recall for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    colors = ['blue', 'red', 'green']
    class_name_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    for class_label in model_classes:
        # Convert to binary: current class vs rest
        y_test_binary = (y_test == class_label).astype(int)
        proba_idx = class_to_index[class_label]
        y_proba_binary = y_proba[:, proba_idx]
        precision[class_label], recall[class_label], _ = precision_recall_curve(y_test_binary, y_proba_binary)
        pr_auc[class_label] = auc(recall[class_label], precision[class_label])
    
    # Plot Precision-Recall curves for each class
    plt.figure(figsize=(10, 8))
    for class_label in model_classes:
        class_name = class_name_map.get(class_label, f'Class {class_label}')
        color_idx = list(model_classes).index(class_label) % len(colors)
        plt.plot(recall[class_label], precision[class_label], color=colors[color_idx], lw=2,
                label=f'PR curve for {class_name} (area = {pr_auc[class_label]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, 'precision_recall_curve.png'))
    plt.close()
    
    # Calculate macro-average PR-AUC
    macro_pr_auc = np.mean(list(pr_auc.values()))
    return macro_pr_auc

# Function to plot and save accuracy graph
def plot_accuracy(y_test, y_pred, directory='output'):
    create_output_directory(directory)
    accuracy = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='green')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.savefig(os.path.join(directory, 'accuracy_graph.png'))
    plt.close()

# Function to save classification report and accuracy to a text file
def save_evaluation_to_text(y_test, y_pred, y_proba, model_classes, macro_roc_auc, macro_pr_auc, directory='output'):
    create_output_directory(directory)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Map class labels to their indices in model_classes
    class_to_index = {cls: idx for idx, cls in enumerate(model_classes)}
    class_name_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    # Calculate per-class ROC-AUC
    roc_auc_per_class = {}
    pr_auc_per_class = {}
    
    for class_label in model_classes:
        class_name = class_name_map.get(class_label, f'Class {class_label}')
        y_test_binary = (y_test == class_label).astype(int)
        proba_idx = class_to_index[class_label]
        y_proba_binary = y_proba[:, proba_idx]
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba_binary)
        roc_auc_per_class[class_name] = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test_binary, y_proba_binary)
        pr_auc_per_class[class_name] = auc(recall, precision)
    
    with open(os.path.join(directory, 'evaluation_report.txt'), 'w') as file:
        # Write Accuracy
        file.write(f"Accuracy on test data: {accuracy:.4f}\n\n")
        
        # Write Classification Report
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\n")

        # Write Confusion Matrix
        file.write("Confusion Matrix:\n")
        file.write("Labels: [-1: Negative, 0: Neutral, 1: Positive]\n")
        file.write(np.array2string(cm))
        file.write("\n\n")

        # Write ROC-AUC Information
        file.write("ROC-AUC Scores (per class):\n")
        for class_name, auc_score in roc_auc_per_class.items():
            file.write(f"  {class_name}: {auc_score:.4f}\n")
        file.write(f"Macro-average ROC-AUC: {macro_roc_auc:.4f}\n\n")

        # Write Precision-Recall Information
        file.write("Precision-Recall AUC Scores (per class):\n")
        for class_name, pr_score in pr_auc_per_class.items():
            file.write(f"  {class_name}: {pr_score:.4f}\n")
        file.write(f"Macro-average PR-AUC: {macro_pr_auc:.4f}\n")

# Function to evaluate the model's performance and save outputs
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Probability estimates for all classes
    
    # Get the order of classes as used by the model
    model_classes = model.classes_

    # Evaluate model accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {accuracy:.4f}")
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred)

    # Save ROC curve plot (returns macro-averaged AUC)
    macro_roc_auc = plot_roc_curve(y_test, y_proba, model_classes)

    # Save Precision-Recall curve plot (returns macro-averaged PR-AUC)
    macro_pr_auc = plot_precision_recall_curve(y_test, y_proba, model_classes)

    # Save accuracy graph
    plot_accuracy(y_test, y_pred)

    # Save evaluation report to a text file
    save_evaluation_to_text(y_test, y_pred, y_proba, model_classes, macro_roc_auc, macro_pr_auc)

# Example usage (assuming you have a model, X_test, and y_test ready)
# evaluate_model(your_model, X_test, y_test)
