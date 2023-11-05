from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(labels, preds, metrics=['acc', 'precision', 'recall', 'f1']):
    '''
    Compute metrics from labels and preds.
    Args: 
        labels: list of labels
        preds: list of predictions
        metrics: list of metrics to compute
    Returns:
        metrics_dict: dict of metrics
    '''
    metrics_dict = {}
    # compute accuracy
    if 'acc' in metrics:
        metrics_dict['acc'] = accuracy_score(labels, preds)
    # compute precision
    if 'precision' in metrics:
        metrics_dict['precision'] = precision_score(labels, preds)
    # compute recall
    if 'recall' in metrics:
        metrics_dict['recall'] = recall_score(labels, preds)
    # compute f1
    if 'f1' in metrics:
        metrics_dict['f1'] = f1_score(labels, preds)
        
    return metrics_dict