# ML Model Evaluation Report

## Model Information
- **Model**: Segformer MIT-B0
- **Architecture**: segmentation-models-pytorch Segformer
- **Pretrained Weights**: imagenet
- **Num Classes**: 10
- **Input Size**: 256x256
- **Model File**: segformer.pth (14.25 MB)

## Training Status: ✅ MODEL IS ALREADY TRAINED

## Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.5107 |
| **Precision** | 0.5103 |
| **Recall** | 0.5107 |
| **F1 Score** | 0.4917 |
| **IoU Score** | 0.0154 |

## Class Labels
1. Trees
2. Lush Bushes
3. Dry Grass
4. Dry Bushes
5. Ground Clutter
6. Flowers
7. Logs
8. Rocks
9. Landscape
10. Sky

## Evaluation Files
- Confusion Matrix: `backend/ml/model_evaluation/confusion_matrix.png`
- Accuracy Graph: `backend/ml/model_evaluation/accuracy_graph.png`

## Notes
- The model shows moderate accuracy (51.07%) on the validation set
- The IoU score is relatively low (1.54%), indicating the model may need more training
- Consider retraining with more epochs or tuning hyperparameters for improved performance

## How to Retrain
If you want to retrain the model:
```bash
python backend/ml/train.py
```

This will:
1. Load training data from `backend/ml/datasets/dataset/train/`
2. Load validation data from `backend/ml/datasets/dataset/val/`
3. Train for 10 epochs (configurable in `backend/config.yaml`)
4. Save trained weights to `backend/ml/weights/segformer.pth`
