# Arabic Handwritten Character Recognition and Retrieval

End-to-end deep learning project for Arabic handwritten character classification and similarity-based retrieval using PyTorch, EfficientNet-B1 transfer learning, and metric learning.

## Highlights

- Uses the official `AHCD` dataset with the standard train/test split.
- Compares a custom `CNN from scratch` against `EfficientNet-B1` transfer learning.
- Extracts embeddings for nearest-neighbor retrieval with cosine similarity.
- Adds a metric-learning stage with `Triplet Loss`.
- Includes `t-SNE`, `Grad-CAM`, and a `Gradio` demo in the notebook.

## Dataset

- Dataset: `AHCD` via `mloey1/ahcd1`
- Classes: `28` Arabic characters
- Train: `12,096`
- Validation: `1,344`
- Test: `3,360`

The notebook keeps the official AHCD test split and creates validation from the training split only.

## Final Results

### Classification

- `Transfer Learning (EfficientNet-B1)`
  - Test Accuracy: `98.72%`
  - Test Macro F1: `98.72%`

- `CNN From Scratch`
  - Test Accuracy: `39.37%`
  - Test Macro F1: `35.61%`

### Retrieval

- `Base Classifier Embeddings`
  - Top-1 Retrieval Accuracy: `98.33%`
  - Top-5 Retrieval Accuracy: `99.17%`

- `Metric Learning Embeddings`
  - Top-1 Retrieval Accuracy: `98.15%`
  - Top-5 Retrieval Accuracy: `99.05%`

In this AHCD run, transfer learning clearly dominated the scratch baseline, while the extra metric-learning stage stayed competitive but did not outperform the base embedding setup.

## Notebook

Main notebook:

- [arabic_handwritten_character_project_all_in_one.ipynb](./arabic_handwritten_character_project_all_in_one.ipynb)

## Visual Assets

- [assets/transfer_training_curves.png](./assets/transfer_training_curves.png)
- [assets/retrieval_comparison_bar.png](./assets/retrieval_comparison_bar.png)
- [assets/retrieval_before_after.png](./assets/retrieval_before_after.png)
- [assets/tsne_before_after.png](./assets/tsne_before_after.png)
- [assets/gradcam_example.png](./assets/gradcam_example.png)
- [assets/metric_learning_curves_and_example.png](./assets/metric_learning_curves_and_example.png)

## Tech Stack

- Python
- PyTorch
- Torchvision
- scikit-learn
- Matplotlib
- PIL
- Gradio
- KaggleHub

## Notes

- The notebook already contains executed outputs and experiment results.
- The demo works best with dataset-style character images.
- The retrieval stage is useful for visual similarity search, but for this run the strongest headline result is the classifier itself.

## Author

Osama Ali Naji  
GitHub: [Osama12145](https://github.com/Osama12145)
