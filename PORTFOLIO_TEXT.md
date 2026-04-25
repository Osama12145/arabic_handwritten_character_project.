# Portfolio Text

## CV Version

Built an Arabic handwritten character recognition and retrieval system using deep learning. Compared a CNN trained from scratch with transfer learning using EfficientNet-B1, then improved similarity-based search with metric learning using Triplet Loss. Achieved `70.3%` test accuracy and `65.4%` macro F1 for classification, and improved retrieval from `55.4%` to `63.4%` Top-1 and from `85.1%` to `91.1%` Top-5 after metric learning.

## Short CV Bullet

Developed an Arabic handwritten character recognition and retrieval project using PyTorch, transfer learning, and metric learning, achieving `70.3%` test accuracy and `91.1%` Top-5 retrieval accuracy.

## LinkedIn Project Description

Built a complete deep learning pipeline for Arabic handwritten character recognition and similarity-based image retrieval. I compared a CNN baseline against transfer learning with EfficientNet-B1, then used Triplet Loss metric learning to improve the embedding space for retrieval. The final system achieved `70.3%` test accuracy, `65.4%` macro F1, and improved retrieval performance to `63.4%` Top-1 and `91.1%` Top-5. I also added t-SNE visualization, Grad-CAM explainability, and a Gradio demo for interactive testing.

## GitHub Repo Description

Arabic handwritten character recognition and similarity search using PyTorch, EfficientNet-B1 transfer learning, and metric learning.

## One-Paragraph Project Summary

This project focuses on Arabic handwritten character recognition and retrieval using deep learning. I started with a CNN baseline trained from scratch, then fine-tuned a pretrained EfficientNet-B1 model for classification. After building a retrieval pipeline based on cosine similarity over learned embeddings, I introduced metric learning with Triplet Loss to improve retrieval quality. The final system combined classification, image retrieval, embedding visualization, Grad-CAM explainability, and a Gradio demo in one notebook-based workflow.

## Interview Explanation

This project was designed to go beyond standard image classification. Instead of only predicting the character label, I also built a retrieval system that returns the most visually similar handwritten samples. I first compared a custom CNN against transfer learning and found that EfficientNet-B1 gave much stronger results. Then I trained an embedding model with Triplet Loss, which improved retrieval quality over the base classifier embeddings. To make the project more complete, I added t-SNE to visualize the embedding space, Grad-CAM to interpret predictions, and a Gradio demo for interactive use.

