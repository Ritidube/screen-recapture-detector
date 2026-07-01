# Spot the Fake Photo — Approach Note


## What I did

I used classical computer vision techniques instead of training a deep neural network, since the dataset was small and the assignment allowed traditional methods as well. The system combines 8 different signals commonly used in recapture and screen-detection tasks: wavelet sub-band statistics, FFT-based moiré patterns, LBP texture, noise residuals, chromatic aberration near edges, sharpness and blur statistics, glare and highlight features, and color-space statistics. These are merged into a 194-dimensional feature vector for each image. The same feature extraction pipeline is used consistently in the notebook, prediction script, and Streamlit demo to avoid mismatch. For efficiency, texture and frequency features are computed on a larger 384 px image, while color and glare features are extracted from a 200 px thumbnail. The final model uses StandardScaler followed by SelectKBest and Logistic Regression. I also tested RBF SVM and Random Forest, but Logistic Regression gave the best cross-validation performance, so I selected it for the final system.

## Architecture
 
<!-- ![Architecture diagram](assets/architecture.jpeg) -->
## Architecture

<p align="center">
  <img src="assets/architecture.jpeg" alt="Architecture Diagram" width="600">
</p>

## Accuracy

The dataset used for experimentation contained 100 images: 50 real photos and 50 screen recaptures collected using a phone under different lighting conditions, angles, and screens. The model achieved 90% 5-fold cross-validation accuracy and 0.955 ROC-AUC. The confusion matrix was balanced, with 45 out of 50 real images and 45 out of 50 screen images classified correctly. A train accuracy of 93% was also observed, but that is not the reported performance because it was measured on data the model had already seen. The 90% cross-validation result is the more reliable metric. Since the dataset is small, the result may still vary slightly with more data.

## Latency & Cost

The average inference time was around 150–180 ms per image on Colab CPU in a single-threaded setup, including image loading. The median time was slightly lower depending on the run. Since the solution is based on lightweight numerical operations rather than a heavy deep learning model, it can run efficiently on a local system or mobile device without requiring a GPU. If deployed on a server, the compute cost would remain low because no model download or network call is needed.

## What I would improve

The main limitation of this project is the small dataset size. With more images from different screens, devices, and lighting conditions, the model would likely become more robust. I would also test the system on completely unseen screens and devices instead of relying only on cross-validation. In addition, gradient boosting methods such as XGBoost or LightGBM could be explored to capture feature interactions better than a linear model.