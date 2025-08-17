Shoplifting Detection with Video Classification

This project focuses on classifying surveillance videos into two categories:

Shoplifter

Non-Shoplifter

The system uses deep learning techniques with a pretrained video classification model, applying transfer learning & fine-tuning to adapt it to the shoplifting detection task.

🔹 Key Features

Preprocessing of video data (frame sampling & resizing).

Data split into training, validation, and test sets.

Model trained using transformer-based video classification architecture.

Fine-tuning applied to improve performance on custom dataset.

Training loop with loss monitoring and validation accuracy tracking.

Evaluation on test set to measure final performance.

🔹 Technologies Used

Python

PyTorch

HuggingFace Transformers

Decord (for video loading)

TQDM (for training progress visualization)

🔹 Project Structure
├── data/                # Train, validation, test video datasets  
├── dataset.py           # Custom PyTorch dataset for videos  
├── train.py             # Training & evaluation loop  
├── model.py             # Model loading & fine-tuning setup  
├── requirements.txt     # Python dependencies  
└── README.md            # Project description


🔹 Dataset

The dataset contains surveillance videos labeled as shoplifter or non-shoplifter.

Videos are split into training, validation, and test sets.

Each video is preprocessed to a fixed number of frames for model input.

🔹 Training

The model is fine-tuned on the shoplifting dataset.

Training tracks loss and validation accuracy per epoch.

Optimizer: AdamW, Learning rate scheduler: Linear.

Example command to train:

python train.py --train_csv data/train.csv --val_csv data/val.csv --batch_size 4 --epochs 5

🔹 Evaluation

After training, the model is evaluated on the test set.

Outputs metrics:

Accuracy

Confusion matrix

Precision / Recall / F1-score (optional)

🔹 Goal

Build an AI-powered surveillance system capable of detecting shoplifting incidents automatically from video footage, supporting security monitoring and theft prevention in retail environments.
