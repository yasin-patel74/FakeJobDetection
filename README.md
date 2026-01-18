# FakeJobDetection
"Fake Job Posting Detection using Deep Learning"
# FakeJobDetection

This project detects **fake job postings** using a deep learning model (Bi-LSTM + Embedding).

---

## Project Files

| File | Description | Download Link |
|------|-------------|---------------|
| `FakeJobDetection.ipynb` | Main Jupyter notebook with preprocessing, training, and testing code | Directly in repo |
| `fake_job_model.keras` | Pretrained deep learning model (Bi-LSTM) | [Google Drive Link](https://drive.google.com/uc?export=download&id=1TJQEu0enx3L0rOtJZxQ8YI1WtvUj2BiX) |
| `tokenizer.pkl` | Tokenizer used to preprocess text before prediction | [Google Drive Link](https://drive.google.com/uc?export=download&id=1LfwePbWfyb8c7WLgoDgFV_EFpWY9T-64) |

---

## How to Use

1. Download the notebook (`FakeJobDetection.ipynb`) from GitHub.  
2. Download `fake_job_model.keras` and `tokenizer.pkl` from Google Drive.  
3. In the notebook, update the paths to these files if needed.  
4. Run the notebook cells sequentially:  
   - Load dataset (optional, for testing)  
   - Load model & tokenizer  
   - Test new job postings for fake/real prediction  

---

## Example Usage

```python
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model("path/to/fake_job_model.keras")

# Load tokenizer
with open("path/to/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Sample job posting
job_post = "Urgent hiring for online data entry work. Registration fee required."
seq = tokenizer.texts_to_sequences([job_post])
# pad sequence if necessary
# prediction = model.predict(padded_seq)
