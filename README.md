# PPL_Counter

Starting with a simple demo project (for now) for people counting and density map generation using the Mall dataset.

---

## Project Structure

PPL_Counter/
│ .gitignore
│ requirements.txt
│ main.py
│
├── datasets/
│ └── mall_dataset/ # pre-downloaded dataset
│
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── model.py
│ └── train.py

yaml
Copy code

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/035uros/PPL_Counter.git
cd PPL_Counter
2. Set up Python virtual environment
On Windows PowerShell:

powershell
Copy code
# Create virtual environment
py -m venv venv

# Allow script execution temporarily (needed for PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\venv\Scripts\activate
On Command Prompt (CMD):

cmd
Copy code
py -m venv venv
venv\Scripts\activate.bat
Once activated, you should see (venv) in front of your terminal prompt.

3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Verify installation
bash
Copy code
python main.py
You should see something like:

sql
Copy code
✅ PyTorch check: 2.x.x+cpu
✅ NumPy check: 1.x.x
5. Dataset
Place the Mall dataset inside datasets/mall_dataset/.

The folder should include:

frames/

mall_gt.mat

mall_feat.mat

perspective_roi.mat

6. Next Steps
Implement data loading and preprocessing in src/data_loader.py.

Implement your CNN model for density map regression in src/model.py.

Train the model using src/train.py.

Notes
Python 3.13+ is recommended.

Tested with PyTorch 2.8+ (CPU version).

Make sure the virtual environment is activated before running scripts.

yaml
Copy code
