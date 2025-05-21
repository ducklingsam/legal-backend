# LegalTech Bot

Brief description of what your project does and its purpose.

## Table of Contents

- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ducklingsam/legal-backend/tree/dev_stepan
2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Download fine-tuned SBERT model form Google Drive and unzip it https://drive.google.com/file/d/1U2scnBap8fw5d_PiNshtxLBShPjJLluZ/view?usp=drive_link 

## Environment Variables
1. Fill in the ```.env``` file<br />
Make sure to never commit .env to version control. You can add it to your .gitignore file
2. Create the ```appeals.db``` file in the project root
3. Create  ```./uploads``` folder in the root of the project

## Usage
1. Make sure your ```.env``` file is configured correctly.
2. Run a python file to create a table in the database:
    ```bash
    python db_models.py
5. Launch the application:
    ```bash
    uvicorn app:app


## Project Structure

```bash
legal-backend/
│
├── templates/             
│   ├── application_form.html   # Page for filling out the form
│   ├── index.html              # Main page
│   └── similarity_check.html   # Page for finding similar patents
│
├── uploads/
│   └── ...                     # Uploaded documents from the form
│
├── .env                        # Environment variables
├── .gitignore
├── app.py                      # main file with FastAPI backend
├── appeals.db                  # Database file
├── config.py                   # Config file
├── db_models.py                # Database config file
├── finetune_sbert              # A notebook with finetuning and SBERT scores
├── models.py                   # SBERT backend
├── rospatent_connect.py        # API
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
