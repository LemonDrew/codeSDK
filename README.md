# codeSDK

A Flask-based REST API server built to submit our answers for the UBS Global Coding Challenge 2025. The challenge contained a wide variety of questions including data structures & algorithms, game simulation, and cybersecurity CTFs.

## Tech Stack

- **Framework**: [Flask](https://flask.palletsprojects.com/en/2.3.x/) 
- **Architecture**: REST API
- **Language**: Python

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LemonDrew/codeSDK.git
cd codeSDK
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python app.py
```

The server will start on `http://localhost:8080` by default.
