# MemoTag Speech Intelligence Module

This module analyzes speech patterns to detect early signs of cognitive impairment using advanced audio processing and machine learning techniques.

## Features

- Audio preprocessing and feature extraction
- Speech-to-text conversion using Wav2Vec2
- Prosodic feature analysis (pitch, energy, rate)
- Hesitation and pause pattern detection
- Anomaly detection for cognitive impairment risk assessment
- FastAPI endpoint for real-time analysis
- Comprehensive visualization and analysis notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/memotag.git
cd memotag
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
memotag/
├── data/               # Sample audio files and processed data
├── src/               # Source code
│   ├── speech_processor.py  # Core processing logic
│   └── api.py              # FastAPI endpoint
├── notebooks/         # Jupyter notebooks for analysis
├── tests/            # Unit tests
├── models/           # Saved model files
└── requirements.txt  # Project dependencies
```

## Usage

### Running the API

```bash
cd src
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- POST `/analyze-speech/`: Upload and analyze audio file
- GET `/health`: Health check endpoint

### Example API Request

```python
import requests

files = {
    'audio_file': ('sample.wav', open('sample.wav', 'rb'))
}

response = requests.post('http://localhost:8000/analyze-speech/', files=files)
result = response.json()
```

### Running the Analysis Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/speech_analysis_demo.ipynb`

## Feature Details

### Audio Features Extracted:

1. Pause Patterns
   - Mean pause duration
   - Pause frequency
   - Pause duration variance

2. Prosodic Features
   - Pitch statistics
   - Energy contours
   - Speech rate

3. Linguistic Features
   - Hesitation markers
   - Word recall patterns
   - Speech fluency metrics

### Machine Learning Approach

- Unsupervised anomaly detection using Isolation Forest
- Feature standardization and normalization
- Risk score calculation based on anomaly scores

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wav2Vec2 model from Hugging Face
- LibROSA for audio processing
- FastAPI for API development
