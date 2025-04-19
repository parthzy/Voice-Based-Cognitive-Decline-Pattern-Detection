import os
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load and resample audio file."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Convert audio to text using Wav2Vec2."""
        inputs = self.wav2vec_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.wav2vec_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
        return transcription

class FeatureExtractor:
    def __init__(self):
        self.features = {}
    
    def extract_pause_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract pause-related features from audio."""
        # Detect silence intervals
        intervals = librosa.effects.split(audio, top_db=20)
        pauses = np.diff([i[1] - i[0] for i in intervals])
        
        return {
            'pause_mean': np.mean(pauses),
            'pause_std': np.std(pauses),
            'pause_count': len(pauses)
        }
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract prosodic features (pitch, energy, rate)."""
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0])
        pitch_std = np.std(pitches[pitches > 0])
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        
        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms)
        }
    
    def extract_hesitation_features(self, text: str) -> Dict[str, int]:
        """Extract hesitation markers from text."""
        hesitation_markers = ['uh', 'um', 'er', 'ah', 'like']
        counts = {marker: text.lower().count(marker) for marker in hesitation_markers}
        counts['total_hesitations'] = sum(counts.values())
        return counts

class CognitiveAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def prepare_feature_matrix(self, features_list: List[Dict]) -> np.ndarray:
        """Convert list of feature dictionaries to matrix."""
        df = pd.DataFrame(features_list)
        X = self.scaler.fit_transform(df)
        return X
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies in feature matrix."""
        return self.anomaly_detector.fit_predict(X)
    
    def calculate_risk_score(self, features: Dict[str, float]) -> float:
        """Calculate cognitive impairment risk score."""
        # Convert single feature dict to matrix
        X = self.scaler.transform(pd.DataFrame([features]))
        
        # Get anomaly score (-1 for anomalies, 1 for normal)
        anomaly_score = self.anomaly_detector.score_samples(X)
        
        # Convert to risk score (0-1 range, higher means more risk)
        risk_score = 1 - (anomaly_score - self.anomaly_detector.offset_) / np.abs(self.anomaly_detector.offset_)
        return float(risk_score)

def process_audio_file(file_path: str) -> Dict:
    """Process a single audio file and return all features and risk score."""
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Load and process audio
    audio, sr = preprocessor.load_audio(file_path)
    transcription = preprocessor.transcribe_audio(audio)
    
    # Extract features
    pause_features = feature_extractor.extract_pause_features(audio, sr)
    prosodic_features = feature_extractor.extract_prosodic_features(audio, sr)
    hesitation_features = feature_extractor.extract_hesitation_features(transcription)
    
    # Combine all features
    all_features = {
        **pause_features,
        **prosodic_features,
        **hesitation_features,
        'transcription': transcription
    }
    
    # Calculate risk score
    analyzer = CognitiveAnalyzer()
    feature_matrix = analyzer.prepare_feature_matrix([{k: v for k, v in all_features.items() if isinstance(v, (int, float))}])
    risk_score = analyzer.calculate_risk_score(all_features)
    
    return {
        'features': all_features,
        'risk_score': risk_score
    } 
