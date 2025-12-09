"""
Speaker Recognition Module (Resemblyzer-based)
Handles speaker enrollment, identification, and profile management using Deep Learning embeddings.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

class SpeakerRecognizer:
    """
    Speaker recognition system using Resemblyzer (Deep Learning Embeddings).
    Supports enrollment, identification, and profile management.
    """
    
    def __init__(self, profiles_dir: str = "data/speaker_profiles"):
        """
        Initialize speaker recognizer.
        
        Args:
            profiles_dir: Directory to store speaker profiles
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading Resemblyzer VoiceEncoder model...")
        try:
            self.encoder = VoiceEncoder()
            print("VoiceEncoder loaded successfully.")
        except Exception as e:
            print(f"Error loading VoiceEncoder: {e}")
            self.encoder = None
        
        # Load existing profiles
        self.profiles = self._load_all_profiles()
    
    def _load_all_profiles(self) -> Dict[str, np.ndarray]:
        """
        Load all speaker profiles from disk.
        
        Returns:
            Dictionary mapping speaker names to embedding vectors
        """
        profiles = {}
        
        if not self.profiles_dir.exists():
            return profiles
        
        for profile_file in self.profiles_dir.glob("*.pkl"):
            try:
                with open(profile_file, 'rb') as f:
                    data = pickle.load(f)
                    speaker_name = data['name']
                    features = data['features']
                    profiles[speaker_name] = features
            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")
        
        return profiles
    
    def _extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract embedding features from audio file using Resemblyzer.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding vector (256-d) or None if error
        """
        if self.encoder is None:
            print("Error: VoiceEncoder not initialized")
            return None
            
        try:
            # Preprocess wav (resample, normalize, etc.)
            wav = preprocess_wav(audio_path)
            
            # Create embedding
            embedding = self.encoder.embed_utterance(wav)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _extract_features_from_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract features from audio bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Embedding vector or None if error
        """
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Extract features
            features = self._extract_features(tmp_path)
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from bytes: {e}")
            return None
    
    def enroll_speaker(self, name: str, audio_samples: List[str]) -> bool:
        """
        Enroll a new speaker with multiple audio samples.
        
        Args:
            name: Speaker name
            audio_samples: List of paths to audio files
            
        Returns:
            True if enrollment successful, False otherwise
        """
        if len(audio_samples) < 3:
            print("Error: Need at least 3 audio samples for enrollment")
            return False
        
        try:
            # Extract features from all samples
            embeddings = []
            for audio_path in audio_samples:
                embed = self._extract_features(audio_path)
                if embed is not None:
                    embeddings.append(embed)
            
            if len(embeddings) < 3:
                print("Error: Could not extract enough valid features")
                return False
            
            # Average embeddings to create robust profile
            # Resemblyzer embeddings are normalized, so averaging works well
            avg_embedding = np.mean(embeddings, axis=0)
            # Re-normalize after averaging
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Save profile
            profile_data = {
                'name': name,
                'features': avg_embedding,
                'num_samples': len(embeddings),
                'method': 'resemblyzer'
            }
            
            profile_path = self.profiles_dir / f"{name.lower().replace(' ', '_')}.pkl"
            with open(profile_path, 'wb') as f:
                pickle.dump(profile_data, f)
            
            # Update in-memory profiles
            self.profiles[name] = avg_embedding
            
            print(f"Successfully enrolled {name} with {len(embeddings)} samples")
            return True
            
        except Exception as e:
            print(f"Error enrolling speaker: {e}")
            return False
    
    def enroll_speaker_from_bytes(self, name: str, audio_bytes_list: List[bytes]) -> bool:
        """
        Enroll a new speaker with multiple audio samples (from bytes).
        
        Args:
            name: Speaker name
            audio_bytes_list: List of audio data as bytes
            
        Returns:
            True if enrollment successful, False otherwise
        """
        if len(audio_bytes_list) < 3:
            print("Error: Need at least 3 audio samples for enrollment")
            return False
        
        try:
            # Extract features from all samples
            embeddings = []
            for audio_bytes in audio_bytes_list:
                embed = self._extract_features_from_bytes(audio_bytes)
                if embed is not None:
                    embeddings.append(embed)
            
            if len(embeddings) < 3:
                print("Error: Could not extract enough valid features")
                return False
            
            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            # Re-normalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Save profile
            profile_data = {
                'name': name,
                'features': avg_embedding,
                'num_samples': len(embeddings),
                'method': 'resemblyzer'
            }
            
            profile_path = self.profiles_dir / f"{name.lower().replace(' ', '_')}.pkl"
            with open(profile_path, 'wb') as f:
                pickle.dump(profile_data, f)
            
            # Update in-memory profiles
            self.profiles[name] = avg_embedding
            
            print(f"Successfully enrolled {name} with {len(embeddings)} samples")
            return True
            
        except Exception as e:
            print(f"Error enrolling speaker: {e}")
            return False
    
    def identify_speaker(self, audio_path: str, threshold: float = 0.75) -> Tuple[Optional[str], float]:
        """
        Identify speaker from audio file.
        
        Args:
            audio_path: Path to audio file
            threshold: Similarity threshold (0-1). 
                       0.75 is typical for Resemblyzer.
            
        Returns:
            Tuple of (speaker_name, confidence) or (None, 0) if unknown
        """
        if len(self.profiles) == 0:
            return None, 0.0
        
        try:
            # Extract features from input audio
            input_embedding = self._extract_features(audio_path)
            
            if input_embedding is None:
                return None, 0.0
            
            # Compare with all enrolled speakers
            best_match = None
            best_similarity = 0.0
            
            for name, profile_embedding in self.profiles.items():
                # Check if profile is compatible (size 256)
                if profile_embedding.shape != input_embedding.shape:
                    print(f"Warning: Profile {name} has incompatible shape {profile_embedding.shape}")
                    continue
                
                # Calculate cosine similarity (dot product of normalized vectors)
                # Resemblyzer embeddings are L2 normalized
                similarity = np.inner(input_embedding, profile_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Check if similarity exceeds threshold
            if best_similarity >= threshold:
                return best_match, float(best_similarity)
            else:
                return None, float(best_similarity)
                
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return None, 0.0
    
    def delete_speaker(self, name: str) -> bool:
        """
        Delete a speaker profile.
        
        Args:
            name: Speaker name to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Remove from in-memory profiles
            if name in self.profiles:
                del self.profiles[name]
            
            # Remove profile file
            profile_path = self.profiles_dir / f"{name.lower().replace(' ', '_')}.pkl"
            if profile_path.exists():
                profile_path.unlink()
                print(f"Successfully deleted profile for {name}")
                return True
            else:
                print(f"Profile not found for {name}")
                return False
                
        except Exception as e:
            print(f"Error deleting speaker: {e}")
            return False
    
    def list_enrolled_speakers(self) -> List[str]:
        """
        Get list of all enrolled speakers.
        
        Returns:
            List of speaker names
        """
        return list(self.profiles.keys())
    
    def get_num_enrolled(self) -> int:
        """
        Get number of enrolled speakers.
        
        Returns:
            Number of enrolled speakers
        """
        return len(self.profiles)
