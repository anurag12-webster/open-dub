import os
import torch
import whisper
import numpy as np
import nltk
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import re
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
from whisper import load_model, transcribe
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from utils import (
    get_cat_logger
)

logger = get_cat_logger(name="Transcription", level="INFO")

device = "cuda" if torch.cuda.is_available() else "cpu"

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str = ""
    ref_text: Optional[str] = None
    id: Optional[int] = None
    gender: Optional[str] = None

class TranscriptionResult(BaseModel):
    text: str
    segments: List[TranscriptionSegment] = []
    language: Optional[str] = None
    error: Optional[str] = None

class TranscriptionModel:
    def __init__(self, model_size: str = "large"):
        try:
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("Failed to download NLTK punkt")
        
        self.model_size: str = model_size
        self.model = None
    
    def load_model(self) -> bool:
        try:
            logger.animate('loading', duration=1.0, message=f"Loading Whisper {self.model_size} model")
            self.model = load_model(self.model_size, device=device)
            logger.info(f"Loaded Whisper {self.model_size} model on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        if not self.model:
            self.load_model()
        
        try:
            temperature = tuple(np.arange(0, 1.0 + 1e-6, 0.2))
            
            logger.animate('thinking', duration=1.0, message="Transcribing audio")
            result = transcribe(
                self.model, 
                audio_path, 
                temperature=temperature,
                language=language,
                best_of=5,
                beam_size=5,
                patience=None,
                length_penalty=None,
                suppress_tokens="-1",
                initial_prompt=None,
                condition_on_previous_text=True,
                fp16=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.3,
                verbose=False,
            )
            
            segments = [
                TranscriptionSegment(
                    start=seg['start'],
                    end=seg['end'],
                    text=seg['text'],
                    ref_text=seg['text']
                ) for seg in result["segments"]
            ]
            
            return TranscriptionResult(
                text=result["text"],
                segments=segments,
                language=result.get("language", language)
            )
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return TranscriptionResult(text="", error=str(e))

    def process_segments(self, result: TranscriptionResult, total_audio_chunks: int, 
                       gender: Optional[str] = None) -> List[TranscriptionSegment]:
        try:
            filtered_segments = result.segments
            
            # Add gender marker
            if gender:
                for segment in filtered_segments:
                    segment.gender = gender
            
            # Sort segments by start time
            filtered_segments.sort(key=lambda x: x.start)
            modified_segments: List[TranscriptionSegment] = []
            current_time = 0.0

            # Fill gaps between segments
            for segment in filtered_segments:
                if current_time < segment.start:
                    modified_segments.append(TranscriptionSegment(
                        start=current_time, 
                        end=segment.start, 
                        text="",
                        ref_text=""
                    ))
                modified_segments.append(segment)
                current_time = segment.end

            # Add final segment if needed
            if filtered_segments and current_time < filtered_segments[-1].end:
                modified_segments.append(TranscriptionSegment(
                    start=current_time, 
                    end=filtered_segments[-1].end, 
                    text="",
                    ref_text=""
                ))
            
            # Assign chunk IDs
            for segment in modified_segments:
                segment_end = int(segment.end // 15)
                segment.id = total_audio_chunks - 1 if segment_end >= total_audio_chunks else segment_end
            
            return modified_segments
            
        except Exception as e:
            logger.error(f"Error processing segments: {str(e)}")
            return []
    
    def get_audio_duration(self, audio_path: str) -> float:
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0
    
    def split_audio(self, audio_path: str, chunk_length: int = 10, 
                  min_last_chunk_length: int = 10) -> List[str]:
        try:
            audio = AudioSegment.from_file(audio_path)
            total_length_ms = len(audio)
            chunk_length_ms = chunk_length * 1000
            min_last_chunk_length_ms = min_last_chunk_length * 1000

            num_full_chunks = total_length_ms // chunk_length_ms
            remaining_length_ms = total_length_ms % chunk_length_ms

            chunks_dir = "audio_chunks"
            os.makedirs(chunks_dir, exist_ok=True)

            audio_chunks_paths = []

            for i in range(num_full_chunks):
                start_time = i * chunk_length_ms
                end_time = start_time + chunk_length_ms
                chunk = audio[start_time:end_time]
                chunk_path = os.path.join(chunks_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                audio_chunks_paths.append(chunk_path)

            if remaining_length_ms >= min_last_chunk_length_ms:
                start_time = num_full_chunks * chunk_length_ms
                chunk = audio[start_time:]
                chunk_path = os.path.join(chunks_dir, f"chunk_{num_full_chunks}.wav")
                chunk.export(chunk_path, format="wav")
                audio_chunks_paths.append(chunk_path)

            return audio_chunks_paths

        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}")
            return []
    
    def convert_to_mono(self, file_path: str) -> str:
        try:
            audio = AudioSegment.from_wav(file_path)
            audio = audio.set_channels(1)
            mono_file_path = file_path.replace(".wav", "_mono.wav")
            audio.export(mono_file_path, format="wav")
            return mono_file_path
        except Exception as e:
            logger.error(f"Error converting to mono: {str(e)}")
            return file_path
    
    def process_audio_file(self, file_path: str, source_lang: Optional[str] = None, 
                         gender: Optional[str] = None) -> List[TranscriptionSegment]:
        try:
            logger.info(f"Processing audio file: {file_path}")
            
            # Convert to mono if needed
            mono_path = self.convert_to_mono(file_path)
            
            # Split audio into chunks
            chunk_paths = self.split_audio(mono_path)
            
            # Transcribe the full file
            result = self.transcribe_audio(mono_path, language=source_lang)
            
            # Process segments
            segments = self.process_segments(result, len(chunk_paths), gender)
            
            # Clean up text
            for segment in segments:
                if not segment.text and segment.ref_text:
                    segment.text = segment.ref_text
                segment.text = self.clean_text(segment.text)
            
            return segments
        
        except Exception as e:
            logger.error(f"Error in process_audio_file: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        try:
            text = re.sub(r'<.*?>', '', text)
            text = text.replace('"', '')
            return text.strip()
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text

def extract_audio(video_file_path: str, output_audio_path: str) -> None:
    try:
        video_clip = VideoFileClip(video_file_path)
        duration = video_clip.duration
        if duration > 120:
            video_clip = video_clip.subclip(0, 120)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec='mp3')
        video_clip.close()
        audio_clip.close()
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")

def detect_gender(input_audio_path: str) -> str:
    # Placeholder for actual gender detection
    logger.info(f"Detecting gender from audio: {input_audio_path}")
    return "male"  # Default return
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            
        Returns:
            Dict containing transcription results including text and segments
        """
        if not self.model:
            self.load_model()
        
        try:
            # Set up temperature for beam search
            temperature = tuple(np.arange(0, 1.0 + 1e-6, 0.2))
            
            # Perform transcription
            result = transcribe(
                self.model, 
                audio_path, 
                temperature=temperature,
                language=language,
                best_of=5,
                beam_size=5,
                patience=None,
                length_penalty=None,
                suppress_tokens="-1",
                initial_prompt=None,
                condition_on_previous_text=True,
                fp16=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.3,
                verbose=False,
            )
            
            return result
        except Exception as e:
            logger.exception(f"Error transcribing audio: {e}")
            return {"error": str(e), "text": "", "segments": []}

    def process_segments(self, result: Dict[str, Any], total_audio_chunks: int, 
                        gender: str = None) -> List[Dict[str, Any]]:
        """
        Process transcription segments and prepare them for translation/TTS
        
        Args:
            result: Transcription result from Whisper
            total_audio_chunks: Number of chunks the audio was split into
            gender: Detected gender of speaker (male/female), used for voice selection
            
        Returns:
            List of segments with start/end times and text
        """
        try:
            segments = result["segments"]
            filtered_segments = [{'start': seg['start'], 'end': seg['end'], 'ref_text': seg['text']} 
                               for seg in segments]
            
            # Add gender marker for potential voice selection later
            if gender:
                for item in filtered_segments:
                    item['gender'] = gender
            
            # Sort segments by start time
            filtered_segments.sort(key=lambda x: x['start'])
            modified_segments = []
            current_time = 0.0

            # Fill gaps between segments with empty segments
            for segment in filtered_segments:
                if current_time < segment['start']:
                    modified_segments.append({
                        'start': current_time, 
                        'end': segment['start'], 
                        'ref_text': '',
                        'text': ''
                    })
                modified_segments.append(segment)
                current_time = segment['end']

            # Add a final segment if needed
            if filtered_segments:
                end_time = filtered_segments[-1]['end']
                if current_time < end_time:
                    modified_segments.append({
                        'start': current_time, 
                        'end': end_time, 
                        'ref_text': '',
                        'text': ''
                    })
            
            # Assign chunk IDs based on end time
            def assign_id(segment):
                if int(segment['end'] // 15) == total_audio_chunks:
                    segment['id'] = total_audio_chunks - 1
                else:
                    segment['id'] = int(segment['end'] // 15)
                return segment

            segments_with_id = [assign_id(segment) for segment in modified_segments]
            return segments_with_id
            
        except Exception as e:
            logger.exception(f"Error processing segments: {e}")
            return []
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            return duration_ms / 1000
        except Exception as e:
            logger.exception(f"Error getting audio duration: {e}")
            return 0
    
    def split_audio(self, audio_path: str, chunk_length: int = 10, 
                   min_last_chunk_length: int = 10) -> List[str]:
        """
        Split audio file into chunks for processing
        
        Args:
            audio_path: Path to audio file
            chunk_length: Length of each chunk in seconds
            min_last_chunk_length: Minimum length for the last chunk in seconds
            
        Returns:
            List of paths to chunk files
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            total_length_ms = len(audio)
            chunk_length_ms = chunk_length * 1000
            min_last_chunk_length_ms = min_last_chunk_length * 1000

            num_full_chunks = total_length_ms // chunk_length_ms
            remaining_length_ms = total_length_ms % chunk_length_ms

            chunks_dir = "audio_chunks"
            if not os.path.exists(chunks_dir):
                os.makedirs(chunks_dir)

            audio_chunks_paths = []

            for i in range(num_full_chunks):
                start_time = i * chunk_length_ms
                end_time = start_time + chunk_length_ms
                chunk = audio[start_time:end_time]
                chunk_path = os.path.join(chunks_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                audio_chunks_paths.append(chunk_path)

            if remaining_length_ms >= min_last_chunk_length_ms:
                start_time = num_full_chunks * chunk_length_ms
                chunk = audio[start_time:]
                chunk_path = os.path.join(chunks_dir, f"chunk_{num_full_chunks}.wav")
                chunk.export(chunk_path, format="wav")
                audio_chunks_paths.append(chunk_path)

            return audio_chunks_paths

        except Exception as e:
            logger.exception(f"Error splitting audio: {e}")
            return []
    
    def convert_to_mono(self, file_path: str) -> str:
        """Convert audio file to mono"""
        try:
            audio = AudioSegment.from_wav(file_path)
            audio = audio.set_channels(1)
            mono_file_path = file_path.replace(".wav", "_mono.wav")
            audio.export(mono_file_path, format="wav")
            return mono_file_path
        except Exception as e:
            logger.exception(f"Error converting to mono: {e}")
            return file_path  # Return original path in case of error
    
    @timed(log=True)
    def process_audio_file(self, file_path: str, source_lang: str = None, gender: str = None) -> List[Dict]:
        """
        Full pipeline to process audio file: convert to mono, split, and transcribe
        
        Args:
            file_path: Path to audio file
            source_lang: Source language code
            gender: Speaker gender (male/female) or None for auto-detection
            
        Returns:
            List of segments with transcriptions
        """
        try:
            # Convert to mono if needed
            mono_path = self.convert_to_mono(file_path)
            
            # Split audio into chunks
            chunk_paths = self.split_audio(mono_path)
            
            # Transcribe the full file
            result = self.transcribe_audio(mono_path, language=source_lang)
            
            # Process segments
            segments = self.process_segments(result, len(chunk_paths), gender)
            
            # Clean up text
            for entry in segments:
                if 'text' not in entry and 'ref_text' in entry:
                    entry['text'] = entry['ref_text']
                if 'text' in entry:
                    entry['text'] = self.clean_text(entry['text'])
            
            return segments
        
        except Exception as e:
            logger.exception(f"Error in process_audio_file: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean up transcribed text"""
        try:
            import re
            text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
            text = text.replace('"', '')       # Remove quotes
            return text.strip()
        except Exception as e:
            logger.exception(f"Error in clean_text: {e}")
            return text
    
# Helper functions that can be used outside the class

def extract_audio(video_file_path: str, output_audio_path: str):
    """Extract audio from video file"""
    try:
        video_clip = VideoFileClip(video_file_path)
        duration = video_clip.duration
        if duration > 120:
            video_clip = video_clip.subclip(0, 120)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec='mp3')
        video_clip.close()
        audio_clip.close()

    except Exception as e:
        logger.exception(f"Error while extracting audio: {e}")

def detect_gender(input_audio_path: str) -> str:
    """
    Basic gender detection helper function.
    
    Note: This is simplified. In a real implementation, you'd need the gender detection model.
    """
    try:
        # Placeholder for actual gender detection
        # In a real implementation, you'd need to load and use the gender detection model
        
        # Example of how it would be called in the real implementation:
        gender_identify_model = create_model()
        gender_identify_model.load_weights("path/to/model.h5")
        features = extract_feature(input_audio_path, mel=True).reshape(1, -1)
        male_prob = gender_identify_model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        del gender_identify_model
        return gender
    except Exception as e:
        logger.exception(f"Error in gender detection: {e}")
        return e
