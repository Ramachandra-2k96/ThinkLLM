import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
import threading
import queue
import logging
import time
import re
import signal
import os
import numpy as np
from threading import Event
from langchain_ollama import ChatOllama
import webrtcvad
from langdetect import detect  # For language detection

class VoiceChatSystem:
    def __init__(self, model_name="llama3.2:1b"):
        # Initialize logging
        self.logger = self._setup_logger()
        
        # Speech recognition settings
        self.recognizer = sr.Recognizer()
        self.energy_threshold = 1000
        
        # Initialize queues
        self.text_queue = queue.Queue()
        self.sentence_queue = queue.Queue()
        
        # Initialize flags and events
        self.is_listening = True
        self.current_task_id = 0
        self.interrupt_event = Event()
        self.speaking_event = Event()
        self.is_system_speaking = False
        
        # Initialize audio control
        pygame.mixer.init(frequency=16000)
        
        # Initialize VAD for speech detection
        self.vad = webrtcvad.Vad(3)
        
        # Initialize model
        self.model = ChatOllama(model=model_name)
        
        # Keep track of active threads
        self.active_threads = set()
        self.thread_lock = threading.Lock()
        
        # Audio output detection
        self.audio_output_devices = self._get_audio_output_devices()
        
    def _setup_logger(self):
        logger = logging.getLogger('VoiceChatSystem')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def kill_thread(self, thread_id):
        """Remove thread from active threads"""
        with self.thread_lock:
            self.active_threads.discard(thread_id)

    def register_thread(self, thread_id):
        """Register new thread"""
        with self.thread_lock:
            self.active_threads.add(thread_id)

    def immediate_interrupt(self):
        """Completely stop all ongoing processes"""
        self.interrupt_event.set()
        
        # Stop audio
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        
        # Clear all queues
        self._clear_queues()
        
        # Kill all active threads except the current one
        current_thread_id = threading.get_ident()
        with self.thread_lock:
            threads_to_kill = self.active_threads.copy()
            threads_to_kill.discard(current_thread_id)
            self.active_threads.clear()
            self.active_threads.add(current_thread_id)
        
        self.interrupt_event.clear()
        self.is_system_speaking = False


    def _get_audio_output_devices(self):
        """Get list of audio output devices to ignore during recording"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [device['name'] for device in devices if device['max_output_channels'] > 0]
        except Exception as e:
            self.logger.error(f"Error getting audio devices: {str(e)}")
            return []

    def detect_language(self, text):
        """Detect language of input text"""
        try:
            detected_lang = detect(text)
            self.logger.info(f"Detected language: {detected_lang}")
            return detected_lang
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            return 'en'  # Default to English if detection fails

    def speak_text(self, text, task_id):
        """Convert text to speech with dynamic language detection"""
        if task_id != self.current_task_id:
            return False
        
        try:
            if self.interrupt_event.is_set():
                return False
                
            self.is_system_speaking = True
            self.speaking_event.set()
            
            # Detect language for this specific text
            lang = self.detect_language(text)
            
            mp3_fp = BytesIO()
            tts = gTTS(text=text, lang=lang)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            temp_file = f"temp_audio_{task_id}.mp3"
            with open(temp_file, 'wb') as f:
                f.write(mp3_fp.getvalue())
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy() and not self.interrupt_event.is_set():
                time.sleep(0.1)
            
            pygame.mixer.music.stop()
            os.remove(temp_file)
            
            return not self.interrupt_event.is_set()
            
        finally:
            self.is_system_speaking = False
            self.speaking_event.clear()
            mp3_fp.close()

    def is_valid_human_speech(self, audio_data):
        """Check if audio is valid human speech"""
        try:
            # Convert audio data to the format needed by WebRTC VAD
            raw_data = np.frombuffer(audio_data.frame_data, dtype=np.int16)
            
            # Split audio into frames and check each frame
            frame_duration = 30  # ms
            frames = len(raw_data) // (16000 * frame_duration // 1000)
            
            speech_frames = 0
            for i in range(frames):
                start = i * (16000 * frame_duration // 1000)
                end = start + (16000 * frame_duration // 1000)
                frame = raw_data[start:end].tobytes()
                
                if self.vad.is_speech(frame, 16000):
                    speech_frames += 1
            
            # Require at least 30% of frames to contain speech
            return speech_frames / frames > 0.3
            
        except Exception as e:
            self.logger.error(f"Error in speech validation: {str(e)}")
            return False

    def listen_continuously(self):
        """Listen for speech with automatic language detection"""
        thread_id = threading.get_ident()
        self.register_thread(thread_id)
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    self.logger.info("Listening...")
                    audio = self.recognizer.listen(source, phrase_time_limit=None)
                    
                    if not self.is_valid_human_speech(audio):
                        continue
                    
                    # Try to recognize speech in multiple languages
                    text = None
                    for lang in ['en', 'hi', 'kn']:  # Add more languages as needed
                        try:
                            text = self.recognizer.recognize_google(audio, language=lang)
                            if text:
                                break
                        except:
                            continue
                    
                    if not text:
                        continue
                        
                    if text.lower() in ['quit', 'exit', 'stop', 'bye']:
                        self.stop_system()
                        break
                    
                    self.logger.info(f"Recognized speech: {text}")
                    
                    # New task
                    self.current_task_id += 1
                    self.text_queue.put((text, self.current_task_id))
                    
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.logger.error(f"Could not request results: {str(e)}")
                    continue
                
        self.kill_thread(thread_id)
    
    def process_text(self):
        """Process text with complete task isolation"""
        thread_id = threading.get_ident()
        self.register_thread(thread_id)
        
        while self.is_listening:
            try:
                text, task_id = self.text_queue.get(timeout=1)
                
                if self.interrupt_event.is_set():
                    continue
                
                current_sentence = ""
                stream = self.model.stream(text)
                
                for chunk in stream:
                    if self.interrupt_event.is_set() or task_id != self.current_task_id:
                        break
                    
                    current_sentence += chunk.content
                    sentences = re.split(r'([.!?]+)', current_sentence)
                    
                    while len(sentences) >= 2 and not self.interrupt_event.is_set():
                        sentence = sentences.pop(0) + sentences.pop(0)
                        if sentence.strip():
                            self.sentence_queue.put((sentence, task_id))
                    
                    current_sentence = ''.join(sentences)
                
                if current_sentence.strip() and not self.interrupt_event.is_set():
                    self.sentence_queue.put((current_sentence, task_id))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing: {str(e)}")
        
        self.kill_thread(thread_id)

    def speak_responses(self):
        """Speak responses with complete task isolation"""
        thread_id = threading.get_ident()
        self.register_thread(thread_id)
        
        while self.is_listening:
            try:
                if self.interrupt_event.is_set():
                    continue
                    
                sentence, task_id = self.sentence_queue.get(timeout=1)
                
                if task_id == self.current_task_id and not self.interrupt_event.is_set():
                    self.speak_text(sentence, task_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in speaking: {str(e)}")
        
        self.kill_thread(thread_id)

    def _clear_queues(self):
        """Completely clear all queues"""
        # Clear text queue
        while True:
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear sentence queue
        while True:
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break

    def stop_system(self):
        """Stop everything completely"""
        self.is_listening = False
        self.immediate_interrupt()
        pygame.mixer.quit()

    def start(self):
        """Start the system with complete process isolation"""
        self.listen_thread = threading.Thread(target=self.listen_continuously)
        self.process_thread = threading.Thread(target=self.process_text)
        self.speak_thread = threading.Thread(target=self.speak_responses)
        
        self.listen_thread.start()
        self.process_thread.start()
        self.speak_thread.start()
        
        try:
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_system()
        
        self.listen_thread.join()
        self.process_thread.join()
        self.speak_thread.join()

def main():
    system = VoiceChatSystem(model_name="llama3.2:1b")
    system.start()

if __name__ == "__main__":
    main()