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
from threading import Event, Lock
from langchain_ollama import ChatOllama
import webrtcvad
from langdetect import detect
from collections import deque
import sounddevice as sd
from langchain_core.output_parsers import StrOutputParser

prompt = {'role':'system','content':"you are an AI assistant Named Kali ; Developed by a greate AI expert Ramachandra Udupa. You will answer to any questions asked with little sweet and short resposenses. Your resposenses sholud be completly same as a human resposense it sholud contain all types of required emotions.You are not allowed to use any kind of emogy in your resposenses.You resposens sholud be preisie and short but with all the required things"}
messages = [prompt]

class VoiceChatSystem:
    def __init__(self, model_name="llama3.2:1b"):
        self.logger = self._setup_logger()
        self.recognizer = sr.Recognizer()
        self.energy_threshold = 360  # Lowered threshold for better detection
        self.text_queue = queue.Queue()
        self.sentence_queue = queue.Queue()
        self.audio_queue = deque(maxlen=120)  # Queue for batch audio processing
        self.is_listening = True
        self.current_task_id = 0
        self.interrupt_event = Event()
        self.speaking_event = Event()
        self.speaking_lock = Lock()
        self.is_system_speaking = False
        self.last_system_audio_end = 0
        self.silence_duration = 1.4  # Adjusted for better pause detection
        self.current_sound = None
        
        self.lang = "en"
        # Initialize audio settings
        pygame.mixer.init(frequency=22000, channels=2)
        pygame.mixer.set_num_channels(4)
        sd.default.samplerate = 22000
        sd.default.channels = 2
        
        self.vad = webrtcvad.Vad(2)  # Reduced aggressiveness for better detection
        self.model = ChatOllama(model=model_name)

    def _setup_logger(self):
        logger = logging.getLogger('VoiceChatSystem')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def is_valid_human_speech(self, audio_data, timestamp):
        try:
            raw_data = np.frombuffer(audio_data.frame_data, dtype=np.int16)
            
            # Calculate audio energy
            audio_energy = np.abs(raw_data).mean()
            
            # Time since last system audio ended
            time_since_system_audio = timestamp - self.last_system_audio_end
            
            # More permissive energy threshold when system is speaking
            energy_threshold = self.energy_threshold
            if self.is_system_speaking or time_since_system_audio < 0.5:
                return False  # Ignore input while system is speaking
            
            if audio_energy < energy_threshold:
                return False
            
            # VAD analysis with larger frames for better detection
            frame_duration = 30  # ms
            frames = len(raw_data) // (16000 * frame_duration // 1000)
            
            if frames == 0:
                return False
                
            speech_frames = 0
            for i in range(frames):
                start = i * (16000 * frame_duration // 1000)
                end = start + (16000 * frame_duration // 1000)
                frame = raw_data[start:end].tobytes()
                
                try:
                    if self.vad.is_speech(frame, 16000):
                        speech_frames += 1
                except:
                    continue
            
            speech_ratio = speech_frames / frames
            return speech_ratio > 0.2  # Lowered threshold for better detection
            
        except Exception as e:
            self.logger.error(f"Error in speech validation: {str(e)}")
            return False

    def speak_text(self, text, task_id):
        if task_id != self.current_task_id or not text.strip():
            return False
        
        try:
            if self.interrupt_event.is_set():
                return False
                
            with self.speaking_lock:
                self.is_system_speaking = True
                self.speaking_event.set()
                
                mp3_fp = BytesIO()
                tts = gTTS(text=text, lang=self.lang)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                temp_file = f"temp_audio_{task_id}_{time.time()}.mp3"
                try:
                    with open(temp_file, 'wb') as f:
                        f.write(mp3_fp.getvalue())
                    
                    self.current_sound = pygame.mixer.Sound(temp_file)
                    channel = pygame.mixer.find_channel()
                    
                    if channel is None:
                        # Force stop all channels if none available
                        for i in range(pygame.mixer.get_num_channels()):
                            pygame.mixer.Channel(i).stop()
                        channel = pygame.mixer.Channel(0)
                    
                    channel.play(self.current_sound)
                    
                    while channel.get_busy() and not self.interrupt_event.is_set():
                        pygame.time.wait(10)
                    
                    channel.stop()
                    self.current_sound = None
                    
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                self.last_system_audio_end = time.time()
                return not self.interrupt_event.is_set()
                
        finally:
            self.is_system_speaking = False
            self.speaking_event.clear()
            mp3_fp.close()

    def _clear_queues(self):
        """Clear all queues"""
        try:
            # Clear text queue
            while True:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
        except queue.Empty:
            pass

        try:
            # Clear sentence queue
            while True:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
        except queue.Empty:
            pass
        
        self.logger.debug("Queues cleared")

    def immediate_interrupt(self):
        self.interrupt_event.set()
        
        with self.speaking_lock:
            # Stop current sound if exists
            if self.current_sound is not None:
                for i in range(pygame.mixer.get_num_channels()):
                    pygame.mixer.Channel(i).stop()
                self.current_sound = None
            
            # Clear audio queue
            self.audio_queue.clear()
            
            # Clear all other queues
            self._clear_queues()
        
        self.interrupt_event.clear()
        self.is_system_speaking = False
        self.last_system_audio_end = time.time()
        self.logger.debug("System interrupted and all queues cleared")

    def listen_continuously(self):
        with sr.Microphone() as source:
            self.logger.info("Adjusting for ambient noise...")
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.logger.info("Ready to listen!")
            
            while self.is_listening:
                try:
                    self.logger.info("Listening...")
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    current_time = time.time()
                    
                    if not self.is_valid_human_speech(audio, current_time):
                        continue
                    
                    # Interrupt and clear queues if new speech detected while system is speaking
                    # or there's pending text in the queue
                    if self.is_system_speaking or not self.text_queue.empty():
                        self.immediate_interrupt()
                    
                    text = None
                    for lang in ['en', 'hi', 'kn']:
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
                    
                    self.logger.info(f"Recognized: {text}")
                    
                    self.current_task_id += 1
                    self.text_queue.put((text, self.current_task_id))
                    
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.logger.error(f"Could not request results: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error in listening: {str(e)}")
                    continue

    def process_text(self):
        while self.is_listening:
            try:
                text, task_id = self.text_queue.get(timeout=1)
                
                if self.interrupt_event.is_set():
                    continue
                
                # Handle multi-line text by splitting and rejoining with spaces
                text = ' '.join(text.split('\n'))
                
                current_sentence = ""

                buffer = []
                human_message = {'role': 'user', 'content': text}
                messages.append(human_message)
                chain = ( self.model | StrOutputParser())

                complete = ""
                stream = chain.stream(messages)
                flag= False
                for chunk in stream:
                    if self.interrupt_event.is_set() or task_id != self.current_task_id:
                        break
                    if not flag and len(complete)>18:
                        self.lang = self.detect_language(complete)
                        flag = True
                    current_sentence += chunk
                    complete += chunk
                    # Split on sentence endings, keeping the punctuation
                    sentences = re.split(r'([.!?]+(?:\s+|$))', current_sentence)
                    
                    while len(sentences) >= 2:
                        sentence = sentences.pop(0) + (sentences.pop(0) if sentences else '')
                        if sentence.strip():
                            buffer.append(sentence.strip())
                            # Once we have enough sentences, send them as a batch
                            if len(buffer) >= 3:
                                combined_text = ' '.join(buffer)
                                if combined_text.strip():
                                    self.sentence_queue.put((combined_text, task_id))
                                buffer = []
                    
                    current_sentence = ''.join(sentences)
                
                # Handle any remaining text
                if buffer or current_sentence.strip():
                    remaining_text = ' '.join(buffer + [current_sentence]).strip()
                    AI_message = {'role': 'assistant', 'content': complete}
                    print(AI_message)
                    messages.append(AI_message)
                    if remaining_text:
                        self.sentence_queue.put((remaining_text, task_id))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing: {str(e)}")

    def speak_responses(self):
        while self.is_listening:
            try:
                batch_text = ""
                current_task_id = None
                
                # Collect sentences for batch processing
                try:
                    while len(batch_text.split()) < 50:  # Increased batch size
                        sentence, task_id = self.sentence_queue.get_nowait()
                        
                        if current_task_id is None:
                            current_task_id = task_id
                        
                        if task_id != current_task_id:
                            # If task ID changes, process current batch first
                            if batch_text.strip():
                                self.audio_queue.append((batch_text.strip(), current_task_id))
                            batch_text = sentence
                            current_task_id = task_id
                        else:
                            batch_text += " " + sentence
                        
                except queue.Empty:
                    if batch_text.strip():
                        self.audio_queue.append((batch_text.strip(), current_task_id))
                
                # Process audio queue without interruption from self-voice
                if self.audio_queue and not self.interrupt_event.is_set():
                    text, task_id = self.audio_queue.popleft()
                    # Set a flag to ignore input during speech
                    self.is_system_speaking = True
                    self.speak_text(text, task_id)
                    # Add a small delay after speaking
                    time.sleep(0.2)
                    self.is_system_speaking = False
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in speaking: {str(e)}")

    def detect_language(self, text):
        try:
            detected_lang = detect(text)
            return detected_lang
        except:
            return 'en'

    def stop_system(self):
        self.logger.info("Stopping system...")
        self.is_listening = False
        self.immediate_interrupt()
        pygame.mixer.quit()

    def start(self):
        self.logger.info("Starting voice chat system...")
        threads = [
            threading.Thread(target=self.listen_continuously, name="ListenThread"),
            threading.Thread(target=self.process_text, name="ProcessThread"),
            threading.Thread(target=self.speak_responses, name="SpeakThread")
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt detected, stopping...")
            self.stop_system()
        
        for thread in threads:
            thread.join()
        
        self.logger.info("System stopped")

def main():
    system = VoiceChatSystem(model_name="llama3.2")
    system.start()

if __name__ == "__main__":
    main()