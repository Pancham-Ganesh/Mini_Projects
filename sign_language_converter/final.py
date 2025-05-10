import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import openai
from pygame import mixer, display, time
import pygame
import threading

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:
    def gpt_generate_response(self, user_input, conversation_history=[]):
        messages = [
        {"role": "system", "content": 
        "You are a friendly and supportive chatbot. Your goal is to provide warm and understanding responses to users who need emotional support. \
        Acknowledge their feelings, encourage positivity, and offer helpful insights. If a user is in distress, offer kind words and coping strategies, \
        but do not mention professional help unless explicitly asked."
        }
        ]

        for msg in conversation_history:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["bot"]})

        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.6,
                top_p=0.95
            )
            bot_response = response.choices[0].message.content.strip()
            conversation_history.append({"user": user_input, "bot": bot_response})
            return bot_response

        except Exception as e:
            print("Error with OpenAI API:", e)
            return "I'm here for you. If you're feeling low, it's okay to talk about it. You're not alone."
        
    def chat_with_bot(self, use_text_input=True):
        message = ""
        if use_text_input:
            message = self.text_input.get("1.0", tk.END).strip()
            if not message:
                return
            self.text_input.delete("1.0", tk.END)
        else:
            message = self.str.strip()
            if not message:
                return
            self.str = " "
            self.panel5.config(text=self.str)
        
        if message:
            # Disable send button during processing
            self.text_send.config(state=tk.DISABLED)
            self.sign_send.config(state=tk.DISABLED)
            
            # Show loading indicator
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, "Processing...\n", "loading")
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
            
            # Start API call in separate thread
            threading.Thread(target=self._process_chat_message, args=(message,), daemon=True).start()

    def _process_chat_message(self, message):
        try:
            # Display user message
            self.root.after(0, self.display_message, "You: " + message, "user")
            
            # Get bot response
            bot_response = self.gpt_generate_response(message, self.conversation_history)
            
            # Update UI with bot response
            self.root.after(0, self.display_message, "Chatbot: " + bot_response, "bot")
            
        except Exception as e:
            error_msg = "I'm here for you. If you're feeling low, it's okay to talk about it. You're not alone."
            self.root.after(0, self.display_message, "Chatbot: " + error_msg, "bot")
        
        finally:
            # Re-enable send buttons
            self.root.after(0, lambda: self.text_send.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.sign_send.config(state=tk.NORMAL))
            
            # Remove loading indicator
            self.root.after(0, self._remove_loading_indicator)

    def _remove_loading_indicator(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("end-2l", "end-1c")  # Remove "Processing..." line
        self.chat_display.config(state=tk.DISABLED)

    def display_message(self, message, sender):
        self.chat_display.config(state=tk.NORMAL)
        
        # Insert the message with appropriate tag
        self.chat_display.insert(tk.END, f"{'You: ' if sender == 'user' else 'Chatbot: '}", "bold")
        self.chat_display.insert(tk.END, message[len("You: ") if sender == "user" else len("Chatbot: "):] + "\n\n", sender)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Play sign language video for bot responses
        if sender == "bot":
            bot_message = message[len("Chatbot: "):]
            self.play_sign_language_video(bot_message)

    def play_sign_language_video(self, text):
    # Stop any currently playing video
        if self.video_playing:
            self.video_playing = False
            if self.video_capture is not None:
                self.video_capture.release()
        
        # Clean the text
        words = []
        for word in text.split():
            clean_word = ''.join([c for c in word if c.isalpha()]).title()
            if clean_word:
                words.append(clean_word)
        
        # Find matching video files
        video_files = []
        video_dir = "videos"
        
        for word in words:
            # Try variations
            variations = [word, word.upper(), word.lower(), word.capitalize()]
            
            for variant in variations:
                word_video = os.path.join(video_dir, f"{variant}.mp4")
                if os.path.exists(word_video):
                    video_files.append(word_video)
                    # Add pause video after each word
                    video_files.append(os.path.join(video_dir, "pause.mp4"))
                    break
            else:
                # Try letter by letter
                for letter in word:
                    letter_video = os.path.join(video_dir, f"{letter.upper()}.mp4")
                    if os.path.exists(letter_video):
                        video_files.append(letter_video)
                        # Add short pause after each letter
                        video_files.append(os.path.join(video_dir, "short_pause.mp4"))
        
        if video_files:
            # Remove the last pause if it exists
            if video_files[-1].endswith("pause.mp4") or video_files[-1].endswith("short_pause.mp4"):
                video_files = video_files[:-1]
                
            self.current_video_index = 0
            self.video_files = video_files
            self.play_next_video()
        else:
            self.video_name_label.config(text="No sign language video available")

    def play_next_video(self):
        if self.video_playing or self.current_video_index >= len(self.video_files):
            return
        
        self.video_file_path = self.video_files[self.current_video_index]
        video_name = os.path.splitext(os.path.basename(self.video_file_path))[0]
        
        # Skip showing name for pause videos
        if not (video_name == "pause" or video_name == "short_pause"):
            self.video_name_label.config(text=video_name)
        else:
            self.video_name_label.config(text="")
        
        try:
            if self.video_capture is not None:
                self.video_capture.release()
            
            self.video_capture = cv2.VideoCapture(self.video_file_path)
            if not self.video_capture.isOpened():
                raise Exception("Could not open video file")
                
            # Set slower playback speed (50% of normal speed)
            self.video_capture.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS
            
            self.video_playing = True
            self.update_video_frame()
        except Exception as e:
            print(f"Error playing video: {e}")
            self.current_video_index += 1
            self.root.after(100, self.play_next_video)

    def update_video_frame(self):
        if not self.video_playing or self.video_capture is None:
            return
        
        try:
            ret, frame = self.video_capture.read()
            
            if ret:
                frame = cv2.resize(frame, (320, 240))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_panel.imgtk = imgtk
                self.video_panel.config(image=imgtk)
                
                # Schedule next frame update with slower delay (66ms for ~15fps)
                self.root.after(66, self.update_video_frame)
            else:
                # Video ended
                self.video_capture.release()
                self.video_playing = False
                self.current_video_index += 1
                
                # Add extra delay between videos
                if self.current_video_index < len(self.video_files):
                    # Longer pause after words, shorter after letters
                    if self.video_files[self.current_video_index].endswith("pause.mp4"):
                        delay = 1000  # 1 second pause after words
                    elif self.video_files[self.current_video_index].endswith("short_pause.mp4"):
                        delay = 500   # 0.5 second pause after letters
                    else:
                        delay = 300   # 0.3 second pause between letters of same word
                    
                    self.root.after(delay, self.play_next_video)
        except Exception as e:
            print(f"Error updating video frame: {e}")
            self.video_playing = False

    def decrease_video_speed(self):
        """Slow down the video playback even more"""
        # This can be called from a "Slower" button if you add one
        self.video_capture.set(cv2.CAP_PROP_FPS, 10)  # Reduce to 10fps
        self.root.after_cancel(self.update_video_frame)  # Cancel pending updates
        self.update_video_frame()  # Restart with new speed

    def increase_video_speed(self):
        """Speed up the video playback"""
        # This can be called from a "Faster" button if you add one
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)  # Increase to 30fps
        self.root.after_cancel(self.update_video_frame)  # Cancel pending updates
        self.update_video_frame()  # Restart with new speed
    
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        self.client = openai.OpenAI(api_key="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")
        self.conversation_history = []

        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10

        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        # Initialize pygame for video playback
        pygame.display.init()
        mixer.init()

        self.root = tk.Tk()
        self.root.title("Sign Language To Text")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Create main frames with scrollable left panel
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollable left frame
        self.left_canvas = tk.Canvas(self.root)
        self.left_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollable_frame = ttk.Frame(self.left_canvas)
        
        self.left_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(
                scrollregion=self.left_canvas.bbox("all")
            )
        )
        
        self.left_canvas.create_window((0, 0), window=self.left_scrollable_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Camera and processed image panels
        self.camera_frame = ttk.LabelFrame(self.left_scrollable_frame, text="Camera Feed")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.panel = tk.Label(self.camera_frame)
        self.panel.pack(fill=tk.BOTH, expand=True)

        self.processed_frame = ttk.LabelFrame(self.left_scrollable_frame, text="Processed Image")
        self.processed_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.panel2 = tk.Label(self.processed_frame)
        self.panel2.pack(fill=tk.BOTH, expand=True)

        # Text output area
        self.text_frame = ttk.LabelFrame(self.left_scrollable_frame, text="Text Output")
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.T1 = tk.Label(self.text_frame, text="Character:", font=("Courier", 12))
        self.T1.pack(anchor=tk.W)

        self.panel3 = tk.Label(self.text_frame, text="", font=("Courier", 30))
        self.panel3.pack(anchor=tk.W)

        self.T3 = tk.Label(self.text_frame, text="Sentence:", font=("Courier", 12))
        self.T3.pack(anchor=tk.W)

        self.panel5 = tk.Label(self.text_frame, text="", font=("Courier", 14), wraplength=400)
        self.panel5.pack(anchor=tk.W, fill=tk.X)

        # Suggestions
        self.suggestions_frame = ttk.LabelFrame(self.left_scrollable_frame, text="Suggestions")
        self.suggestions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.T4 = tk.Label(self.suggestions_frame, text="Suggestions:", fg="red", font=("Courier", 12))
        self.T4.pack(anchor=tk.W)

        self.b1 = tk.Button(self.suggestions_frame, text="", font=("Courier", 12), wraplength=400, command=self.action1)
        self.b1.pack(fill=tk.X)

        self.b2 = tk.Button(self.suggestions_frame, text="", font=("Courier", 12), wraplength=400, command=self.action2)
        self.b2.pack(fill=tk.X)

        self.b3 = tk.Button(self.suggestions_frame, text="", font=("Courier", 12), wraplength=400, command=self.action3)
        self.b3.pack(fill=tk.X)

        self.b4 = tk.Button(self.suggestions_frame, text="", font=("Courier", 12), wraplength=400, command=self.action4)
        self.b4.pack(fill=tk.X)

        # Control buttons
        self.control_frame = ttk.Frame(self.left_scrollable_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.speak = tk.Button(self.control_frame, text="Speak", font=("Courier", 12), command=self.speak_fun)
        self.speak.pack(side=tk.LEFT, padx=5)

        self.clear = tk.Button(self.control_frame, text="Clear", font=("Courier", 12), command=self.clear_fun)
        self.clear.pack(side=tk.LEFT, padx=5)

        self.sign_send = tk.Button(self.control_frame, text="Send Sign", font=("Courier", 12), 
                                command=lambda: self.chat_with_bot(use_text_input=False))
        self.sign_send.pack(side=tk.LEFT, padx=5)

        # Chat area
        self.chat_frame = ttk.LabelFrame(self.right_frame, text="Chat")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat_display = tk.Text(self.chat_frame, height=20, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.chat_frame, command=self.chat_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.config(yscrollcommand=scrollbar.set)

        # Configure tags for chat display
        self.chat_display.tag_config("bold", font=("Courier", 10, "bold"))
        self.chat_display.tag_config("user", foreground="black")
        self.chat_display.tag_config("bot", foreground="blue", lmargin1=20, lmargin2=20)
        self.chat_display.tag_config("loading", foreground="gray", font=("Courier", 10, "italic"))

        # Text input area
        self.input_frame = ttk.Frame(self.right_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.text_input = tk.Text(self.input_frame, height=3, width=60, font=("Courier", 12), 
                                wrap=tk.WORD, relief=tk.GROOVE, borderwidth=2)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.text_input.bind("<Return>", lambda event: self.chat_with_bot())

        self.text_send = tk.Button(self.input_frame, text="Send", font=("Courier", 12), 
                                command=self.chat_with_bot)
        self.text_send.pack(side=tk.LEFT)

        # Video display area
        self.video_display_frame = ttk.LabelFrame(self.right_frame, text="Sign Language Video")
        self.video_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_name_label = tk.Label(self.video_display_frame, text="", font=("Courier", 10))
        self.video_name_label.pack()

        self.video_panel = tk.Label(self.video_display_frame)
        self.video_panel.pack(fill=tk.BOTH, expand=True)

        # Clear chat button
        self.clear_chat = tk.Button(self.right_frame, text="Clear Chat", font=("Courier", 12), 
                                command=self.clear_chat_fun)
        self.clear_chat.pack(pady=5)

        # Loading indicator
        self.loading_label = tk.Label(self.right_frame, text="", fg="blue")
        self.loading_label.pack()

        # Video playback variables
        self.video_playing = False
        self.current_video_frame = None
        self.video_capture = None
        self.video_file_path = ""
        self.current_video_index = 0
        self.video_files = []

        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # Enable mousewheel scrolling
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.video_loop()

    def _on_mousewheel(self, event):
        self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def clear_chat_fun(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.conversation_history = []

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word2.upper()

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            if cv2image.any:
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                if hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    x, y, w, h = map['bbox']
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    white = cv2.imread("white.jpg")
                    if image.all:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz[0]:
                            hand = handz[0]
                            handmap = hand[0]
                            self.pts = handmap['lmList']
                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                         (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1),
                                     (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1),
                                     (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1),
                                     (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1),
                                     (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1),
                                     (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)

                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                            res = white
                            self.predict(res)

                            self.current_image2 = Image.fromarray(res)
                            imgtk = ImageTk.PhotoImage(image=self.current_image2)
                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk)

                            self.panel3.config(text=self.current_symbol, font=("Courier", 30))

                            self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
                            self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
                            self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)

                self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
        except Exception:
            print(Exception.__traceback__)

        self.root.after(1, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word2.upper()

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "
                print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"

        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1

        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")
(Application()).root.mainloop()