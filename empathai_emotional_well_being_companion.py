import os
import numpy as np
import gradio as gr
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

try:
    import cv2S
    from transformers import pipeline
except ImportError:
    print("Installing required packages... This may take a moment.")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "transformers", "torch", "opencv-python", "Pillow", "timm", "gradio"])
    import cv2
    from transformers import pipeline

print(f"Gradio version: {gr.__version__}")

# --- Global variables for session state ---
session_history = []
current_mood = "neutral"
conversation_history = []

# ======== AI MODEL INITIALIZATION ========
# This is done once when the application starts to avoid slow loading times later.

# --- NLP Text Model ---
print("Loading NLP model for text analysis... This may take a few minutes on first run.")
text_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
print("NLP model loaded successfully.")

# --- Vision Transformer (ViT) Model for Images ---
print("Loading Vision Transformer (ViT) model for facial emotion recognition...")
# FIXED: Replaced the unavailable model with another stable alternative.
vision_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
print("ViT model loaded successfully.")


# ======== CORE ANALYSIS FUNCTIONS ========

def analyze_text_sentiment_with_nlp(text):
    """Analyzes text emotion using a Hugging Face NLP model."""
    if not text or not text.strip():
        return "neutral"
    try:
        # Get all emotion scores from the classifier
        results = text_classifier(text)[0]
        # Find the emotion with the highest score
        dominant_emotion_data = max(results, key=lambda x: x['score'])
        dominant_emotion_label = dominant_emotion_data['label']

        # Map the model's detailed 28 emotions to the app's 7 simpler categories
        emotion_map = {
            'admiration': 'happy', 'amusement': 'happy', 'approval': 'happy', 'caring': 'happy',
            'desire': 'happy', 'excitement': 'happy', 'gratitude': 'happy', 'joy': 'happy',
            'love': 'happy', 'optimism': 'happy', 'pride': 'happy', 'relief': 'happy',
            'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry',
            'disappointment': 'sad', 'disgust': 'upset', 'embarrassment': 'upset', 'grief': 'sad',
            'remorse': 'sad', 'sadness': 'sad',
            'fear': 'anxious', 'nervousness': 'anxious',
            'surprise': 'surprised', 'neutral': 'neutral',
            'realization': 'neutral', 'confusion': 'upset', 'curiosity': 'neutral'
        }
        return emotion_map.get(dominant_emotion_label, 'neutral')
    except Exception as e:
        print(f"Error in text analysis: {e}")
        return "neutral"

def analyze_face_emotion_with_vit(image_np):
    """Detects a face and analyzes its emotion using a Vision Transformer."""
    if image_np is None:
        return "neutral", None

    # Load OpenCV's face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    detected_emotion = "neutral"
    output_image = image_np.copy()

    if len(faces) > 0:
        # Process the largest detected face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

        # Crop the face from the image
        face_roi_np = image_np[y:y+h, x:x+w]
        # Convert to PIL Image, which the ViT model expects
        face_pil = Image.fromarray(face_roi_np)

        # Use the ViT model to classify the emotion of the face
        results = vision_classifier(face_pil)
        dominant_emotion_data = max(results, key=lambda x: x['score'])
        detected_emotion = dominant_emotion_data['label'].lower()

        # Map the model's output labels to our app's categories
        emotion_map = {'fear': 'anxious', 'disgust': 'upset'}
        detected_emotion = emotion_map.get(detected_emotion, detected_emotion)

        # Draw a rectangle and label on the output image for visualization
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (36, 255, 12), 2)
        cv2.putText(output_image, f"Emotion: {detected_emotion.capitalize()}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    return detected_emotion, output_image


# ======== CHATBOT AND WELL-BEING FUNCTIONS ========

def generate_companion_response(user_input, emotion):
    """Generates an empathetic response based on the detected emotion."""
    response_templates = {
        "happy": [
            "That's wonderful to hear! What's bringing you joy today?",
            "I'm so glad you're feeling happy. Your positive energy is contagious!",
            "It's great to see you in good spirits. Let's cherish this feeling."
        ],
        "sad": [
            "I'm sorry you're feeling this way. Remember that it's okay to be sad. I'm here to listen.",
            "It sounds like things are tough right now. Would you like to talk about what's on your mind?",
            "I'm here for you. Sometimes just sharing what we're going through can make a difference."
        ],
        "anxious": [
            "I understand that anxiety can be overwhelming. Let's try to focus on the present moment together.",
            "Take a slow, deep breath. Inhale for 4, hold for 4, exhale for 6. You're safe here.",
            "Anxious feelings are temporary, even if they don't feel that way. What's one small thing we can focus on right now?"
        ],
        "angry": [
            "It's completely valid to feel angry. What's causing this frustration?",
            "Feeling angry can be draining. Is there a way we can channel this energy constructively?",
            "I'm here to listen without judgment. Feel free to vent if you need to."
        ],
        "neutral": [
            "Thanks for checking in. How's your day going so far?",
            "I'm here and ready to chat. What's on your mind?",
            "Is there anything you'd like to talk about or explore today?"
        ],
        "surprised": [
            "Oh! That sounds unexpected. How are you feeling about it?",
            "Surprises can be a lot to take in. Take your time to process.",
            "Wow, what happened? I'm curious to hear more."
        ],
        "upset": [
            "It sounds like you're upset, and that's okay. Your feelings are valid.",
            "I'm sorry you're going through this. I'm here to support you.",
            "Would it help to write down what's making you feel upset?"
        ]
    }
    return np.random.choice(response_templates.get(emotion, response_templates["neutral"]))

def get_wellbeing_activities(emotion):
    """Suggests wellbeing activities based on the detected emotion."""
    suggestions = {
        "sad": ["Listen to a favorite uplifting song.", "Step outside for 5 minutes of fresh air.", "Write down three things you are grateful for."],
        "anxious": ["Try the 5-4-3-2-1 grounding technique.", "Hold a piece of ice to focus your senses.", "Do a 10-minute guided meditation."],
        "angry": ["Engage in some vigorous exercise.", "Scribble on a piece of paper and then tear it up.", "Listen to calming, slow-tempo music."],
        "happy": ["Share your good news with a friend.", "Write down what's making you happy to savor it.", "Do a small act of kindness for someone else."],
        "neutral": ["Take a moment to stretch your body.", "Drink a full glass of water.", "Think about one goal for the rest of your day."],
        "upset": ["Wrap yourself in a warm blanket.", "Make a soothing cup of tea or hot chocolate.", "Watch a comforting or funny short video."]
    }
    # Add a default for 'surprised' or any other unlisted emotion
    return suggestions.get(emotion, suggestions["neutral"])


# ======== SESSION TRACKING & VISUALIZATION ========

def track_mood(emotion, timestamp):
    """Adds the current mood and timestamp to the session history."""
    global session_history
    session_history.append({"timestamp": timestamp, "emotion": emotion})

def generate_mood_chart():
    """Generates a matplotlib chart of mood over the session."""
    global session_history
    if len(session_history) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data yet.\nKeep interacting to track your journey!", ha='center', va='center')
        ax.axis('off')
        return fig

    emotion_map = {"happy": 5, "surprised": 4, "neutral": 3, "anxious": 2, "upset": 1.5, "angry": 1, "sad": 0}
    timestamps = [entry["timestamp"] for entry in session_history]
    numeric_emotions = [emotion_map.get(entry["emotion"], 3) for entry in session_history]
    start_time = timestamps[0]
    elapsed_minutes = [(t - start_time).total_seconds() / 60 for t in timestamps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(elapsed_minutes, numeric_emotions, 'o-', color='mediumseagreen', markersize=8)
    ax.set_yticks(list(emotion_map.values()))
    ax.set_yticklabels([key.capitalize() for key in emotion_map.keys()])
    ax.set_xlabel('Time Elapsed (minutes)')
    ax.set_ylabel('Emotional State')
    ax.set_title('Your Emotional Journey This Session')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig


# ======== GRADIO INTERFACE FUNCTIONS ========

def reset_conversation():
    """Resets all global state and clears the UI components."""
    global conversation_history, session_history, current_mood
    conversation_history, session_history = [], []
    current_mood = "neutral"
    # Returns a default value for each UI component that needs to be cleared
    return "", [], None, "😐 Neutral", "Suggestions will appear here.", None

def chat_interface_response(message, image, history):
    """
    Main function to process user input (text and image), orchestrate AI analysis,
    and generate a response for the Gradio UI.
    """
    global current_mood, conversation_history

    # If there's no new input, do nothing.
    if (not message or not message.strip()) and image is None:
        return "", history, None

    # 1. Analyze emotions from both text and face
    text_emotion = analyze_text_sentiment_with_nlp(message)
    visual_emotion, processed_image = analyze_face_emotion_with_vit(image)

    # 2. Combine emotions: Prioritize a clearly detected facial emotion over text.
    if visual_emotion != "neutral":
        current_mood = visual_emotion
    else:
        current_mood = text_emotion

    # 3. Track the determined mood
    track_mood(current_mood, datetime.now())

    # 4. Generate AI response only if there is a text message
    ai_response = ""
    if message and message.strip():
        conversation_history.append({"role": "user", "content": message})
        ai_response = generate_companion_response(message, current_mood)
        conversation_history.append({"role": "assistant", "content": ai_response})
        history.append((message, ai_response))

    # 5. Return updated values for the UI
    return "", history, processed_image

def update_ui_after_response():
    """Returns updated mood display and suggestions for the UI."""
    global current_mood
    mood_emoji = {
        "happy": "😊 Happy", "sad": "😢 Sad", "anxious": "😰 Anxious",
        "angry": "😠 Angry", "neutral": "😐 Neutral", "surprised": "😮 Surprised",
        "upset": "😟 Upset"
    }
    mood_display_text = mood_emoji.get(current_mood, "😐 Neutral")
    
    suggestions = get_wellbeing_activities(current_mood)
    suggestion_text = "\n• " + "\n• ".join(suggestions)
    
    return mood_display_text, suggestion_text

# ======== MAIN APPLICATION LAYOUT ========
def main():
    """Defines and launches the Gradio web application."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="emerald"), css=".gradio-container {background-color: #F0F4F8;}") as demo:
        gr.Markdown("# 🌟 EmpathAI: Vision-Enhanced Emotional Companion")
        gr.Markdown("I'm here to listen. Share what's on your mind or simply let the camera see how you're doing. I'll do my best to support you.")

        with gr.Row():
            with gr.Column(scale=2):
                webcam = gr.Image(type="numpy", sources=["webcam"], label="Your Webcam Feed", height=400)
                chatbot = gr.Chatbot(label="Conversation", height=300)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4)
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                mood_display = gr.Textbox(label="Current Detected Mood", value="😐 Neutral", interactive=False)
                suggestion_box = gr.Textbox(label="Well-being Suggestions", lines=5, interactive=False, value="Suggestions will appear here.")
                get_suggestions_btn = gr.Button("Get New Suggestions")
                with gr.Accordion("Your Emotion Journey", open=False):
                    mood_chart = gr.Plot(label="Emotion Over Time")
                    mood_chart_btn = gr.Button("Update Journey Chart")
                clear_btn = gr.Button("🔄 Reset Session")

        # --- Define event handlers for UI interactions ---
        all_inputs = [msg, webcam, chatbot]
        all_outputs = [msg, chatbot, webcam]
        
        # Combined event for submitting text (click or enter)
        submit_event = submit_btn.click(chat_interface_response, inputs=all_inputs, outputs=all_outputs)
        msg.submit(chat_interface_response, inputs=all_inputs, outputs=all_outputs)

        # Chain UI updates to happen *after* the main response is processed
        submit_event.then(update_ui_after_response, outputs=[mood_display, suggestion_box])
        msg.submit(update_ui_after_response, outputs=[mood_display, suggestion_box])

        # A separate event for when only the webcam image changes
        webcam.change(chat_interface_response, inputs=all_inputs, outputs=all_outputs).then(
            update_ui_after_response, outputs=[mood_display, suggestion_box])
        
        # Buttons for other features
        get_suggestions_btn.click(update_ui_after_response, outputs=[mood_display, suggestion_box])
        mood_chart_btn.click(generate_mood_chart, outputs=[mood_chart])
        clear_btn.click(reset_conversation, outputs=[msg, chatbot, webcam, mood_display, suggestion_box, mood_chart])
        
        gr.Examples(
            [
                ["I had a fantastic day today!"],
                ["I'm really worried about my exam tomorrow."],
                ["I got into an argument with my friend and I feel terrible."],
                ["Nothing much is happening, just a regular day."]
            ],
            inputs=[msg],
            label="Example Conversation Starters"
        )

    demo.launch(debug=True)

if __name__ == "__main__":
    main()

