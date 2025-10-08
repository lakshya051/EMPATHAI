🌟 EmpathAI: Vision-Enhanced Emotional Companion
Your personal AI-powered space to understand, track, and navigate your emotions.

EmpathAI is a multi-modal emotional well-being application that combines the power of Natural Language Processing (NLP) and Computer Vision to provide a supportive and interactive experience. It listens to what you say, sees how you feel, and offers gentle guidance to help you through your day.

✨ Key Features
Multi-Modal Emotion Detection:

Text Analysis: Utilizes a state-of-the-art Hugging Face Transformer model (roberta-base-go_emotions) to understand the nuanced emotions in your written text.

Facial Emotion Recognition: Employs a Vision Transformer (ViT) model (trpakov/vit-face-expression) to detect emotions like happiness, sadness, and anger in real-time through your webcam.

Empathetic Chatbot:

Engages in conversation with responses tailored to your currently detected emotional state.

Personalized Well-being Toolkit:

Receives actionable, context-aware suggestions (e.g., breathing exercises for anxiety, gratitude prompts for happiness) to help you manage your feelings.

Emotional Journey Visualization:

Tracks your mood throughout a session and generates a visual chart, helping you recognize emotional patterns over time.

Intuitive Web Interface:

A clean, simple, and responsive user interface built with Gradio that runs right in your browser.

🛠️ Tech Stack
This project is built with a modern, AI-focused Python stack:

Backend & Logic: Python

Web UI: Gradio

AI & Machine Learning:

Hugging Face Transformers for accessing pre-trained models.

PyTorch as the deep learning framework.

Computer Vision:

OpenCV for initial face detection from the webcam feed.

Pillow (PIL) for image manipulation.

Data Handling & Visualization:

NumPy for numerical operations on images.

Matplotlib for generating the mood chart.

🚀 Getting Started
Follow these steps to get EmpathAI running on your local machine.

Prerequisites
Python 3.8 or higher.

pip package manager.

A webcam (for the facial emotion recognition feature).

Installation
Clone the repository:

git clone [https://github.com/your-username/EmpathAI-Your-Emotional-Well-being-Companion-main.git](https://github.com/your-username/EmpathAI-Your-Emotional-Well-being-Companion-main.git)
cd EmpathAI-Your-Emotional-Well-being-Companion-main

Create a virtual environment (recommended):

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required dependencies:
The project uses several powerful libraries. You can install them all with the following command:

pip install -r requirements.txt

(Note: If a requirements.txt is not available, you can install the packages directly: pip install gradio transformers torch opencv-python Pillow timm matplotlib)

Running the Application
Execute the main script:

python empathai_app.py

Wait for the models to load:
The first time you run the app, it will download the NLP and Vision Transformer models. This may take a few minutes and requires an internet connection. Subsequent launches will be much faster.

Open the app:
Once the models are loaded, you will see a message in your terminal like:
Running on local URL: http://127.0.0.1:7860
Open this URL in your web browser to start using EmpathAI.

📖 How to Use
Enable Your Webcam: Allow your browser to access your webcam when prompted. You should see your video feed in the top-left panel.

Start Chatting: Type a message about how you're feeling into the textbox and press Send or hit Enter.

See the Magic:

The chatbot will respond in the conversation window.

The "Current Detected Mood" will update based on your face and/or your text.

The "Well-being Suggestions" box will provide relevant tips.

A green box will appear around your face with the detected emotion.

Track Your Journey: After a few interactions, click "Update Journey Chart" to see a graph of your emotional state over the session.

Reset: Click "Reset Session" at any time to start fresh.

🔮 Future Enhancements
This project has a solid foundation with many possibilities for future growth:

[ ] Integrate a Generative LLM: Replace the template-based chatbot with a powerful Large Language Model (like Gemini or an open-source alternative) for more dynamic, context-aware conversations.

[ ] Persistent User History: Save mood and conversation history locally, allowing users to track their emotional journey over days, weeks, and months.

[ ] Interactive Well-being Modules: Build guided exercises (e.g., timed breathing, journaling prompts) directly into the interface.

[ ] Mobile Application: Develop a cross-platform mobile app using a framework like React Native for on-the-go support.

[ ] Disclaimer & Resources: Add a clearer disclaimer and a dedicated section with links to professional mental health resources.

🤝 Contributing
Contributions are welcome! If you have ideas for new features, bug fixes, or improvements, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourAmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/YourAmazingFeature).

Open a Pull Request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
