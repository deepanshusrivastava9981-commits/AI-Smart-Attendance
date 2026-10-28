🤖 AI Smart Attendance System

An AI-powered attendance management system that uses face recognition to identify registered individuals and record their attendance.

🚀 Project Overview

The AI Smart Attendance System is developed using Python, OpenCV, Face Recognition, and Streamlit. It aims to simplify attendance tracking by using facial recognition instead of manual attendance.

✨ Features

- Face recognition for registered individuals
- Automated attendance recording
- Attendance data storage in CSV format
- Streamlit-based user interface
- Webcam integration for face detection and recognition

🛠️ Technologies Used

- Python
- OpenCV
- Face Recognition
- Streamlit
- Pandas
- CSV

📂 Project Structure

AI-Smart-Attendance/
├── attendance_app.py
├── attendance.csv
├── known_faces/
├── README.md
└── .gitignore

Additional test and backup files may also be present in the project directory.

⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/deepanshusrivastava9981-commits/AI-Smart-Attendance.git

2. Open the project directory

cd AI-Smart-Attendance

3. Create a virtual environment

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

4. Install the required packages

Install the dependencies required by the project:

pip install streamlit opencv-python face-recognition pandas

Note: The "face-recognition" package may require additional dependencies, including "dlib", and may need specific Windows build tools.

5. Run the application

streamlit run attendance_app.py

🔒 Privacy and Security

- Use only authorized face images.
- Do not publish personal photographs or real attendance records.
- Keep private or sensitive information out of the public repository.

👨‍💻 Author

Deepanshu Srivastava

B.Tech Computer Science — Data Science & Artificial Intelligence

🎯 Project Goal

To explore the practical application of computer vision and AI in attendance management.