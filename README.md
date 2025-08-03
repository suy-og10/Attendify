# AMS - Attendance Management System

A face dataset collection system for building an attendance management system using computer vision.

## Features

- **Automated Image Capture**: Captures images at specified time intervals
- **User Input System**: Prompts for student name and roll number
- **Organized Storage**: Creates structured directories with metadata
- **Real-time Feedback**: Shows capture progress and countdown
- **Batch Processing**: Support for multiple students in one session

## Requirements

- Python 3.7+
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AMS.git
cd AMS
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Image Capture

Run the image capture script:
```bash
python capture_images.py
```

The system will prompt you for:
- Student name
- Roll number
- Capture interval (seconds between images)
- Total number of images to capture

### Dataset Structure

Images are organized as follows:
```
dataset/
├── StudentName_RollNumber/
│   ├── person_info.json
│   ├── StudentName_RollNumber_image_001.jpg
│   ├── StudentName_RollNumber_image_002.jpg
│   └── ...
```

## Controls

- **Press 'q'**: Quit capture early
- **Automatic capture**: Images are captured at specified intervals
- **3-second countdown**: Preparation time before capture starts

## Project Structure

```
AMS/
├── capture_images.py    # Main image capture script
├── photos.py           # Legacy capture script
├── requirements.txt    # Python dependencies
├── dataset/           # Captured images storage
└── README.md          # This file
```

## Future Enhancements

- Face recognition model training
- Web-based interface using Flask
- Attendance tracking system
- Database integration
- Real-time attendance monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
