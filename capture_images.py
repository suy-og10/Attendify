import cv2
import os
import time
import json

def capture_images(person_name, roll_number, output_dir='dataset', capture_interval=2, total_images=10):
    """
    Captures images from the webcam automatically at specified intervals and saves them in a structured directory.

    Args:
        person_name (str): The name of the person whose face is being captured.
        roll_number (str): The roll number of the person.
        output_dir (str): The parent directory to store the dataset.
        capture_interval (int): Time interval between captures in seconds (default: 2).
        total_images (int): Total number of images to capture (default: 10).
    """
    # Create the directory for the person's images using name and roll number
    person_folder_name = f"{person_name}_{roll_number}"
    person_dir = os.path.join(output_dir, person_folder_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"Created directory: {person_dir}")
    else:
        print(f"Directory already exists: {person_dir}. Images will be added.")
    
    # Save person info to a JSON file
    info_file = os.path.join(person_dir, 'person_info.json')
    person_info = {
        'name': person_name,
        'roll_number': roll_number,
        'capture_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': total_images
    }
    with open(info_file, 'w') as f:
        json.dump(person_info, f, indent=2)

    # Initialize the webcam
    # '0' typically refers to the default webcam. Change this if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a window to display the webcam feed
    window_name = f"Capturing Images for {person_name} (Roll: {roll_number})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    image_count = 0
    print(f"\nStarting automatic image capture...")
    print(f"Person: {person_name} | Roll Number: {roll_number}")
    print(f"Capturing {total_images} images with {capture_interval} second intervals")
    print(f"Press 'q' to quit early\n")
    
    # Countdown before starting capture
    print("Get ready! Starting in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Starting capture!\n")
    
    start_time = time.time()
    next_capture_time = start_time

    while image_count < total_images:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If the frame was not captured successfully, break the loop
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the live frame with capture info
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Images: {image_count}/{total_images}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Next capture in: {max(0, int(next_capture_time - time.time()))}s", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, display_frame)

        # Check if it's time to capture an image
        current_time = time.time()
        if current_time >= next_capture_time:
            image_filename = os.path.join(person_dir, f"{person_name}_{roll_number}_image_{image_count:03d}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Image {image_count + 1}/{total_images} saved: {image_filename}")
            image_count += 1
            next_capture_time = current_time + capture_interval

        # Check for 'q' key press to quit early
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCapture stopped by user.")
            break
    
    if image_count >= total_images:
        print(f"\nCapture complete! {total_images} images saved for {person_name}.")

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def get_user_input():
    """
    Get user name and roll number from input.
    
    Returns:
        tuple: (name, roll_number) or (None, None) if user wants to quit
    """
    print("\n" + "="*50)
    print("FACE DATASET COLLECTION SYSTEM")
    print("="*50)
    
    while True:
        name = input("\nEnter student name (or 'quit' to exit): ").strip()
        if name.lower() == 'quit':
            return None, None
        
        if not name:
            print("Name cannot be empty. Please try again.")
            continue
            
        # Replace spaces with underscores for folder naming
        name = name.replace(" ", "_")
        
        roll_number = input("Enter roll number: ").strip()
        if not roll_number:
            print("Roll number cannot be empty. Please try again.")
            continue
            
        # Confirm the details
        print(f"\nConfirm details:")
        print(f"Name: {name}")
        print(f"Roll Number: {roll_number}")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            return name, roll_number
        else:
            print("Let's try again...")



# --- How to use the function ---
if __name__ == "__main__":
    print("Welcome to the Face Dataset Collection System!")
    print("Quick Capture Mode: 150 images with 0.5 second intervals")
    
    # Default settings for quick capture
    CAPTURE_INTERVAL = 0.5  # 0.5 seconds for quick capture
    TOTAL_IMAGES = 150      # 150 images as requested
    
    while True:
        # Get user input (only name and roll number)
        name, roll_number = get_user_input()
        
        if name is None:  # User wants to quit
            print("\nThank you for using the Face Dataset Collection System!")
            break
        
        print(f"\nPreparing to capture {TOTAL_IMAGES} images with {CAPTURE_INTERVAL} second intervals...")
        
        try:
            # Start image capture with default settings
            capture_images(name, roll_number, capture_interval=CAPTURE_INTERVAL, total_images=TOTAL_IMAGES)
        except KeyboardInterrupt:
            print("\nCapture interrupted by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        
        # Ask if user wants to continue with another person
        continue_choice = input("\nDo you want to capture images for another person? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("\nDataset collection session ended.")