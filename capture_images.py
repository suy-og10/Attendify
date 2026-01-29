import cv2
import os
import time
import json

def capture_images(person_name, PRN, output_dir='dataset', capture_interval=2, total_images=10):
    """
    Captures images from the webcam automatically at specified intervals and saves them in a structured directory.

    Args:
        person_name (str): The name of the person whose face is being captured.
        PRN (str): The PRN (Permanent Registration Number) of the person (10 digits).
        output_dir (str): The parent directory to store the dataset.
        capture_interval (int): Time interval between captures in seconds (default: 2).
        total_images (int): Total number of images to capture (default: 10).
    """
    # Extract last 4 digits of PRN for folder naming
    # Assuming PRN is always 10 digits as specified
    prn_suffix = PRN[-4:]

    # Create the directory for the person's images using name and last 4 digits of PRN
    person_folder_name = f"{person_name}_{prn_suffix}"
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
        'PRN': PRN, # Storing full PRN in info file
        'capture_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images_requested': total_images # Renamed for clarity
    }
    with open(info_file, 'w') as f:
        json.dump(person_info, f, indent=2)

    # Initialize the webcam
    cap = cv2.VideoCapture(0) # Default camera

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Check connection and ensure it's not in use.")
        return

    # Create a window to display the webcam feed
    window_name = f"Capturing Images for {person_name} (PRN: {PRN})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Use NORMAL to allow resizing

    image_count = 0
    print(f"\nStarting automatic image capture...")
    print(f"Person: {person_name} | PRN: {PRN}")
    print(f"Capturing {total_images} images with {capture_interval} second intervals")
    print(f"Saving to: {os.path.abspath(person_dir)}")
    print(f"Press 'q' in the camera window to quit early\n")

    # Countdown before starting capture
    print("Get ready! Starting in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        # Display a countdown frame? (Optional)
        ret, frame = cap.read()
        if ret:
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, str(i), (int(frame.shape[1]/2)-50, int(frame.shape[0]/2)+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            cv2.imshow(window_name, countdown_frame)
            cv2.waitKey(1000) # Wait 1 second
        else:
            time.sleep(1) # Fallback if frame read fails during countdown

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Smaller font
        cv2.putText(display_frame, f"Next capture in: {max(0, int(next_capture_time - time.time()))}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window_name, display_frame)

        # Check if it's time to capture an image
        current_time = time.time()
        if current_time >= next_capture_time:
            # Use name and suffix in image filename
            image_filename = os.path.join(person_dir, f"{person_folder_name}_image_{image_count:03d}.jpg")
            try:
                success = cv2.imwrite(image_filename, frame)
                if success:
                    print(f"Image {image_count + 1}/{total_images} saved: {os.path.basename(image_filename)}")
                    image_count += 1
                else:
                     print(f"Failed to save image {image_count + 1}.")
                     # Optionally, retry or skip? For simplicity, we just log and continue timing.
            except Exception as e:
                print(f"Error saving image {image_count + 1}: {e}")

            next_capture_time = time.time() + capture_interval # Schedule next capture based on actual save time

        # Check for 'q' key press to quit early
        key = cv2.waitKey(1) & 0xFF # Wait only 1ms
        if key == ord('q'):
            print("\nCapture stopped by user.")
            break

    if image_count >= total_images:
        print(f"\nCapture complete! {image_count} images saved for {person_name}.") # Use actual saved count
    else:
        print(f"\nCapture ended early. {image_count} images saved.")

    # Update total images captured in JSON file
    person_info['total_images_captured'] = image_count
    person_info['capture_end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(info_file, 'w') as f:
            json.dump(person_info, f, indent=2)
    except Exception as e:
        print(f"Error updating info file: {e}")

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Add a small delay to ensure window closes completely on some systems
    cv2.waitKey(1)


def get_user_input():
    """
    Get user name and PRN from input.

    Returns:
        tuple: (name, PRN) or (None, None) if user wants to quit
    """
    print("\n" + "="*50)
    print("FACE DATASET COLLECTION SYSTEM")
    print("="*50)

    while True:
        name_input = input("\nEnter student name (or 'quit' to exit): ").strip()
        if name_input.lower() == 'quit':
            return None, None

        if not name_input:
            print("Name cannot be empty. Please try again.")
            continue

        # Replace spaces with underscores for folder naming convention
        name = name_input.replace(" ", "_")

        PRN = input("Enter 10-digit PRN: ").strip()
        if not PRN.isdigit() or len(PRN) != 10:
            print("PRN must be exactly 10 digits. Please try again.")
            continue

        # Confirm the details
        print(f"\nConfirm details:")
        print(f"Name (for folder): {name}")
        print(f"PRN: {PRN}")
        confirm = input("Is this correct? (y/n): ").strip().lower()

        if confirm in ['y', 'yes']:
            return name, PRN
        else:
            print("Let's try again...")


# --- Main Execution Logic ---
if __name__ == "__main__":
    print("Welcome to the Face Dataset Collection System!")

    # Prompt for settings or use defaults
    try:
         # Use defaults for quicker testing, uncomment to ask user
        # CAPTURE_INTERVAL = float(input("Enter capture interval in seconds (e.g., 0.5): ") or "0.5")
        # TOTAL_IMAGES = int(input("Enter total number of images to capture (e.g., 100): ") or "100")
        CAPTURE_INTERVAL = 0.5
        TOTAL_IMAGES = 100
        print(f"Using settings: {TOTAL_IMAGES} images, {CAPTURE_INTERVAL}s interval.")

    except ValueError:
        print("Invalid input for interval or total images. Using defaults: 0.5s, 100 images.")
        CAPTURE_INTERVAL = 0.5
        TOTAL_IMAGES = 100

    output_directory = 'dataset' # Define the base dataset directory

    while True:
        # Get user input
        name, PRN = get_user_input()

        if name is None:  # User entered 'quit'
            break

        print(f"\nPreparing to capture {TOTAL_IMAGES} images for {name}...")

        try:
            # Start image capture
            capture_images(
                person_name=name,
                PRN=PRN,
                output_dir=output_directory,
                capture_interval=CAPTURE_INTERVAL,
                total_images=TOTAL_IMAGES
            )
        except KeyboardInterrupt:
            print("\nCapture interrupted by user (Ctrl+C).")
            # Ensure resources are released if interrupted mid-capture
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            print(f"\nAn error occurred during capture: {e}")
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        # Ask if user wants to continue with another person
        continue_choice = input("\nCapture images for another person? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

    print("\nDataset collection session ended.")