import cv2
import base64
import time
import argparse
from llama_cpp import Llama

# -------------------------
# Load the LLM + vision model
# -------------------------
def load_model(model_path, mmproj_path):
    """
    Load the multimodal LLaMA (SmolVLM) model.

    Args:
        model_path (str): Path to the GGUF model file.
        mmproj_path (str): Path to the multimodal projection file (mmproj).

    Returns:
        Llama: An initialized Llama object ready for inference.
    """
    print("Loading model...")
    return Llama(
        model_path=model_path,   # Path to main LLM weights (GGUF)
        mmproj=mmproj_path,      # Path to multimodal projection weights
        n_ctx=4096,              # Context window size (tokens)
        n_gpu_layers=-1,         # Offload all layers to GPU (-1 = all, 0 = CPU only)
        verbose=False,           # Suppress debug logs
    )

# -------------------------
# Convert OpenCV frame → base64 string
# -------------------------
def frame_to_base64(frame):
    """
    Convert an OpenCV image frame into a base64-encoded JPEG string.

    Args:
        frame (numpy.ndarray): Frame captured from the webcam.

    Returns:
        str: Base64-encoded JPEG image wrapped as a data URI.
    """
    # Encode frame as JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert JPEG bytes → base64 string
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    # Return in "data URI" format (so model can read it as image_url)
    return f"data:image/jpeg;base64,{jpg_as_text}"

# -------------------------
# Main application loop
# -------------------------
def run_app(model_path, mmproj_path, interval=3):
    """
    Run the realtime webcam + vision-language model pipeline.

    This function:
      - Loads the model.
      - Opens the webcam feed.
      - Captures frames at a regular interval.
      - Sends frames + prompt to the model for description.
      - Displays the live webcam feed until user quits.

    Args:
        model_path (str): Path to the GGUF model file.
        mmproj_path (str): Path to the mmproj vision file.
        interval (int): Seconds between model inferences.
    """
    # Load model once at start
    llm = load_model(model_path, mmproj_path)

    # Initialize webcam (device 0 = default camera)
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Track when to process next frame
    next_process_time = time.time()

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Show webcam feed in a window
            cv2.imshow("Webcam Feed (press 'q' to quit)", frame)

            # Check if it's time to run inference
            current_time = time.time()
            if current_time >= next_process_time:
                print("\nProcessing new frame...")
                next_process_time = current_time + interval  # Set time for next frame

                # Convert frame → base64 image string
                image_url = frame_to_base64(frame)

                # Build chat messages in LLaVA-style format
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who describes images in detail."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},  # Send image
                            {"type": "text", "text": "What is in this image?"}       # Send prompt
                        ]
                    }
                ]

                # Run local inference (no streaming)
                response = llm.create_chat_completion(messages=messages, stream=False)

                # Print model’s reply
                print("Model Response:")
                print(response['choices'][0]['message']['content'])

            # Allow quitting with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping application.")
    finally:
        # Always release camera + close windows
        cap.release()
        cv2.destroyAllWindows()

# -------------------------
# Entry point: parse arguments
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime webcam + SmolVLM inference")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--mmproj", required=True, help="Path to mmproj vision file")
    parser.add_argument("--interval", type=int, default=3, help="Seconds between inferences")
    args = parser.parse_args()

    # Run main loop with user-provided arguments
    run_app(args.model, args.mmproj, args.interval)
