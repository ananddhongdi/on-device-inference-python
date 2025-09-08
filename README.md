# on-device-inference-python

Python application that uses VLMs for real-time inference. The application captures frames from a laptop's webcam, processes them, and generates visual descriptions or performs inference tasks.

## Features
- Captures frames from the webcam in real-time.
- On-device inference.

## Requirements
- Python 3.8 or higher
- llama_cpp
- OpenCV (`cv2`)
- model files

## How to setup
1. Install llama.cpp:
   ```
   pip install llama-cpp-python
   ```

2. Install opencv:
    ```
    pip install opencv-python
    ```
   
3. Clone the repository:
   ```bash
   git clone https://github.com/your-username/on-device-inference-python.git
   cd on-device-inference-python
   ```
4. Download model and mmproj file locally on the device.
    
    _Example_:
    
    SmolVLM-500M-Instruct-GGUF: https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF
    
## Usage
1. Run python application with 3 arguments:

     _argument-1_: model file
    
     _argument-2_: mmproj file
    
     _argument-3_: interval in seconds
    
    ```python
    python on_device_vlm.py --model argument-1 --mmproj argument-2 --interval argument-3
    ``` 
    __Example__:
    ```
    python on_device_vlm.py --model SmolVLM.gguf --mmproj mmproj.gguf --interval 3
    ```

3. The application will capture frames from the webcam and start on-device inference.

    > **Note:**
    > The application will open a new window with a live camera preview, which will be closed only after the Python application is closed.

### Example Output
The python application process a frame and instruction: `What do you see?`

_The application responds with a description of the frame._
> A man holds up a white coffee mug in front of his face
