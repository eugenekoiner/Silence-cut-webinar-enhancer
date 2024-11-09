# Silence-Cut Webinar Enhancer

This Python script was created out of frustration with the time wasted during long video lectures or webinars. I often found myself spending hours watching videos where there were long pauses or slow-paced sections. The idea behind this script is simple — **remove silences and speed up the video** to save time.

By trimming out unnecessary pauses and speeding up the video (at least 1.25x), the script can reduce the overall length of the video by 30-50%, with a 45% reduction being more common in my tests (almost cutting the duration in half!). This makes it ideal for long educational videos, tutorials, or webinars where the content is usually delivered at a slow pace.
## Features

- **Silence Removal**: Automatically detects and removes silent sections from videos.
- **Speed Adjustment**: Optionally speeds up the video for more engaging content.
- **Chunking for Long Videos**: Splits videos longer than 20 minutes into smaller chunks for better silence detection and transcription.
- **Transcription Support**: Optionally transcribes video audio using Whisper for subtitles or text search.**
- **Customizable**: Easily adjust settings like speed, silence threshold, and transcription via a configuration file.
- **Resumption of Interrupted Process**: If the process is interrupted, the script can resume from where it left off.
- **Automatic Cache Management**: When processing a new file, the script clears the cache from previous runs (for resumption). Cache can be kept if you want to use it for further processing, such as adding transcription.
- **Easy Setup**: Automatically creates necessary directories and configuration file on first run.

## How it Works

For videos longer than 20 minutes, the script uses ffmpeg to detect the longest silent sections and splits the video into chunks around these silences, aiming for approximately 15-minute segments. This helps with better quality for subsequent contextual transcriptions and silence removal, as longer pauses typically indicate disconnected segments. For videos shorter than 20 minutes, the script processes them as a whole without splitting. In both cases, the script removes silence from each chunk, optionally transcribes the audio using Whisper for subtitles or text search, speeds up the video, and then merges everything into a single file. The result is a much shorter, faster, and more engaging video.
<details>
  <summary>Here’s a step-by-step overview of how the process works:</summary>
  
1. **Video Chunking**: 
   - For videos longer then 20 minutes the script first analyzes the source video and looks for the longest periods of silence. 
   - It tries to split the video into smaller chunks (around **15 minutes** each). This chunking is done smartly by analyzing silent gaps and avoiding cutting in the middle of active speech.

2. **Threshold Calculation**: 
   - For each chunk, the script calculates the **silence threshold** independently. It uses `ffmpeg` to analyze the audio in each chunk, determining the periods of silence based on the set threshold. The threshold can be adjusted using the `offset_dB` parameter, which fine-tunes the detection of silence by setting the sensitivity of the audio analysis.
   - The script also takes into account the **minimum silence gap** (`silence_gap`), which specifies the shortest duration of silence that will **not** be removed. This ensures that brief pauses, which may occur naturally between words or phrases, are preserved. Any silence longer than this threshold will be removed from the video, ensuring that only meaningful content remains.


3. **Silence Removal**:
   - After detecting the silent sections in each chunk, the script uses `ffmpeg` to **trim** these silences. The result is a chunk of the video with the silence removed, keeping only the important, active speech.
   
4. **Optional Transcription with Whisper**:  
	- If needed, each chunk of the video can be transcribed into text using **Whisper**, an AI-powered transcription model by OpenAI. The script utilizes your **NVIDIA GPU** for fast transcription (or your CPU if you don’t have a GPU, though it will take longer). You can choose the specific language and transcription model for this step, allowing you to generate accurate transcriptions for each segment of the video. This feature is useful if you want to generate a **subtitle file** from the video, or if you plan to perform **text-based search** through the video content.
5.  **Merging Subtitle Segments**:
   	- Once Whisper completes processing each segment, all transcriptions are merged into a single SRT file in the `.output` folder. The timing in the final SRT file is adjusted according to the previously set video speed factor.

6. **Merging Video Segments**:  
	- Once silence is removed and transcription (if enabled) is completed, the script automatically merges the individual video segments into one intermediate file.

7. **Speed Adjustment**:  
	- After processing the video, the script provides an option to apply a speed adjustment, with a default of 1.25x. This adjustment is optional: you can keep the default 1.25x speed or set the speed_factor parameter to 1 to retain the original speed. This is the final step, and the completed video will be saved in the .output folder.

</details>

## Installation

1. Clone the repository:

   `git clone https://github.com/eugenekoiner/Silence-cut-webinar-enhancer.git`

2. Install dependencies (automatically handled by the script in most cases).  
   - **ffmpeg**: Required for video processing. If not installed, download and install [ffmpeg](https://ffmpeg.org/download.html).  
   - **Whisper (optional)**: If you plan to use transcription, you’ll need to install CUDA for GPU support. Follow [this guide](https://pytorch.org/get-started/locally/) to set up CUDA.

3. Use the script by running:

   `python webinar_enhancer.py`

   Follow the instructions in the script to configure the settings on the first run.


## How to use and customize

When you run the script for the first time, it will not find a configuration file and will prompt you to create one by asking a series of questions in the console. These questions allow you to customize various settings, such as the video speed, silence threshold, and whether transcription is required. 

You can adjust the following parameters:
- **Speed Adjustment**: Set the desired speed factor for the video (default is 1.25 for a slight speedup, set to 1 if no speed adjustment is needed).
- **Silence Threshold Offset**: Adjust the dB level for detecting silence (default is 1).
- **Min Silence Gap**: Set the minimum allowed duration of silence before it's removed (default is 1).
- **Final Video Bitrate**: Define the bitrate for the final video (default is 300k).
- **Transcription**: Choose whether you want transcription (subtitles). Set to `yes` to enable.
- **Language Settings**: Set the source language for transcription and choose a Whisper model if needed.

## Good to know:

- After answering these questions, the script will create a configuration file in the `.config` directory with your choices.
- In the future, if you wish to change any settings, you can simply edit this configuration file manually.
- If you leave a question blank, the script will use the default value for that parameter. 
- Default values work well for most cases and it's not recommended to change them unless you're sure about what you're doing.
- The resolution is taken directly from the original video. Typically, I use 720p sources as it provides enough clarity for tutorials while keeping file sizes manageable. If needed, I can make resolution a configurable parameter, but for most cases, 720p is sufficient to balance detail and storage efficiency.
- A 300k bitrate is sufficient for 720p video to ensure clear visibility, even in coding tutorials, while keeping the file size small.
- The source videos should be placed in the `.source` folder, and all necessary directories will be automatically created during the first run of the script.
- The Silence Threshold Offset is set to 1 by default, which means the calculated silence threshold for each video segment is raised by a value of 1. This results in slightly shorter silent sections being removed between words, but it keeps the beginnings and ends of words more intact. Through experimentation, a value of 1 proved optimal; reducing the offset causes words to merge together.
- By default, the video speed is increased by 1.25x, based on the observation that speakers in many webinars tend to speak slowly with long pauses. This slight speed boost helps bring the video closer to a more natural speaking pace, without impacting comprehension.
- The script uses multithreaded processing wherever possible.
- Sometimes, certain video files are processed oddly by ffmpeg: after splitting a long file into segments and then merging them back together, there may be slight audio sync issues or a single frame freeze at certain points. This only lasts within one segment and resolves in the next. It’s a rare issue and generally doesn’t disrupt the overall video, though I haven’t identified the exact cause.



## Disclaimer

I am not a Python developer, so the script is probably not written in the best way, but it works well for my use case and should be helpful for anyone looking to speed up long videos or lectures.  
Feel free to contribute, report issues, or suggest improvements!
