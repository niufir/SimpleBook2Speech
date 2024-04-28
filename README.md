# SimpleBook2SpeachPipeline
## Description
Utility for converting large text to audio.

Utilitarian project wrapper
https://github.com/snakers4/silero-models

Converting large texts to audio has been implemented (text 2 speech, noise removal, sound power reduction), with possible splitting into several files.
## Features
Supported languages from the command line: Russian.

Input text file format - simple txt.

Using CPU/GPU - tested on CPU.

Output file parameters - mp3 or ogg


## Running
Run:

    python ToolFile2Sound.py -h
    
or

    python ToolFile2Sound.py --infile "\book\mybook.txt"
