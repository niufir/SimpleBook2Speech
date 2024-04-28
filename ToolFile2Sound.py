import argparse
import os.path
import sys

import Config
import Text2SpechWrapper
from ProjectEnums import EFileOutputFormatType

parser = argparse.ArgumentParser()
def initArgParse():
    global parser
    parser = argparse.ArgumentParser(description="This is tool for make audio for large text file (Text to speach - TTS")

    parser.add_argument("--infile", type=str, default=None,
                        required = True,
                        help="Path to input text file. Support flat text format only")

    parser.add_argument("--path-out", type=str, default=None,
                        required=False,
                        help="Path to output audio file. If not provided - use input file path directory")

    speakers_voices = ','.join( Text2SpechWrapper.Text2SpechWrapper.GetAvaliableSpeakers() )
    speaker_v_default = Text2SpechWrapper.Text2SpechWrapper.getDefaultSpeakerName()

    parser.add_argument("--voice", type=str, default=speaker_v_default,
                        required=False,
                        help=f"Voice to be used for TTS. Default {speaker_v_default}. Avaliable options: {speakers_voices}")

    time_def = Text2SpechWrapper.Text2SpechWrapper.getDefaultTimePerFile()
    parser.add_argument("--time-long-sec", type=int, default=time_def,
                        required=False,
                        help=f"Approximate file time in seconds. The utility produces several files as output" +
                        f". Min value - 40 sec")

    parser.add_argument("--time-test-sample", type=int, default=None,
                        required=False,
                        help=f"Approximate test sample file time in seconds. Only a test audio file is generated. If set - -time-long-sec parameter will be ignored" +
                        f". Min value - 40 sec")

    parser.add_argument("--audio-format", type=str, default="ogg",
                        required=False,
                        help=f"Output file format. Support ogg, mp3. For export in ogg, you need install ffmpeg tools")

    return
def main():
    initArgParse()
    args = parser.parse_args()
    print(args)
    if not os.path.isfile(args.infile):
        print("Input file does not exists: " + args.infile)
        sys.exit(1)

    path_output_dir = args.path_out
    if (path_output_dir is None)or(path_output_dir == ""):
        path_output_dir = os.path.abspath(os.path.dirname(args.infile))

    audio_format = None
    try:
        print(args.audio_format)
        audio_format = EFileOutputFormatType[str(args.audio_format).upper()]
    except:
        print('Error - unsupport file format', args.audio_format)
        sys.exit(1)


    text2SpeechWrapper = Text2SpechWrapper.Text2SpechWrapper(

        path_torch_cache_nn_dir=Config.Path_Torch_Hub_Cache_Direcotry,
        timeBlockSeconds = max(args.time_long_sec,40),
        path_output_dir = path_output_dir,
        outAudioFormat = audio_format,
        timeTestSampleLong = args.time_test_sample
    )

    text2SpeechWrapper.ConvretingPipeline( args.infile )
    return

if __name__ == '__main__':
    main()