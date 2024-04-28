import typing
from io import BytesIO
from itertools import islice
import numpy as np
from pydub import AudioSegment
import random
import string
import shutil
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import tqdm

from ProjectEnums import *
import torch
import os
import  Config
from nltk.tokenize import sent_tokenize
from tqdm import notebook as tqmdnotebook
from scipy.io.wavfile import write as wav_write
import warnings

class Text2SpechWrapper:

    TIMEOUT_BREAK_HEADER = 2000
    POWER_KOEFF = 0.93
    POWER_KOEFF_AFTER_DENOISE = 0.5
    COUNT_DEFAULT_BLOCK2FILE = 20
    SPEACKER_DEFAULT =  ESpeakerId.SpeakerBaya
    TIME_PER_FILE_SEC = 20*60
    def __init__(self,
                 path_torch_cache_nn_dir:str = None,
                 path_output_dir:str = None,
                 lang:LanguageType = LanguageType.RU,
                 denoiseModelType:EDenoiseModelType = EDenoiseModelType.LargeFast,
                 put_accent:bool = True,
                 put_yo:bool = True,
                 device:torch.device = torch.device('cpu'),
                 isDebug:bool = False,
                 pathTmpDir:str = None,
                 speakerVoice: ESpeakerId = SPEACKER_DEFAULT,
                 timeBlockSeconds:int = TIME_PER_FILE_SEC,
                 outAudioFormat:EFileOutputFormatType = EFileOutputFormatType.OGG,
                 timeTestSampleLong:int = None, # if set, generate sample file only, and stop generation
                 ):

        print('Start Init')

        self.torch_device:torch.device = device
        self.lang:LanguageType =  lang
        self.speaker:ENNModelType = None
        if self.lang == LanguageType.RU:
            self.speaker = ENNModelType.Speaker4Ru
        self.path_cache_nn_dir:str = path_torch_cache_nn_dir
        self.timeBlockSeconds = timeBlockSeconds
        self.fileFormatOut = outAudioFormat
        self.timeTestSampleLong = timeTestSampleLong

        self.path_output_dir:str = path_output_dir
        if (self.path_output_dir is None):
            self.path_output_dir = os.getcwd();

        self.speakerVoice = speakerVoice
        self.isDebug = isDebug
        print('Use debug mode ', self.isDebug)

        self.model_tts = None
        self.model_denoise = None

        self.put_accent=put_accent
        self.put_yo=put_yo

        self.pathTmpDir = pathTmpDir
        if self.pathTmpDir is None:
            self.pathTmpDir =  os.path.join( os.path.dirname( os.path.abspath(__file__) ), 'tmp' )
        os.makedirs(self.pathTmpDir, exist_ok=True)

        # this values init in load denoise model
        self.read_audio = None
        self.save_audio = None
        self.denoise = None
        self.model_type_denoise = denoiseModelType
        self.model_dnc = None
        self.__load_models()

        print('~~~~~~~ Settings ~~~~~~~ ')
        print("{:<30}".format('Use device'), self.torch_device)
        print("{:<30}".format('Path torch cache dir '), self.path_cache_nn_dir)
        print("{:<30}".format('Path output dir '), self.path_output_dir)
        print("{:<30}".format('Use language'), self.lang.value)
        print("{:<30}".format('Use speaker'), self.speaker.value)
        print("{:<30}".format('Use speaker voice'), self.speakerVoice.value)
        print("{:<30}".format('Use denoise model'), self.model_type_denoise.value)
        print("{:<30}".format('Is debug:'), self.isDebug)
        print("{:<30}".format('Count text block per file:'), self.__calcCountSectionInChancks() )
        print("{:<30}".format('Time block in seconds:'), self.timeBlockSeconds )
        print('')

        warnings.filterwarnings("ignore")
        print('End Init')
        return

    def __setWorkTorhcHubWorkDir(self):
        if not self.path_cache_nn_dir: return
        if os.path.isdir(self.path_cache_nn_dir):
            torch.hub.set_dir(self.path_cache_nn_dir)
        return

    def __load_models(self):
        self.__load__nn()
        self.__load_Denoise_Model(self.model_type_denoise)
        return

    def __load__nn(self):
        print('Loading models start...')
        self.__setWorkTorhcHubWorkDir()

        nnmodel_name = 'silero_tts'
        repodir = 'snakers4/silero-models'
        print(f'Load model {nnmodel_name} from repo {repodir}')

        model, example_text = torch.hub.load(repo_or_dir=repodir,
                                             model=nnmodel_name,
                                             language=self.lang.value,
                                             speaker=self.speaker.value)
        self.model_tts = model

        url_yml = r'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml'
        pathFileLocal = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,'latest_silero_models.yml')
        print(f'Download yml conf from {url_yml} to local file {pathFileLocal}')
        torch.hub.download_url_to_file( url_yml,pathFileLocal                                      ,
                                       progress=False)
        self.model_denoise = OmegaConf.load('latest_silero_models.yml')
        self.__load_Denoise_Model(self.model_type_denoise)
        print('Loading models end.')
        return

    def __load_Denoise_Model(self,
                             name:EDenoiseModelType = EDenoiseModelType.LargeFast  # 'large_fast', 'small_fast'
                             ):

        os.environ['TORCH_HOME'] =  Config.Path_Torch_Hub_Cache_Direcotry
        path_cache = Config.Path_Torch_Hub_Cache_Direcotry
        torch.hub.set_dir(path_cache)

        model_dnc, samples, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_denoise',
            name=name.value,
            device=self.torch_device)

        (read_audio, save_audio, denoise) = utils

        self.read_audio = read_audio
        self.save_audio = save_audio
        self.denoise = denoise
        self.model_dnc = model_dnc

        model_dnc.to(self.torch_device)  # gpu

        return

    def Text2AdioConvert(self, text:str,
                         sample_rate:int=48000,

                         arg_put_accent:bool=None,
                         arg_put_yo:bool=None,
                         ):


        assert self.model_tts is not None, "Model model_tts must be loaded"

        if arg_put_accent is None:
            arg_put_accent = self.put_accent
        if arg_put_yo is None:
            arg_put_yo = self.put_yo

        audio = self.model_tts.apply_tts(ssml_text=text,
                                         speaker=self.speakerVoice.value,
                                         sample_rate=sample_rate,
                                         put_accent=arg_put_accent,
                                         put_yo=arg_put_yo)
        self.__save_audio_isDebugg(audio,sample_rate, 'Base_TTS_Step')

        audio_conv, new_rate = self.denoise_audio(audio, sample_rate)
        self.__save_audio_isDebugg(audio_conv.flatten().numpy(), new_rate, 'Denoised_TTS_Step')

        return audio_conv.flatten(), new_rate


    def ConvretingPipeline(self, pathFile,
                           dbgChunkNumber:int = None
                           ):

        import TextReader

        def chunks(iterable, size):
            it = iter(iterable)
            chunk = list(islice(it, size))
            while chunk:
                yield chunk
                chunk = list(islice(it, size))

        """

        :return:
        """

        txtReader = TextReader.TextReader( pathfile= pathFile, language=self.lang )
        channks_per_audio_file = [i for i in chunks( txtReader.getTextChankDataFromFile(), self.__calcCountSectionInChancks() )]

        if self.isDebug and (dbgChunkNumber is not None):
            channks_per_audio_file = [ [ list( txtReader.getTextChankDataFromFile() )[dbgChunkNumber] ] ]

        for ix_part,text_data in enumerate(channks_per_audio_file):
            print('Iteration {} of {}\n'.format(ix_part,len(channks_per_audio_file)))

            audios = []
            for textitem in tqdm(text_data):
                xml_item = self.makeSSml(textitem)
                audio_conv, new_rate = self.Text2AdioConvert(xml_item)
                audios.append( audio_conv.flatten() )

            audio_block_orig = torch.cat(audios)
            audio = audio_block_orig.numpy()

            if self.isDebug:
                fname_dbg = "temp4ogg"
                fpath_dbg = self.__getFilePath(fname_dbg,'wav')
                print('Saved audio to',fpath_dbg)
                self.__save_wav_file(audio_block_orig.numpy(), rate=new_rate,
                                     path4save= fpath_dbg )
                audio = AudioSegment.from_wav( fpath_dbg )
            else:
                with BytesIO() as bio:
                    sf.write(bio, audio_block_orig.numpy(), new_rate, format='WAV')
                    audio = AudioSegment.from_file(BytesIO(bio.getvalue()), format='wav')

            pathFile_out = self.__getFilePath(f'part_{ix_part}',self.fileFormatOut.value.lower()  )
            print('Save part audio to', pathFile_out)
            audio.export(pathFile_out, format=  self.fileFormatOut.value.lower())

            if self.timeTestSampleLong is not None: break;

        return


    def denoise_audio(self,audio, rate:int ):
        assert self.model_dnc is not None, "Denoise model not loaded"
        new_rate = 48000
        rate4denoise = 24000
        path_file_tmp = self.makeTmpPathWavFile()
        try:
            self.saveAsWave( path_file_tmp, audio, rate)
            audio_crt = self.read_audio(path_file_tmp, sampling_rate=rate4denoise).to(torch.device('cpu'))
            self.__save_audio_isDebugg(audio_crt.flatten().numpy(), rate4denoise, "Change_rate_Denoise_change_power")
            audio_crt = audio_crt * Text2SpechWrapper.POWER_KOEFF
            self.__save_audio_isDebugg(audio_crt.flatten().numpy(),rate4denoise,"Before_Denoise_change_power")
        finally:
            if os.path.isfile(path_file_tmp):
                os.remove(path_file_tmp)

        output = self.model_dnc(audio_crt.to( self.torch_device ))
        output = output * Text2SpechWrapper.POWER_KOEFF_AFTER_DENOISE
        return output.squeeze(1).cpu(), new_rate

    def saveAsWave(self,fname:str,audio_numpy:np.ndarray, samplerate:int):
        sf.write(fname, audio_numpy, samplerate)
        return

    def generate_random_string(self,length = 20):
        # Combine all the types of characters you want to use
        characters = string.ascii_letters + string.digits  # ascii_letters includes both lowercase and uppercase letters

        # Use random.choices to select 'length' number of characters from the set
        random_string = ''.join(random.choices(characters, k=length))

        return random_string
    def makeTmpPathWavFile(self):
        return  os.path.join(self.pathTmpDir,
                             self.generate_random_string()+'.wav')


    def makeSSml(self, text:str)->str:
        text = '\n'.join([i.strip() for i in text.split('\n')])

        items = text.split('\n\n')

        sbreak = f'<break time="{Text2SpechWrapper.TIMEOUT_BREAK_HEADER}ms"/>'
        res = '<speak>' + f'\n\n{sbreak}'.join(items) + '</speak>'
        return res

    def __save_audio_isDebugg(self, audio, rate:int,step_name:str):
        if not self.isDebug: return;
        path4save = self.__getFilePath(step_name,'wav')
        print('Save audio in file', path4save)
        sf.write( path4save, audio, rate)
        return

    def __save_wav_file(self, audio, rate:int, path4save:str):
        if self.isDebug:
            print('Save wav in file', path4save)
        sf.write( path4save, audio, rate)


    def __getFilePath(self,fname, ext):
        return os.path.join(self.path_output_dir,f'{fname}.{ext}')

    def __calcCountSectionInChancks(self):
        if self.timeTestSampleLong is not None:
            return max(1, self.timeTestSampleLong // 38)  # 38- approx sound per block
        return max(1, self.timeBlockSeconds // 38) # 38- approx sound per block

    @staticmethod
    def GetAvaliableSpeakers()->typing.List[str]:
        all_values = [e.value for e in ESpeakerId.__members__.values()]
        return all_values
    @staticmethod
    def getDefaultSpeakerName()->str:
        return Text2SpechWrapper.SPEACKER_DEFAULT.value

    @staticmethod
    def getDefaultTimePerFile()->int:
        return Text2SpechWrapper.TIME_PER_FILE_SEC
