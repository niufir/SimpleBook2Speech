from enum import Enum
class ELanguage(Enum):
    RU = 'ru'

class LanguageType(Enum):
    RU = "ru"

class ENNModelType(Enum):
    Speaker4Ru = 'v4_ru'

class ESpeakerId(Enum):
    SpeakerAidar ='aidar'
    SpeakerBaya  ='baya'
    SpeakerKseniay = 'kseniya'
    SpeakerXenia = 'xenia'
    SpeakerEugene = 'eugene'
    SpeakerRandom = 'random'

class EDenoiseModelType(Enum):
    LargeFast = 'large_fast'
    SmallFast = 'small_fast'

class EFileOutputFormatType(Enum):
    MP3 = 'mp3'
    OGG = 'ogg'

