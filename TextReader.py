from pprint import pprint

import re

import os
from ProjectEnums import *
from nltk.tokenize import sent_tokenize
from num2words import num2words
import soundfile as sf

class TextReader:

    CNT_CHAR_PER_AUDIO_SPLIT = 512

    def __init__(self,
                 pathfile:str,
                 language:ELanguage = ELanguage.RU
                 ):
        self.pathfile = pathfile
        assert os.path.exists(self.pathfile)
        assert os.path.isfile(self.pathfile)
        self.language:ELanguage = language

        return

    def splitTextOnParagraphs(self, text):
        paragraphs = text.split('\n\n')
        return paragraphs

    def getTextChankDataFromFile(self)->str:
        """
        this functoin read text file, split by sentences and yield
        buffer for maka aoudio, which has small size

        :return:
        """
        buff = []
        with open(self.pathfile, 'r', encoding='utf8') as f:
            for cline in f:
                buff.append(cline)
        alltext = ' '.join(buff)
        sentences = sent_tokenize(alltext)
        sentences = [ self.transformLine_ReplaceNumber2Words(sent) for sent in sentences ]


        sizebuff = 0
        buff = []
        for ix, sent in enumerate(sentences):
            sizebuff += len(sent)
            buff.append(sent)
            if sizebuff > TextReader.CNT_CHAR_PER_AUDIO_SPLIT:
                yield  ' '.join(buff)
                buff = []
                sizebuff = 0

        if len(buff)>0:
            yield ' '.join(buff)

        return

    def convertNum2Words(self, value):
        in_words = num2words(value, lang=self.language.value)
        return in_words

    def transformLine_ReplaceNumber2Words( self,
                                           sline:str
                                           , isDebug:bool = False)->str:
        matches = re.finditer(r'\d+', sline)
        store_2_replace = []
        for match in matches:
            start, end = match.span()
            store_2_replace.append([start, end,sline[start:end],
                                    self.convertNum2Words(sline[start:end])] )
        if len(store_2_replace) == 0:
            return sline

        res = []
        ixstart = 0
        for repitem in store_2_replace:
            start, end,substr_num, num_as_str = repitem
            res.append( sline[ixstart:start] )
            res.append( num_as_str )
            ixstart = end


        if isDebug:
            pprint( store_2_replace )

        return ' '.join(res).replace('  ',' ')