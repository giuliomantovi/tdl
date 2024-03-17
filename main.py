from audio_classification import spectrogram_models, mfcc_models
from audio_processing import source_separation, transcription
from lyrics_classification.lda_models import pretrained_lda_model, scratch_lda_model
from Config.Constants import *
# import warnings

# warnings.filterwarnings('ignore')


if __name__ == '__main__':
    """mic = Recorder()
    mic.setMicrophone()
    mic.record()"""

    #PROSSIMA COSA DA FARE: Tradurre le RISPOSTE DEI MODELLI LDA IN LEGGIBILI/PARAGONABILI

    # source_separation.source_separation(INPUT_AUDIO)
    # transcription.fast_transcript(INPUT_AUDIO)
    # print_pickle("WASABI_DB/topics/song_id_to_topics.pickle")
    # print_pickle("")
    # pretrained_lda_model.evaluate_text("Config/input/user/audio/pop.00026/vocals.txt")
    # scratch_lda_model.predict_text("Config/input/user/audio/pop.00026/vocals.txt")
    # efficientnet_model.testefficientnetmodel("Config/input/images", Constants.EFFICIENTNET_PRETRAINED_PATH)
    # spectrogram_models.testimagemodel(INPUT_IMAGES_CNN, CNN_IMAGE_PATH)
    # lda_model.predict_text()
    # lda_model.create_model_chunks()
    # lda_model.predict_text()
    # compute_text_similarity("Config/input/text/rollingInTheDeepPop/vocals.txt", "Config/input/text/ringoffire/vocals.txt")
    # predict_genre_from_lyrics()
    # lda_model.create_model()
    # evaluate_text_classifier() 46%
    # data=mfcc_models.preprocess_dataset(INPUT_AUDIO)
    # mfcc_models.testaudiomodel(data,LSMT_PATH)
    # create_pretrained_efficientnet_model(Constants.GTZAN_IMAGE_PATH)
    # convert_to_wav(Constants.INPUT_AUDIO)
    # CREATING SPECTROGRAMS FROM .WAV DIR
    # spectrogram_models.audio_to_spectrograms(INPUT_AUDIO)
