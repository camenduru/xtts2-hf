import sys
import os
# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

import langid 

import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")

def predict(prompt, language, audio_file_pth, mic_file_path, use_mic, agree):
    if agree == True:
        supported_languages=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn"]

        if language not in supported_languages:
            gr.Warning("Language you put in is not in is not in our Supported Languages, please choose from dropdown")
                
            return (
                    None,
                    None,
                ) 

        language_predicted=langid.classify(prompt)[0].strip() # strip need as there is space at end!

        if language_predicted == "zh": 
            #we use zh-cn 
            language_predicted = "zh-cn"
        #This is for identifying problems only. 
        print(f"Detected language:{language_predicted}, Chosen language:{language}, text:{prompt}")

        if language_predicted != language:
            #Please duplicate and remove this check if you really want this
            #Or auto-detector fails to identify language (which it can on pretty short text or mixed text)
            gr.Warning(f"Auto-Predicted Language in prompt (detected: {language_predicted}) does not match language you chose (chosen: {language}) , please choose correct language id. If you think this is incorrect please duplicate this space and modify code.")
            
            return (
                    None,
                    None,
                ) 


        
        if use_mic == True:
            if mic_file_path is not None:
                speaker_wav=mic_file_path
            else:
                gr.Warning("Please record your voice with Microphone, or uncheck Use Microphone to use reference audios")
                return (
                    None,
                    None,
                ) 
                
        else:
            speaker_wav=audio_file_pth

        if len(prompt)<2:
            gr.Warning("Please give a longer prompt text")
            return (
                    None,
                    None,
                )
        if len(prompt)>200:
            gr.Warning("Text length limited to 200 characters for this demo, please try shorter text")
            return (
                    None,
                    None,
                )  
        try:   
            tts.tts_to_file(
                text=prompt,
                file_path="output.wav",
                speaker_wav=speaker_wav,
                language=language,
            )
        except RuntimeError as e :
            if "device-assert" in str(e):
                # cannot do anything on cuda device side error, need tor estart
                gr.Warning("Unhandled Exception encounter, please retry in a minute")
                print("Cuda device-assert Runtime encountered need restart")
                sys.exit("Exit due to cuda device-assert")
            else:
                raise e
            
        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")
        return (
                None,
                None,
            ) 


title = "Coqui🐸 XTTS"

description = """
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
<br/>
XTTS is built on previous research, like Tortoise, with additional architectural innovations and training to make cross-language voice cloning and multilingual speech generation possible. 
<br/>
This is the same model that powers our creator application <a href="https://coqui.ai">Coqui Studio</a> as well as the <a href="https://docs.coqui.ai">Coqui API</a>. In production we apply modifications to make low-latency streaming possible.
<br/>
Leave a star on the Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, where our open-source inference and training code lives.
<br/>
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
<p>Language Selectors: 
Arabic: ar, Brazilian Portuguese: pt , Chinese: zh-cn, Czech: cs,<br/> 
Dutch: nl, English: en, French: fr, Italian: it, Polish: pl,<br/> 
Russian: ru, Spanish: es, Turkish: tr <br/> 
</p>
"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
</div>
"""
examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        None,
        False,
        True,
    ],
    [
        "Als ich sechs war, sah ich einmal ein wunderbares Bild",
        "de",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Cuando tenía seis años, vi una vez una imagen magnífica",
        "es",
        "examples/male.wav",
        None,
        False,
        True,
    ],
    [
        "Quando eu tinha seis anos eu vi, uma vez, uma imagem magnífica",
        "pt",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Kiedy miałem sześć lat, zobaczyłem pewnego razu wspaniały obrazek",
        "pl",
        "examples/male.wav",
        None,
        False,
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Когда мне было шесть лет, я увидел однажды удивительную картинку",
        "ru",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Toen ik een jaar of zes was, zag ik op een keer een prachtige plaat",
        "nl",
        "examples/male.wav",
        None,
        False,
        True,
    ],
    [
        "Když mi bylo šest let, viděl jsem jednou nádherný obrázek",
        "cs",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "examples/female.wav",
        None,
        False,
        True,
    ],
]



gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="Hi there, I'm your new voice clone. Try your best to upload quality audio",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female.wav",
        ),
        gr.Audio(source="microphone",
                 type="filepath",
                 info="Use your microphone to record audio",
                 label="Use Microphone for Reference"),
        gr.Checkbox(label="Check to use Microphone as Reference",
                    value=False,
                    info="Notice: Microphone input may not work properly under traffic",),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).queue().launch(debug=True)