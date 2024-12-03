
from .components import BERTForVADMapping, StringLabelClassifier
from .unit import ArcturusConfig, AudioInput, BERTForVADMappingOutput
from .util import (
    load_from_lora_checkpoint, 
    load_chatmusician_from_pretrained,
    load_chatmusician_tokenizer,
    PROMPT_GEN,
    PROMPT_STR,
    PROMPT_VAD
)
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TextStreamer, BatchEncoding, GenerationConfig
from typing import TypeVar
import torch
import warnings

Input = TypeVar("Input", str, AudioInput) # can either be a string or an audio file
Output = TypeVar("Output", str, torch.Tensor)

class ArcturusModel():

    def __init__(self, config : ArcturusConfig):

        self.config = config

        self.generation_config = GenerationConfig(
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            min_new_tokens=10,
            max_new_tokens=2048
        )

        if not config['from_text']:
            self.processor = WhisperProcessor.from_pretrained(config['whisper_variant'])
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(config['whisper_variant'])
            self.whisper_model.config.forced_decoder_ids = None
        
        if 'vad_checkpoint' in config:
            self.vad_classifier = BERTForVADMapping.from_pretrained(config['vad_checkpoint'])
        else:
            self.vad_classifier = BERTForVADMapping()
            warnings.warn("WARNING: You are initializing an untrained BERTForVADMapping model. ARCturus will suck at emotion identification.")


        if not config['from_vad']:
            assert 'label_classifier_config' in config
            if 'label_checkpoint' in config:
                self.string_classifier = StringLabelClassifier.from_pretrained(config['label_classifier_config'], config['label_checkpoint'])
            else:
                self.string_classifier = StringLabelClassifier(config['label_classifier_config'])
                warnings.warn(
                    "WARNING: You are initializing an untrained StringLabelClassifier model. The model will suck at emotion inference."
                )
        else:
            self.string_classifier = None

        self.chatmusician = load_chatmusician_from_pretrained()
        self.chatmusician_tokenizer = load_chatmusician_tokenizer()
        if config['use_finetuned_chatmusician']:
            assert 'chatmusician_checkpoint' in config
            self.chatmusician = load_from_lora_checkpoint(
                model=self.chatmusician, 
                path=config['chatmusician_checkpoint'], 
                device=self.chatmusician.device
            )

        # Set the models to eval mode
        self.vad_classifier.eval()
        if self.string_classifier is not None: self.string_classifier.eval()
        self.chatmusician.eval()

        # Set ChatMusician's eval strategy to use its KV Cache for faster inference
        self.chatmusician.config.use_cache = True


    def _process_audio(self, audio_input : AudioInput) -> str:
        whisper_input : torch.Tensor = self.processor(
            audio_input['array'], 
            sampling_rate=audio_input['sampling_rate'],
            return_tensors="pt"
        )
        ids : BatchEncoding = self.whisper_model(**whisper_input)
        text : str = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text


    def _construct_chatmusician_prompt(self, instruction : str) -> BatchEncoding:
        prompt = f"Human: ${instruction} </s> Assistant: "
        model_input = self.chatmusician_tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        return model_input


    def __call__(self, system_input : Input, debug : bool = False) -> Output:
        
        # Get the text input
        text : str = ""
        if self.config['from_text']:
            assert isinstance(system_input, str)
            text = system_input
        else:
            assert isinstance(system_input, AudioInput)
            text = self._process_audio(system_input)

        # Use the BERTForVADMapping model to get the valence-arousal-dominance values
        emo_recog_input = self.vad_classifier.tokenize(text)
        with torch.no_grad():
            vad_output : BERTForVADMappingOutput = self.vad_classifier(emo_recog_input)
            vad_values : torch.Tensor = vad_output['vad_values']

        # Branch here: if we are using string-based emotions, we get these values from the classifier;
        #              else, we pass in the vad values to the prompt as is.
        if not self.config['from_vad']:
            assert self.string_classifier is not None
            # TODO: Map string classifier predictions to their corresponding labels
            compose_prompt : str = PROMPT_STR.safe_substitute({"emotions" : "happiness, surprise"}) 
        else:
            assert len(vad_values.shape) == 1
            assert vad_values.shape[0] == 3
            valence, arousal, dominance = tuple(vad_values.tolist())
            compose_prompt : str = PROMPT_VAD.safe_substitute({"valence" : valence, "arousal" : arousal, "dominance" : dominance})

        # Set a text streamer for verbose output
        if self.config['stream_output']:
            streamer = TextStreamer(self.chatmusician_tokenizer, skip_prompt=False, skip_special_tokens=True)
        else:
            streamer = None

        # Construct and tokenize the initial prompt
        model_input = self._construct_chatmusician_prompt(compose_prompt)

        # Call the model on this input
        ident_response = self.chatmusician.generate(
            input_ids=model_input["input_ids"].to(self.chatmusician.device),
            streamer=streamer,
            attention_mask=model_input['attention_mask'].to(self.chatmusician.device),
            eos_token_id=self.tokenizer.eos_token_id,
            generation_config=self.generation_config,
        )
        ident_response = self.tokenizer.decode(ident_response[0][model_input["input_ids"].shape[1]:], skip_special_tokens=True)