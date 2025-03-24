def __init__(
    self,
    model_path: str,
    model_type: str = "llama",
    stage: str = "lm_pretrain",
    new_token_type: str = "insert",
    motion_codebook_size: int = 512,
    audio_codebook_size: int = 500,
    motion_framerate: float = 30.0,
    audio_samplerate: float = 16000.0,
    motion_down_sampling: int = 1,
    audio_down_sampling: int = 320,   ### audio down sample rate
    predict_ratio: float = 0.2,
    inbetween_ratio: float = 0.25,
    max_length: int = 512,
    lora: bool = False,
    quota_ratio: float = 0.5,
    noise_density: float = 0.15,
    mean_noise_span_length: int = 3,
    flash_attention: bool = False,
    modalities: dict = None,
    **kwargs,
) -> None:
    super().__init__()

    # Parameters
    self.m_codebook_size = motion_codebook_size
    self.face_codebook_size = motion_codebook_size
    self.hand_codebook_size = motion_codebook_size
    self.upper_codebook_size = motion_codebook_size
    self.lower_codebook_size = motion_codebook_size

    self.a_codebook_size = audio_codebook_size
    self.max_length = max_length
    self.motion_framerate = motion_framerate
    self.audio_samplerate = audio_samplerate
    self.motion_down_sampling = motion_down_sampling
    self.audio_down_sampling = audio_down_sampling
    self.predict_ratio = predict_ratio
    self.inbetween_ratio = inbetween_ratio
    self.mask_ratio_audio = 0.08

    self.noise_density = noise_density
    self.mean_noise_span_length = mean_noise_span_length
    self.quota_ratio = quota_ratio
    self.stage = stage
    self.model_type = model_type  # Store the model type

    # Instantiate language model
    if model_type == "llama3":
        # Special handling for Llama 3
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=True,
            trust_remote_code=True
        )
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.lm_type = 'dec'
    else:
        # Original initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        self.language_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.lm_type = 'dec'
    
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    for modality, settings in modalities.items():
        prefix = settings["prefix"]
        codebook_size = settings["codebook_size"] + 3
        # Generate tokens for the current modality
        tokens = [f"<{prefix}_{i}>" for i in range(codebook_size)]
        self.tokenizer.add_tokens(tokens)

    if new_token_type == "insert":
        self.language_model.resize_token_embeddings(len(self.tokenizer))
    elif new_token_type == "mlp":
        shared = NewTokenEmb(self.language_model.shared,
                             self.m_codebook_size + 3)
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.language_model.shared = shared

    # Lora
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
        from peft.utils.other import fsdp_auto_wrap_policy
        peft_config = LoraConfig(
            bias="none",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05)
        self.language_model = get_peft_model(self.language_model, peft_config)

def forward_dec(
    self,
    texts: List[str],
    face_token: Tensor,
    hand_token: Tensor,
    lower_token: Tensor,
    upper_token: Tensor,
    audio_tokens: Tensor,
    lengths: List[int],
    audio_length: List[int],
    tasks: dict,
    emotion_label: List[str]
):
    
    self.tokenizer.padding_side = "right"
    # Convert tokens to strings
    face_strings, hand_strings, upper_strings, lower_strings, motion_string = self.compositional_motion_token_to_string(
        face_token, hand_token, lower_token, upper_token, lengths)
    audio_strings = self.audio_token_to_string(audio_tokens, audio_length)
    
    # Create empty combine_strings if needed
    if 'text_timestamp' in dir(self) and self.text_timestamp is not None:
        combine_strings = self.audio_transcript_token_to_string(audio_tokens, self.text_timestamp, audio_length)
    else:
        combine_strings = [''] * len(lengths)

    # Get inputs and outputs using template
    inputs, outputs = self.template_fulfill(tasks, lengths, audio_length, 
                                           face_strings, hand_strings, upper_strings, lower_strings,
                                           motion_string, audio_strings, texts, combine_strings, emotion_label)
    
    # Format sequences based on model type
    full_sequences = []
    if self.model_type == "llama3":
        # Format using Llama 3 chat template
        for i in range(len(inputs)):
            # Create a structured chat conversation
            chat_sequence = [
                {"role": "system", "content": "You are a helpful AI assistant that translates between human language and motion tokens."},
                {"role": "user", "content": inputs[i]},
                {"role": "assistant", "content": outputs[i]}
            ]
            # Apply the chat template from the tokenizer
            formatted_chat = self.tokenizer.apply_chat_template(
                chat_sequence, 
                tokenize=False, 
                add_generation_prompt=False
            )
            full_sequences.append(formatted_chat)
    else:
        # Standard format with separator for other models
        for i in range(len(inputs)):
            full_sequences.append(inputs[i] + " ### " + outputs[i] + self.tokenizer.eos_token)

    # Tokenize
    inputs = self.tokenizer(full_sequences,
                          padding='max_length',
                          max_length=self.max_length,
                          truncation=True,
                          return_attention_mask=True,
                          return_tensors="pt")

    labels = inputs.input_ids.clone()
    
    # Mask out the input portion in labels (we only want to compute loss on the output portion)
    if self.model_type == "llama3":
        # For Llama 3, find where the assistant's response starts (after [/INST])
        for i, sequence in enumerate(full_sequences):
            # Look for the assistant's response markers in Llama 3's format
            assistant_marker = sequence.find("[/INST]")
            if assistant_marker == -1:
                # Fallback if format is different
                assistant_marker = sequence.find("assistant:")
            
            if assistant_marker != -1:
                # Get tokens up to the assistant's response
                input_part = self.tokenizer(sequence[:assistant_marker+7], return_tensors="pt").input_ids[0]
                labels[i, :len(input_part)] = -100
            else:
                # If format can't be determined, use a heuristic
                approx_length = len(self.tokenizer(inputs[i], return_tensors="pt").input_ids[0])
                labels[i, :approx_length] = -100
    else:
        # Original approach for other models
        for i, sequence in enumerate(full_sequences):
            separator_idx = sequence.find(" ### ")
            if separator_idx != -1:
                input_part = self.tokenizer(sequence[:separator_idx+5], return_tensors="pt").input_ids[0]
                labels[i, :len(input_part)] = -100
            else:
                # Fallback if separator not found
                approx_length = len(self.tokenizer(inputs[i], return_tensors="pt").input_ids[0])
                labels[i, :approx_length] = -100

    # Move to the appropriate device
    labels = labels.to(face_token.device)
    input_ids = inputs.input_ids.to(face_token.device)
    attention_mask = inputs.attention_mask.to(face_token.device)

    # Forward pass through the model
    outputs = self.language_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

    return outputs 