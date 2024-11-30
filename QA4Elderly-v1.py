import os
# 设置 Hugging Face 缓存目录为 D 盘的 huggingface_cache 文件夹
os.environ['HF_HOME'] = 'D:/huggingface_cache'
from accelerate import Accelerator
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import librosa
import torchaudio
#import outetts
import scipy
from ChatTTS import ChatTTS

# 创建加速器对象
accelerator = Accelerator()

def check_device(model, inputs=None):
    """
    检查模型和输入是否运行在 GPU 上
    """
    print("Model device check:")
    for name, param in model.named_parameters():
        print(f"Layer {name} is on {param.device}")
    if inputs is not None:
        print("Inputs are on:", inputs.device)

def speech_to_text(audio_file_or_data, model_id="openai/whisper-large-v3-turbo"):
    # 判断设备（GPU 或 CPU）
    device = accelerator.device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # 加载模型和处理器
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    
    # 检查模型设备
    check_device(model)

    # 创建 ASR 流水线
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # 加载音频数据并进行推理
    if isinstance(audio_file_or_data, str):
        # 如果输入是文件路径，加载音频文件
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        audio_data = dataset[0]["audio"]
    else:
        # 如果输入是音频数据，直接使用
        audio_data = audio_file_or_data
    
    # 进行语音识别
    result = pipe(audio_data)
    
    # 返回识别结果中的文本
    return result["text"]

def get_model_response(text: str) -> str:
    """
    使用Qwen/Qwen2.5-1.5B-Instruct模型根据输入的文本生成相应的回答。

    参数:
    text (str): 输入文本。

    返回:
    str: 模型生成的回答。
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"  # 加速器将自动处理设备映射
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置prompt
    prompt = text
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Your answer should be concise and to the potint. "},
        {"role": "user", "content": prompt}
    ]
    
    # 创建模型输入
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # 检查模型和输入的设备
    check_device(model, model_inputs.input_ids)

    # 使用accelerator进行加速
    model, model_inputs = accelerator.prepare(model, model_inputs)

    # 生成模型输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码输出并返回
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def text_to_chinese_speech_with_random_timbre(text, output_path="C:/Users/dAnte/Desktop/project/output.wav"):
    """
    使用 ChatTTS 将中文文本转换为中文语音，生成随机音色并保存音色嵌入。
    同时将生成的语音保存为 WAV 文件。

    参数:
    - text (str): 输入的中文文本。
    - output_path (str): 保存生成语音的文件路径，默认为 "C:/Users/dAnte/Desktop/project/output.wav"。

    返回:
    - rand_spk (list): 生成的随机音色嵌入。
    """
    chat = ChatTTS.Chat()

    # 指定自定义路径，防止权限问题
    custom_path = "C:/Users/dAnte/Desktop/project/ChatTTS_models"
    os.makedirs(custom_path, exist_ok=True)
    
    try:
        chat.load(compile=False, custom_path=custom_path)  # 加载模型
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 检查是否成功初始化
    if not chat.has_loaded():
        print("模型初始化失败，请检查组件是否正确加载。")
        return

    # 生成一个随机音色
    rand_spk = chat.sample_random_speaker()
    print(f"生成的随机音色嵌入: {rand_spk}")

    # 设置生成参数，使用随机音色
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,  # 使用生成的随机音色嵌入
        temperature=0.3,   # 较低温度，生成稳定语音
        top_P=0.7,         # 限制候选概率总和
        top_K=20,          # 限制候选数量
    )

    # 生成语音
    try:
        wavs = chat.infer([text], params_infer_code=params_infer_code)  # 输入文本列表
        torchaudio.save(output_path, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
        print(f"语音已保存到: {output_path}")
    except Exception as e:
        print(f"生成语音时出错: {e}")

    # 返回生成的随机音色嵌入
    return rand_spk

def main():
    # 示例使用
    audio, sr = librosa.load("C:/Users/dAnte/Desktop/project/sample1.flac", sr=16000)
    text = speech_to_text(audio)
    print(text)
    response = get_model_response(text)
    print(response)
    text = response
    output_path = "C:/Users/dAnte/Desktop/project/chinese_speech_random_timbre.wav"

    # 调用函数生成语音和随机音色
    random_timbre = text_to_chinese_speech_with_random_timbre(text, output_path)

    # 保存随机音色嵌入
    if random_timbre:
        timbre_path = "C:/Users/dAnte/Desktop/project/random_timbre.txt"
        with open(timbre_path, "w", encoding="utf-8") as f:  # 指定 utf-8 编码
            f.write(str(random_timbre))
        print(f"随机音色嵌入已保存到: {timbre_path}")

if __name__ == "__main__":
    main()
