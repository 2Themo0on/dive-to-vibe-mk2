import librosa
import numpy as np
import torch
import os
import glob

# --- 경로 및 파라미터 설정 ---
# 1. 학습 데이터로 계산했던 정규화 통계 파일 경로
STATS_PATH = r"C:\Users\sc\Downloads\norm_stats.pt"

# 2. 오디오 처리 파라미터 (0.5초 단위 프레임)
SR = 44100
HOP_LENGTH_SECONDS = 0.5
HOP_LENGTH_SAMPLES = int(SR * HOP_LENGTH_SECONDS) # 22050

# --- 정규화 통계 로드 ---
try:
    stats = torch.load(STATS_PATH)
    mean = stats['mean']
    std = stats['std']
    print(f"'{STATS_PATH}'에서 정규화 통계 로드 완료 (Mean: {mean:.4f}, Std: {std:.4f})")
except FileNotFoundError:
    print(f"오류: '{STATS_PATH}' 파일을 찾을 수 없습니다. 모델 학습 시 생성된 파일을 업로드해주세요.")
    raise

def preprocess_and_save_new_song_normalized(audio_path, save_path, n_mels=128, n_fft=2048):
    """
    새로운 오디오를 log-mel 스펙트로그램으로 변환하고, '학습 시 사용된 통계치'로
    정규화한 뒤 저장합니다.
    """
    try:
        print(f"'{os.path.basename(audio_path)}' 파일 처리 시작...")
        
        # 1. 오디오 로드 및 스펙트로그램 생성
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=HOP_LENGTH_SAMPLES, n_mels=n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        spec_tensor = torch.from_numpy(log_mel_spectrogram).unsqueeze(0).float()
        
        # ★★★ (중요) 정규화 단계 추가 ★★★
        # 로드한 mean, std 값을 사용해 정규화를 적용합니다.
        spec_tensor = (spec_tensor - mean) / (std + 1e-8) # 1e-8은 0으로 나누는 것을 방지
        
        # 3. song_id 설정 및 저장
        base_name = os.path.basename(audio_path)
        song_id, _ = os.path.splitext(base_name)
        
        data_to_save = {
            'spectrogram': spec_tensor,
            'song_id': song_id
        }
        torch.save(data_to_save, save_path)
        
        print(f" -> 성공! (정규화 완료) -> '{os.path.basename(save_path)}'에 저장. (Shape: {spec_tensor.shape})")

    except Exception as e:
        print(f" -> 오류: '{os.path.basename(audio_path)}' 파일 처리 중 문제가 발생했습니다. ({e})")


# --- 디렉토리 전체 파일 처리 (사용 예시) ---
input_directory = r"C:\Users\sc\Desktop\dive-to-vibe-mk2\songs"
output_directory = "songs_preprocessed_pt"
os.makedirs(output_directory, exist_ok=True)

print("-" * 50)
print(f"전처리된 파일들은 '{output_directory}' 폴더에 저장됩니다.")

audio_files = []
supported_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a"]
for ext in supported_extensions:
    audio_files.extend(glob.glob(os.path.join(input_directory, ext)))

if not audio_files:
    print(f"오디오 파일을 찾을 수 없습니다. '{input_directory}' 경로를 확인해주세요.")
else:
    print(f"총 {len(audio_files)}개의 오디오 파일을 처리합니다.")
    
    for audio_path in audio_files:
        base_name = os.path.basename(audio_path)
        file_name_without_ext, _ = os.path.splitext(base_name)
        save_path = os.path.join(output_directory, f"{file_name_without_ext}.pt")

        # 정규화 기능이 추가된 함수 호출
        preprocess_and_save_new_song_normalized(audio_path, save_path)

    print("-" * 50)
    print("\n모든 파일 전처리가 완료되었습니다.")