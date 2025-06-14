import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings

# 경고 메시지 무시 (오디오 파일 로딩 시 발생할 수 있는 경고 등)
warnings.filterwarnings('ignore')

# --- 설정 매개변수 ---
# 원본 DEAM 데이터셋의 경로를 설정합니다.
# 이 경로 아래에 'audio' 폴더와 'annotations' 폴더가 있다고 가정합니다.
# 예: DEAM_DATA_ROOT/audio/*.mp3, DEAM_DATA_ROOT/annotations/averaged_dynamic_annotations_2Hz.csv
DEAM_DATA_ROOT = 'C:\\Users\\sc\\Desktop\\dive-to-vibe-mk2\\DEAM Dataset'
AUDIO_DIR = 'C:\\Users\\sc\\Desktop\\dive-to-vibe-mk2\\DEAM Dataset\\DEAM_audio\\MEMD_audio'
ANNOTATIONS_DIR = 'C:\\Users\\sc\\Desktop\\dive-to-vibe-mk2\\DEAM Dataset\\DEAM_Annotations\\annotations\\annotations averaged per song\\dynamic (per second annotations)'

# 처리된 데이터를 저장할 출력 디렉토리
OUTPUT_DIR = './DEAM_Processed'
SPECTROGRAM_NPY_DIR = os.path.join(OUTPUT_DIR, 'spectrograms_npy')
SPECTROGRAM_PNG_DIR = os.path.join(OUTPUT_DIR, 'spectrograms_png')
PYTORCH_TENSOR_DIR = os.path.join(OUTPUT_DIR, 'pytorch_tensors')

# 새로 추가: 분리된 주석 CSV 경로
VALENCE_CSV_PATH     = 'C:\\Users\\sc\\Desktop\\dive-to-vibe-mk2\\DEAM Dataset\\DEAM_Annotations\\annotations\\annotations averaged per song\\dynamic (per second annotations)\\valence.csv'   # 실제 경로로 수정
AROUSAL_CSV_PATH     = 'C:\\Users\\sc\\Desktop\\dive-to-vibe-mk2\\DEAM Dataset\\DEAM_Annotations\\annotations\\annotations averaged per song\\dynamic (per second annotations)\\arousal.csv'   # 실제 경로로 수정


# STFT 및 Mel 스펙트로그램 매개변수
SAMPLE_RATE = 44100  # DEAM 데이터셋의 샘플링 주파수
N_FFT = 2048       # FFT 윈도우 크기 (약 46ms @ 44100Hz)
HOP_LENGTH = int(0.5 * SAMPLE_RATE) # 0.5초 간격으로 스펙트로그램 프레임을 생성하여 주석과 동기화
N_MELS = 128       # Mel 필터 뱅크의 개수
EXCLUDE_SECONDS = 15 # 주석 불안정성으로 인해 각 클립의 처음 15초 제외

# 필요한 디렉토리 생성
os.makedirs(SPECTROGRAM_NPY_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_PNG_DIR, exist_ok=True)
os.makedirs(PYTORCH_TENSOR_DIR, exist_ok=True)

# --- 유틸리티 함수 ---for d in (SPECTROGRAM_NPY_DIR, SPECTROGRAM_PNG_DIR, PYTORCH_TENSOR_DIR):


# --- 주석 파일 로드 ---
valence_df = pd.read_csv(VALENCE_CSV_PATH)
arousal_df = pd.read_csv(AROUSAL_CSV_PATH)


def process_single_song(audio_path, song_id, valence_df, arousal_df):
    print(f"곡 처리 중: {song_id}")
    # 1. 오디오 로드 및 앞 15초 제거
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    start_sample = int(EXCLUDE_SECONDS * sr)
    y_trimmed = y[start_sample:]

    # 2. 해당 곡의 valence/arousal 주석 추출
    vidx = valence_df['song_id'] == int(song_id)
    aidx = arousal_df['song_id'] == int(song_id)
    if not vidx.any() or not aidx.any():
        print(f"경고: {song_id}에 대한 주석이 없습니다. 건너뜁니다.")
        return

    row_val = valence_df.loc[vidx].iloc[0]
    row_aro = arousal_df.loc[aidx].iloc[0]

    # 3. 'sample_XXXXms' 컬럼을 시간 순으로 정렬
    def extract_and_sort(row):
        cols = [c for c in row.index if c.startswith('sample_')]
        # 키: 숫자(ms)로 정렬
        cols_sorted = sorted(cols, key=lambda x: int(x.split('_')[1].replace('ms','')))
        return row[cols_sorted].values.astype(float)

    valence_arr = extract_and_sort(row_val)
    arousal_arr = extract_and_sort(row_aro)

    # 4. Log-Mel 스펙트로그램 생성
    mel_spec = librosa.feature.melspectrogram(
        y=y_trimmed, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 5. 프레임 수 맞추기
    num_spec_frames = log_mel.shape[1]
    num_ann_frames  = min(len(valence_arr), len(arousal_arr))
    min_frames = min(num_spec_frames, num_ann_frames)

    log_mel_aligned = log_mel[:, :min_frames]
    annotations = np.stack([
        valence_arr[:min_frames],
        arousal_arr[:min_frames]
    ], axis=1)  # shape = (min_frames, 2)

    # --- 저장 ---
    base = f"song_{song_id}"
    # (1) .npy
    np.save(os.path.join(SPECTROGRAM_NPY_DIR, f"{base}.npy"),
            log_mel_aligned)
    # (2) .png
    plt.figure(figsize=(10,4))
    librosa.display.specshow(
        log_mel_aligned, sr=sr, x_axis='time', y_axis='mel',
        hop_length=HOP_LENGTH, cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-Mel Spectrogram of Song {song_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(SPECTROGRAM_PNG_DIR, f"{base}.png"))
    plt.close()

    # (3) .pt
    spec_tensor   = torch.from_numpy(log_mel_aligned).float().unsqueeze(0)
    ann_tensor    = torch.from_numpy(annotations).float()
    torch.save({'spectrogram': spec_tensor, 'annotations': ann_tensor},
               os.path.join(PYTORCH_TENSOR_DIR, f"{base}.pt"))

    print(f"완료: {base}")

if __name__ == '__main__':
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')])
    for cnt, fname in enumerate(audio_files, 1):
        sid = os.path.splitext(fname)[0].replace('song_', '')
        process_single_song(os.path.join(AUDIO_DIR, fname), sid, valence_df, arousal_df)
        # if cnt >= 5:  # 테스트용
        #     print("5곡 처리 후 종료")
        #     break
    print("모든 처리 완료")