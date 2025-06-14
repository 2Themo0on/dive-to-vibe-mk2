import os
import warnings
import numpy as np
import pandas as pd
import torch
import librosa

# ──────────────────────────────────────────────────────────────────────────────
# 1) 경고 무시 (librosa 로딩 경고 등)
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 2) 경로 설정
#    실제 환경에 맞게 수정하세요.
# ──────────────────────────────────────────────────────────────────────────────
DEAM_DATA_ROOT = r'C:\Users\sc\Desktop\dive-to-vibe-mk2\DEAM Dataset'

# • 원본 오디오(mp3) 폴더
AUDIO_DIR = os.path.join(DEAM_DATA_ROOT, 'DEAM_audio', 'MEMD_audio')

# • 분리된 wide-format 주석 CSV 파일 경로
VALENCE_CSV_PATH = os.path.join(
    DEAM_DATA_ROOT,
    'DEAM_Annotations',
    'annotations',
    'annotations averaged per song',
    'dynamic (per second annotations)',
    'valence.csv'
)
AROUSAL_CSV_PATH = os.path.join(
    DEAM_DATA_ROOT,
    'DEAM_Annotations',
    'annotations',
    'annotations averaged per song',
    'dynamic (per second annotations)',
    'arousal.csv'
)

# • 최종 .pt 파일 저장 경로
OUTPUT_PT_PATH = './DEAM_Processed/DEAM_dataset.pt'

# ──────────────────────────────────────────────────────────────────────────────
# 3) 스펙트로그램 & 주석 동기화 파라미터
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 44100                    # Hz
N_FFT           = 2048                     # FFT 윈도우 크기
HOP_LENGTH      = int(0.5 * SAMPLE_RATE)   # 0.5초 간격 (2Hz annotation)
N_MELS          = 128                      # Mel 필터 뱅크 개수
EXCLUDE_SECONDS = 15                       # 앞 15초 오디오 컷

# ──────────────────────────────────────────────────────────────────────────────
# 4) 출력 디렉토리 준비
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PT_PATH), exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) 주석 CSV 로드 (wide-format)
#    각 행에 'song_id' + 'sample_XXXXms' 컬럼들이 있습니다.
# ──────────────────────────────────────────────────────────────────────────────
val_df = pd.read_csv(VALENCE_CSV_PATH)
aro_df = pd.read_csv(AROUSAL_CSV_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 6) wide-format 주석 행 → 1D numpy array 변환 함수
# ──────────────────────────────────────────────────────────────────────────────
def extract_annotations_wide(row: pd.Series) -> np.ndarray:
    # 'sample_XXXXms' 로 시작하는 컬럼 찾기
    sample_cols = [c for c in row.index if c.startswith('sample_')]
    # 각 컬럼명에서 밀리초(ms) 값 추출 후 정렬
    ms_pairs = [
        (int(c.split('_')[1].replace('ms', '')), c)
        for c in sample_cols
    ]
    ms_pairs.sort(key=lambda x: x[0])
    sorted_cols = [col for _, col in ms_pairs]
    # float array 반환
    return row[sorted_cols].values.astype(float)

# ──────────────────────────────────────────────────────────────────────────────
# 7) 단일 곡 처리 함수
# ──────────────────────────────────────────────────────────────────────────────
def process_single_song(audio_path: str, song_id: int):
    """
    - 오디오 로드 & 앞 15초 제거
    - wide-format 주석에서 해당 song_id 행 추출 → 1D array 변환
    - Log-Mel 스펙트로그램 생성
    - 스펙트로그램 프레임수와 주석 길이 동기화
    - (spectrogram, annotations) 튜플 반환 또는 None
    """
    # 1) 오디오 로드
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = y[int(EXCLUDE_SECONDS * sr):]  # 앞 15초 제거

    # 2) song_id 행 추출
    vrow = val_df[val_df['song_id'] == song_id]
    arow = aro_df[aro_df['song_id'] == song_id]
    if vrow.empty or arow.empty:
        print(f"⚠️ Skipped song_{song_id}: annotation not found")
        return None

    # 3) wide-format → 1D numpy array
    v_arr = extract_annotations_wide(vrow.iloc[0])
    a_arr = extract_annotations_wide(arow.iloc[0])

    # 4) Log-Mel 스펙트로그램
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 5) 프레임수 동기화
    num_spec = log_mel.shape[1]
    num_ann  = min(len(v_arr), len(a_arr))
    mf       = min(num_spec, num_ann)
    if mf == 0:
        print(f"⚠️ Skipped song_{song_id}: zero frames")
        return None

    spec_slice = log_mel[:, :mf]                     # shape = (n_mels, mf)
    ann_slice  = np.stack([v_arr[:mf], a_arr[:mf]], 1)  # shape = (mf, 2)

    return spec_slice, ann_slice

# ──────────────────────────────────────────────────────────────────────────────
# 8) 메인 루프: 모든 mp3 처리 → 리스트에 수집
# ──────────────────────────────────────────────────────────────────────────────
spectrograms = []
annotations  = []
song_ids     = []

audio_files = sorted(f for f in os.listdir(AUDIO_DIR)
                     if f.lower().endswith('.mp3'))
print(f"총 {len(audio_files)}개 오디오 파일 처리 시작...\n")

for fname in audio_files:
    sid  = int(os.path.splitext(fname)[0].split('_')[-1])
    path = os.path.join(AUDIO_DIR, fname)

    result = process_single_song(path, sid)
    if result is None:
        continue

    spec, ann = result
    spectrograms.append(spec)
    annotations.append(ann)
    song_ids.append(sid)
    print(f"✅ Processed song_{sid}")

# ──────────────────────────────────────────────────────────────────────────────
# 9) 하나의 .pt 파일로 저장
# ──────────────────────────────────────────────────────────────────────────────
# - spectrograms: list of (n_mels × frames) numpy arrays
# - annotations:  list of (frames × 2) numpy arrays
# - song_ids:     list of int
spec_tensors = [torch.from_numpy(s).float().unsqueeze(0)
                for s in spectrograms]  # shape = (1, n_mels, frames)
ann_tensors  = [torch.from_numpy(a).float()
                for a in annotations]   # shape = (frames, 2)

torch.save({
    'spectrograms': spec_tensors,
    'annotations' : ann_tensors,
    'song_ids'    : song_ids
}, OUTPUT_PT_PATH)

print(f"\n🎉 저장 완료: {OUTPUT_PT_PATH}  (총 {len(song_ids)}곡)")

