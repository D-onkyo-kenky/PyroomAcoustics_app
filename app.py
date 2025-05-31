import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pyroomacoustics as pra
import io
from scipy.io import wavfile
from scipy.io.wavfile import write
import os
from acoustic_utils import octave_band_filter, reverb_time_T30, cut_signal_by_threshold

font_path = "fonts/static/NotoSansJP-Light.ttf"
font_prop = fm.FontProperties(fname=font_path)

st.set_page_config(page_title="音響シミュレーションアプリ", layout="wide")
st.title("🔊 音響シミュレーションアプリ")

# --- 材料データの読み込み ---
csv_path = "materials.csv"
if os.path.exists(csv_path):
    materials_df = pd.read_csv(csv_path)
else:
    st.error("materials.csv が見つかりません。編集ページで作成してください。")
    st.stop()

# --- 部屋の設定 ---
st.sidebar.header("部屋の設定")
room_width = st.sidebar.number_input("部屋の幅 X  [m]", 1.0, 100.0, 8.0)
room_depth = st.sidebar.number_input("部屋の奥行 Y  [m]", 1.0, 100.0, 10.0)
room_height = st.sidebar.number_input("部屋の高さ Z  [m]", 1.0, 20.0, 3.0)
room_dim = [room_width, room_depth, room_height]

# --- 音源とマイクの位置 ---
st.sidebar.header("音源の位置")
src_x = st.sidebar.slider("🔉音源 X [m]", 0.0, room_width, room_width / 2)
src_y = st.sidebar.slider("🔉音源 Y [m]", 0.0, room_depth, room_depth / 2)
src_z = st.sidebar.slider("🔉音源 Z [m]", 0.0, room_height, 1.2)
st.sidebar.header("マイクの位置")
mic_x = st.sidebar.slider("🎙マイク X [m]", 0.0, room_width, room_width * 0.85)
mic_y = st.sidebar.slider("🎙マイク Y [m]", 0.0, room_depth, room_depth * 0.85)
mic_z = st.sidebar.slider("🎙マイク Z [m]", 0.0, room_height, 1.2)

# --- 音源ファイルの読み込み ---
st.sidebar.header("音源ファイル")
audio_file = st.sidebar.file_uploader("WAVファイルをアップロード", type=['wav'])
if audio_file is not None:
    fs, audio = wavfile.read(audio_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
elif os.path.exists("source.wav"):
    fs, audio = wavfile.read("source.wav")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    st.sidebar.info("デフォルトの音源を使用中")
else:
    st.sidebar.error("音源が見つかりません")
    st.stop()

# --- 音線法・虚像法の設定 ---
st.sidebar.header("詳細設定")
array_numbers = st.sidebar.number_input("音線数", min_value=10000, max_value=1000000, value=20000, step=10000)
radius = st.sidebar.number_input("受音半径[m]", 0.1, 1.0, 0.5)
energy_threshold = st.sidebar.number_input("エネルギー閾値", min_value=1e-7, max_value=1e-1, value=1e-5, step=1e-7, format="%.2e")
image_numbers= st.sidebar.number_input("虚像法の反射回数", 0, 100, 3)


# --- 各面の材料選択 ---
st.subheader("材料指定（各面）")

# 材料名リスト
material_names = materials_df['material'].tolist()
wall_colors = {
    'west': 'red', 'east': 'blue', 'south': 'green',
    'north': 'orange', 'floor': 'gray', 'ceiling': 'purple'
}

# プリセット定義
presets = {
    "ブラインド吸音なし": {
        'west': 'low abs board',
        'east': 'glass',
        'south': 'glass',
        'north': 'low abs board',
        'floor': 'low abs board',
        'ceiling': 'low abs board'
    },
    "ブラインド吸音あり": {
        'west': 'low abs board',
        'east': '★abs_blind',
        'south': '★abs_blind',
        'north': 'low abs board',
        'floor': 'low abs board',
        'ceiling': 'low abs board'
    },
    "コンクリート": {
        'west': 'concrete',
        'east': 'concrete',
        'south': 'concrete',
        'north': 'concrete',
        'floor': 'concrete',
        'ceiling': 'concrete'
    }    
}

# UIで選ぶプリセット名
selected_preset_name = st.selectbox("🎛 プリセットを選択", list(presets.keys()))

# プリセット適用フラグ
if "apply_preset" not in st.session_state:
    st.session_state.apply_preset = False

# 適用ボタン
if st.button("📋 プリセットを適用"):
    st.session_state.apply_preset = True
    st.session_state.preset_values = presets[selected_preset_name]

# セレクタ生成
cols = st.columns(6)
walls = {}

for i, wall in enumerate(wall_colors):
    with cols[i]:
        # プリセット適用時
        if st.session_state.apply_preset:
            default_material = st.session_state.preset_values[wall]
        else:
            default_material = st.session_state.get(f"{wall}_material", material_names[0])

        selected = st.selectbox(
            f"{wall}（{wall_colors[wall]}）",
            material_names,
            index=material_names.index(default_material),
            key=f"{wall}_material"
        )
        walls[wall] = selected

# 一度適用したら次回以降は適用フラグをオフにする
st.session_state.apply_preset = False


# --- 平面図描画 ---
dpi = 100
fig = plt.figure(figsize=(300 / dpi, 300 / dpi), dpi=dpi)
ax = fig.add_subplot(111)
ax.plot([0, 0], [0, room_depth], color='red', linewidth=3, label='west')
ax.plot([room_width, room_width], [0, room_depth], color='blue', linewidth=3, label='east')
ax.plot([0, room_width], [0, 0], color='green', linewidth=3, label='south')
ax.plot([0, room_width], [room_depth, room_depth], color='orange', linewidth=3, label='north')
ax.plot(src_x, src_y, '^', label='音源')
ax.plot(mic_x, mic_y, '*', label='マイク')
ax.set_xlim(-1, room_width + 1)
ax.set_ylim(-1, room_depth + 1)
ax.set_aspect('equal')
ax.set_title("部屋の平面図（上から見た図）", fontproperties=font_prop, fontsize=12)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop=font_prop)
st.pyplot(fig,use_container_width=False)

# --- シミュレーション実行 ---
if st.button("▶ シミュレーション実行"):
    # 材料定義
    material_dict = {
        row['material']: pra.Material({
            'coeffs': [row[f'{f}Hz'] for f in [125, 250, 500, 1000, 2000, 4000]],
            'center_freqs': [125, 250, 500, 1000, 2000, 4000]
        }, scattering=row['scattering']) for _, row in materials_df.iterrows()
    }

    # 部屋の作成
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials={face: material_dict[walls[face]] for face in walls},
        max_order=image_numbers, # 虚像法の反射回数
        ray_tracing=True, # 音線法の設定
        air_absorption=True # 空気減衰あり
    )

    # 音線法の設定
    room.set_ray_tracing(
    receiver_radius=radius,     # 受音半径[m]（一般的には0.1〜0.5）
    n_rays=array_numbers,       # 発射する音線の数（精度と速度のトレードオフ）
    energy_thres=energy_threshold      # エネルギーの閾値（例：元のエネルギーの1/1e-6まで追跡する）
    )

    room.add_source([src_x, src_y, src_z], signal=audio)
    room.add_microphone_array(pra.MicrophoneArray([[mic_x], [mic_y], [mic_z]], fs))
    room.compute_rir()
    room.simulate()

    # 再生と可視化
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    audio_bytes = io.BytesIO()
    write(audio_bytes, fs, audio_int16)
    audio_bytes.seek(0)
    st.subheader("🔉 元の音源")
    st.audio(audio_bytes, format="audio/wav")

    # 畳み込み
    signal = room.mic_array.signals[0] # インパルス応答の畳み込み＋距離減衰を適用
    #signal = cut_signal_by_threshold(signal, -80) # 音源の長さを調整（最後らへんの残響をカット）

    # 音圧スケールのまま再生（自動正規化を避ける）
    signal_clipped = np.clip(signal, -1.0, 1.0)  # クリッピングで安全処理
    signal_int16 = (signal_clipped * 32767).astype(np.int16)
    wav_bytes = io.BytesIO()
    write(wav_bytes, fs, signal_int16)
    wav_bytes.seek(0)

    # インパルス応答の畳み込み
    st.subheader("🎧 シミュレーション音源")
    signal_norm = signal / (np.max(np.abs(signal)) + 1e-12)
    signal_int16_norm = (signal_norm * 32767).astype(np.int16)
    wav_bytes_norm = io.BytesIO()
    write(wav_bytes_norm, fs, signal_int16_norm)
    wav_bytes_norm.seek(0)
    st.audio(wav_bytes_norm, format="audio/wav")

    # こちらは折りたたみ表示にする「実際の距離・吸音の影響をそのまま反映した音」
    with st.expander("🎧 距離感を反映した音（クリックで展開）"):
        st.caption("距離感を反映した音")
        signal_clipped = np.clip(signal, -1.0, 1.0)
        signal_int16 = (signal_clipped * 32767).astype(np.int16)
        wav_bytes = io.BytesIO()
        write(wav_bytes, fs, signal_int16)
        wav_bytes.seek(0)
        st.audio(wav_bytes, format="audio/wav")


    # インパルス応答の取得と正規化
    rir = room.rir[0][0]
    rir_norm = rir / (np.max(np.abs(rir)) + 1e-12) # 正規化

    # 残響時間（T30）
    st.subheader("⏱ 残響時間")
    cfreqs = [63, 125, 250, 500, 1000, 2000, 4000]
    T30_values = []
    for fc in cfreqs:
        y = octave_band_filter(rir_norm, fs, fc)
        T30 = reverb_time_T30(y, fs)
        T30_values.append(round(T30, 1)) # 小数点第一
    df = pd.DataFrame({'中心周波数 (Hz)': cfreqs, 'T30 (s)': T30_values})
    st.dataframe(df, width=300, hide_index=True)

    # インパルス応答のグラフと再生（正規化しない）
    with st.expander("📈 インパルス応答（クリックで展開）"):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.arange(len(rir)) / fs, rir)
        ax.set_xlabel("Time [s]", fontproperties=font_prop)
        ax.set_ylabel("Amplitude", fontproperties=font_prop)
        ax.set_title("インパルス応答", fontproperties=font_prop)
        st.pyplot(fig, use_container_width=False)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("実際の距離・吸音の影響をそのまま反映した音")
            rir_clipped = np.clip(rir, -1.0, 1.0)
            rir_int16 = (rir_clipped * 32767).astype(np.int16)
            rir_bytes = io.BytesIO()
            write(rir_bytes, fs, rir_int16)
            rir_bytes.seek(0)
            st.audio(rir_bytes, format="audio/wav")

        with col2:
            st.caption("音量を聞きやすく自動調整した音")
            rir_norm = rir / (np.max(np.abs(rir)) + 1e-12)
            rir_int16_norm = (rir_norm * 32767).astype(np.int16)
            rir_bytes_norm = io.BytesIO()
            write(rir_bytes_norm, fs, rir_int16_norm)
            rir_bytes_norm.seek(0)
            st.audio(rir_bytes_norm, format="audio/wav")