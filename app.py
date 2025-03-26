import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pyroomacoustics as pra
from scipy.io import wavfile
import os
from acoustic_utils import octave_band_filter, reverb_time_T30

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
room_width = st.sidebar.number_input("部屋の幅 [m]", 1.0, 50.0, 8.0)
room_depth = st.sidebar.number_input("部屋の奥行 [m]", 1.0, 50.0, 10.0)
room_height = st.sidebar.number_input("部屋の高さ [m]", 1.0, 10.0, 3.0)
room_dim = [room_width, room_depth, room_height]

# --- 音源とマイクの位置 ---
st.sidebar.header("音源とマイクの位置")
src_x = st.sidebar.slider("音源 X [m]", 0.0, room_width, room_width / 10)
src_y = st.sidebar.slider("音源 Y [m]", 0.0, room_depth, room_depth / 10)
src_z = st.sidebar.slider("音源 Z [m]", 0.0, room_height, 1.2)
mic_x = st.sidebar.slider("マイク X [m]", 0.0, room_width, room_width * 0.75)
mic_y = st.sidebar.slider("マイク Y [m]", 0.0, room_depth, room_depth * 0.6)
mic_z = st.sidebar.slider("マイク Z [m]", 0.0, room_height, 1.2)

# --- 各面の材料選択 ---
st.subheader("部屋の平面図と壁材指定")
material_names = materials_df['material'].tolist()
wall_colors = {
    'west': 'red', 'east': 'blue', 'south': 'green',
    'north': 'orange', 'floor': 'gray', 'ceiling': 'purple'
}
walls = {}
cols = st.columns(6)
for i, wall in enumerate(wall_colors):
    with cols[i]:
        walls[wall] = st.selectbox(f"{wall} ({wall_colors[wall]})", material_names, key=wall)

# --- 平面図描画 ---
dpi = 100
fig = plt.figure(figsize=(600 / dpi, 300 / dpi), dpi=dpi)
ax = fig.add_subplot(111)
ax.plot([0, room_width], [0, 0], color='green', linewidth=3, label='south')
ax.plot([0, room_width], [room_depth, room_depth], color='orange', linewidth=3, label='north')
ax.plot([0, 0], [0, room_depth], color='red', linewidth=3, label='west')
ax.plot([room_width, room_width], [0, room_depth], color='blue', linewidth=3, label='east')
ax.plot(src_x, src_y, 'ko', label='音源')
ax.plot(mic_x, mic_y, 'ks', label='マイク')
ax.set_xlim(-1, room_width + 1)
ax.set_ylim(-1, room_depth + 1)
ax.set_aspect('equal')
ax.set_title("部屋の平面図（上から見た図）", fontproperties=font_prop, fontsize=12)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop=font_prop)
st.pyplot(fig)

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
    st.sidebar.info("source.wav を使用中")
else:
    st.sidebar.error("音源が見つかりません")
    st.stop()

# --- シミュレーション実行 ---
if st.button("▶ シミュレーション実行"):
    # 材料定義
    material_dict = {
        row['material']: pra.Material({
            'coeffs': [row[f'{f}Hz'] for f in [125, 250, 500, 1000, 2000, 4000]],
            'center_freqs': [125, 250, 500, 1000, 2000, 4000]
        }) for _, row in materials_df.iterrows()
    }

    # 部屋の作成
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials={face: material_dict[walls[face]] for face in walls},
        max_order=17
    )

    room.add_source([src_x, src_y, src_z], signal=audio)
    room.add_microphone_array(pra.MicrophoneArray([[mic_x], [mic_y], [mic_z]], fs))
    room.compute_rir()
    room.simulate()

    # 再生と可視化
    st.subheader("🔉 元音源")
    st.audio(audio, sample_rate=fs)

    st.subheader("🎧 シミュレーション音源")
    signal = room.mic_array.signals[0]
    signal = signal / np.max(np.abs(signal))
    st.audio(signal, sample_rate=fs)
    # wavfile.write("mic0.wav", fs, (signal * 32767).astype(np.int16))
    # st.success("mic0.wav を保存しました")

    st.subheader("📈 インパルス応答波形")
    rir = room.rir[0][0]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rir)) / fs, rir)
    ax.set_xlabel("Time [s]",fontproperties=font_prop)
    ax.set_ylabel("Amplitude",fontproperties=font_prop)
    ax.set_title("インパルス応答",fontproperties=font_prop)
    st.pyplot(fig)
    st.audio(rir / np.max(np.abs(rir)), sample_rate=fs)

    # 残響時間（T30）
    st.subheader("⏱ 残響時間解析 (1/1 Oct)")
    cfreqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    T30_values = []
    for fc in cfreqs:
        y = octave_band_filter(rir, fs, fc)
        T30 = reverb_time_T30(y, fs)
        T30_values.append(T30)

    df = pd.DataFrame({'中心周波数 (Hz)': cfreqs, 'T30 (s)': T30_values})
    st.dataframe(df, hide_index=True)
