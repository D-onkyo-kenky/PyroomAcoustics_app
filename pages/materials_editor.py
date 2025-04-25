import streamlit as st
import pandas as pd
import os
import io

st.set_page_config(page_title="吸音材料編集", layout="wide")
st.title("🧱 吸音材料データの編集")

csv_path = "materials.csv"

# 初期データ
def default_materials():
    return pd.DataFrame({
        'material': ['concrete', 'carpet', 'curtain'],
        '125Hz': [0.02, 0.08, 0.10],
        '250Hz': [0.03, 0.15, 0.25],
        '500Hz': [0.04, 0.30, 0.45],
        '1000Hz': [0.05, 0.50, 0.60],
        '2000Hz': [0.05, 0.60, 0.70],
        '4000Hz': [0.05, 0.65, 0.75],
        'scattering':[0.05, 0.05, 0.05]
    })

# CSV読込 or 初期化
if os.path.exists(csv_path):
    materials_df = pd.read_csv(csv_path)
else:
    materials_df = default_materials()


# --- CSVインポート処理 ---
uploaded_file = st.file_uploader("CSVファイルから材料を追加", type=["csv"])
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        required_columns = ['material', '125Hz', '250Hz', '500Hz', '1000Hz', '2000Hz', '4000Hz', 'scattering']
        if all(col in new_df.columns for col in required_columns):
            materials_df = pd.concat([materials_df, new_df], ignore_index=True).drop_duplicates(subset='material')
            st.success("CSVから吸音材料を追加しました。")
        else:
            st.error("必要なカラムがありません。'material' と各周波数帯の列が必要です。")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")


# 表形式で編集
st.subheader("📋 吸音材料の編集")
edited_df = st.data_editor(materials_df, num_rows="dynamic", use_container_width=True)


# 保存ボタン
if st.button("💾 保存してデータベースに反映"):
    edited_df.to_csv(csv_path, index=False)
    st.success("データベースに保存しました！")

# CSVとしてダウンロード
csv_buffer = io.StringIO()
edited_df.to_csv(csv_buffer, index=False)
st.download_button("CSVファイルでダウンロード", data=csv_buffer.getvalue(), file_name="materials.csv", mime="text/csv")
