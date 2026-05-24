import csv
import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# PROMPT & AUTO-INSTALL ORANGE3 SEBELUM MENGEKSEKUSI KODE UTAMA
# ==============================================================================
try:
    import Orange
except ImportError:
    # Menampilkan animasi loading/spinner di web Streamlit saat instalasi berjalan
    with st.spinner("Menyiapkan lingkungan aplikasi... Menginstal 'Orange3' di latar belakang."):
        try:
            # Menggunakan "--only-binary" agar PIP mencari Wheel pra-kompilasi
            # Ini mencegah server cloud mencoba merakit modul C++ dari nol yang memicu error
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "Orange3", "--only-binary=:all:"
            ])
            st.success("Pustaka 'Orange3' berhasil dikonfigurasi! Memuat ulang halaman...")
            st.rerun()
        except Exception as e:
            # Jika opsi biner gagal, coba instalasi standar sebagai fallback
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "Orange3"])
                st.rerun()
            except Exception as fallback_error:
                st.error(
                    f"Gagal menginstal 'Orange3' secara otomatis. "
                    f"Silakan pastikan 'Orange3' tertulis di file requirements.txt Anda. Error: {fallback_error}"
                )
                st.stop()
# ==============================================================================

# Jalur dataset dan model di root directory (direktori utama proyek).
DATA_PATH = Path("02_realisasi_anggaran_klasifikasi.csv")
MODEL_PATH = Path("Best_model.pkcls")

st.set_page_config(
    page_title="Dashboard Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def detect_delimiter(text: str) -> str:
    """Mendeteksi delimiter CSV secara otomatis dari contoh teks."""
    try:
        return csv.Sniffer().sniff(text[:4000]).delimiter
    except:
        return ","


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    """Memuat dataset dan menambahkan label biner untuk target."""
    with open(path, "rb") as f:
        raw_data = f.read()
    
    text = raw_data.decode("utf-8", errors="replace")
    delimiter = detect_delimiter(text)
    
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    
    # Sinkronisasi nama kolom kementerian jika ada variasi nama kolom
    if "nama_kementerian" not in df.columns and "kementerian" in df.columns:
        df = df.rename(columns={"kementerian": "nama_kementerian"})
        
    if "realisasi_tercapai_95persen" in df.columns:
        df["target"] = df["realisasi_tercapai_95persen"].map({"Ya": 1, "Tidak": 0})
        
    for col in ["provinsi", "jenis_belanja_utama", "tipe_satker"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            
    return df


@st.cache_resource
def load_model(path: Path):
    """Memuat model Orange yang disimpan dalam file pickle."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_model_feature_names(model) -> list[str]:
    """Mengembalikan urutan fitur yang digunakan oleh model secara adaptif."""
    if hasattr(model, "domain"):
        return [attr.name for attr in model.domain.attributes]
    elif hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "skl_model") and hasattr(model.skl_model, "feature_names_in_"):
        return list(model.skl_model.feature_names_in_)
    else:
        return ["jumlah_spm", "revisi_dipa", "deviasi_rpd_persen", "skor_ikpa", "tipe_satker"]


def build_feature_matrix(df: pd.DataFrame | pd.Series, model) -> np.ndarray:
    """Membangun matriks fitur sesuai urutan model dengan aman."""
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    feature_names = get_model_feature_names(model)
    X = np.zeros((len(df), len(feature_names)), dtype=float)

    if len(df) == 0:
        return X

    for i, name in enumerate(feature_names):
        if name.startswith("tipe_satker="):
            tipe_nama = name.split("=", 1)[1]
            X[:, i] = (df["tipe_satker"] == tipe_nama).astype(float)
        elif name in df.columns:
            X[:, i] = pd.to_numeric(df[name], errors='coerce').fillna(0.0).astype(float)
        else:
            X[:, i] = 0.0

    return X


def predict_from_features(X: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """Menghasilkan prediksi kelas dan probabilitas target positif dalam bentuk 2D."""
    if hasattr(model, "skl_model"):
        predictor = model.skl_model
    elif hasattr(model, "predict"):
        predictor = model
    else:
        raise AttributeError("Model tidak memiliki atribut prediksi yang dikenal.")

    # Memastikan input yang masuk ke model Scikit-Learn berdimensi 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    prediction = predictor.predict(X).astype(int)
    probability = predictor.predict_proba(X)[:, 1]
    return prediction, probability


def build_manual_input_features(model) -> tuple[np.ndarray, dict]:
    """Membuat array fitur 2D dan metadata dari input manual di main section."""
    st.write("### Input Manual untuk Prediksi")
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        jumlah_spm = st.number_input(
            "Jumlah SPM", min_value=0, max_value=1000, value=30, step=1
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA", min_value=0, max_value=20, value=1, step=1
        )
        deviasi_rpd_persen = st.slider(
            "Deviasi RPD (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1
        )
    
    with col_in2:
        skor_ikpa = st.slider(
            "Skor IKPA", min_value=0.0, max_value=100.0, value=80.0, step=0.1
        )
        tipe_satker = st.selectbox(
            "Tipe Satker",
            ["Kantor Pusat", "Kantor Daerah", "Dekonsentrasi", "Tugas Pembantuan"],
        )
        kementerian = st.text_input("Kementerian", value="Kementan")

    input_data = {
        "jumlah_spm": jumlah_spm,
        "revisi_dipa": revisi_dipa,
        "deviasi_rpd_persen": deviasi_rpd_persen,
        "skor_ikpa": skor_ikpa,
        "tipe_satker": tipe_satker,
        "kementerian": kementerian,
    }

    feature_names = get_model_feature_names(model)
    row = np.zeros(len(feature_names), dtype=float)

    for i, name in enumerate(feature_names):
        if name.startswith("tipe_satker="):
            tipe_nama = name.split("=", 1)[1]
            row[i] = 1.0 if tipe_satker == tipe_nama else 0.0
        elif name in input_data and name not in ["tipe_satker", "kementerian"]:
            try:
                row[i] = float(input_data[name])
            except (ValueError, TypeError):
                row[i] = 0.0
        else:
            row[i] = 0.0

    # Mengubah bentuk array menjadi matriks 2D (1 Baris, N Kolom) agar tidak memicu ValueError dimensi
    row_2d = row.reshape(1, -1)
    return row_2d, input_data


# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
def main() -> None:
    st.title("Dashboard Realisasi Anggaran dan Prediksi 95%")
    st.markdown(
        "Dashboard ini menampilkan data anggaran, evaluasi model, dan prediksi interaktif untuk target `realisasi_tercapai_95persen`."
    )

    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di direktori utama: `{DATA_PATH.name}`")
        return
    except Exception as exc:
        st.error(f"Gagal memuat data: {exc}")
        return

    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di direktori utama: `{MODEL_PATH.name}`")
        return
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        return

    if df.empty:
        st.warning("Dataset kosong. Pastikan file CSV berisi data.")
        return

    X_all = build_feature_matrix(df, model)
    prediction_all, probability_all = predict_from_features(X_all, model)

    df = df.copy()
    df["prediksi"] = np.where(prediction_all == 1, "Ya,", "Tidak,")
    df["prediksi_proba"] = probability_all

    st.sidebar.header("Filter Data")
    provinsi_options = sorted(df["provinsi"].cat.categories)
    tipe_options = sorted(df["tipe_satker"].cat.categories)
    jenis_belanja_options = sorted(df["jenis_belanja_utama"].cat.categories)

    provinsi_filter = st.sidebar.multiselect(
        "Provinsi",
        provinsi_options,
        default=provinsi_options,
    )
    tipe_satker_filter = st.sidebar.multiselect(
        "Tipe Satker",
        tipe_options,
        default=tipe_options,
    )
    jenis_belanja_filter = st.sidebar.multiselect(
        "Jenis Belanja Utama",
        jenis_belanja_options,
        default=jenis_belanja_options,
    )

    filtered_df = df[
        df["provinsi"].isin(provinsi_filter)
        & df["tipe_satker"].isin(tipe_satker_filter)
        & df["jenis_belanja_utama"].isin(jenis_belanja_filter)
    ]

    if filtered_df.empty:
        st.warning("Filter menghasilkan 0 baris. Kurangi atau ubah filter untuk melihat data.")
        return

    st.subheader("Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Satker", len(filtered_df))
    col2.metric("Target Tercapai 95%", f"{filtered_df['target'].mean() * 100:.1f}%")
    col3.metric("Rata-rata Skor IKPA", f"{filtered_df['skor_ikpa'].mean():.2f}")
    col4.metric("Rata-rata Deviasi RPD", f"{filtered_df['deviasi_rpd_persen'].mean():.2f}%")

    with st.expander("📊 Statistik Deskriptif Fitur Numerik"):
        available_numeric = [
            col for col in [
                "pagu_miliar", "jumlah_pegawai", "jumlah_spm", "revisi_dipa",
                "realisasi_tw1_persen", "realisasi_tw2_persen", "realisasi_tw3_persen",
                "deviasi_rpd_persen", "skor_ikpa"
            ] if col in filtered_df.columns
        ]
        if available_numeric:
            st.dataframe(filtered_df[available_numeric].describe().T)

    st.subheader("Distribusi Target dan Probabilitas Prediksi")
    target_share = filtered_df.groupby("provinsi", observed=False)["target"].mean()
    pred_share = filtered_df.groupby("provinsi", observed=False)["prediksi_proba"].mean()

    chart_df = pd.DataFrame({"Target 95% Tercapai": target_share, "Probabilitas Prediksi": pred_share}).fillna(0)
    st.bar_chart(chart_df)

    if st.checkbox("Tampilkan data mentah yang difilter", value=True):
        st.dataframe(filtered_df.drop(columns=["target"], errors="ignore"))

    st.subheader("Evaluasi Model pada Seluruh Dataset")
    accuracy = (prediction_all == df["target"]).mean()
    st.write(f"Akurasi model terhadap seluruh dataset: **{accuracy * 100:.2f}%**")

    coef_values = getattr(model, "skl_model", model).coef_[0]
    coef_df = pd.DataFrame(
        {
            "Fitur": get_model_feature_names(model),
            "Koefisien": coef_values,
            "Kekuatan": np.abs(coef_values),
        }
    ).sort_values(by="Kekuatan", ascending=False)

    st.subheader("Interpretasi Model")
    st.dataframe(coef_df.reset_index(drop=True))
    st.bar_chart(coef_df.set_index("Fitur")["Koefisien"])

    st.subheader("Prediksi Interaktif")
    input_mode = st.radio("Pilih sumber input:", ["Pilih baris data", "Input manual"])

    if input_mode == "Pilih baris data":
        st.write("### Filter Pilih Baris")
        row_col1, row_col2, row_col3, row_col4 = st.columns(4)
        provinsi_pred = row_col1.selectbox("Provinsi untuk filter", ["Semua"] + provinsi_options, index=0)
        tipe_pred = row_col2.selectbox("Tipe Satker untuk filter", ["Semua"] + tipe_options, index=0)
        jenis_pred = row_col3.selectbox("Jenis Belanja untuk filter", ["Semua"] + jenis_belanja_options, index=0)
        kementerian_pred = row_col4.text_input("Kementerian untuk filter", "")

        sample_df = filtered_df.copy()
        if provinsi_pred != "Semua": sample_df = sample_df[sample_df["provinsi"] == provinsi_pred]
        if tipe_pred != "Semua": sample_df = sample_df[sample_df["tipe_satker"] == tipe_pred]
        if jenis_pred != "Semua": sample_df = sample_df[sample_df["jenis_belanja_utama"] == jenis_pred]
            
        kementerian_col = "nama_kementerian" if "nama_kementerian" in sample_df.columns else "kementerian"
        if kementerian_pred.strip() and kementerian_col in sample_df.columns:
            sample_df = sample_df[sample_df[kementerian_col].str.contains(kementerian_pred, case=False, na=False)]

        if sample_df.empty:
            st.warning("Filter prediksi tidak menghasilkan baris. Ubah kriteria filter.")
            return

        index = st.selectbox(
            "Pilih baris data:", sample_df.index,
            format_func=lambda i: f"{i} | {sample_df.loc[i, kementerian_col] if kementerian_col in sample_df.columns else 'N/A'} | {sample_df.loc[i,'provinsi']} | {sample_df.loc[i,'tipe_satker']}"
        )
        sample_row = sample_df.loc[index]
        st.write("### Data Terpilih")
        st.write(sample_row.to_frame().T)
        X_sample = build_feature_matrix(sample_row, model)
        pred, prob = predict_from_features(X_sample, model)
    else:
        X_sample, input_data = build_manual_input_features(model)
        pred, prob = predict_from_features(X_sample, model)

    if isinstance(pred, np.ndarray): pred = int(pred[0])
    if isinstance(prob, np.ndarray): prob = float(prob[0])

    st.divider()
    st.write("## Hasil Prediksi")
    
    result_col1, result_col2 = st.columns([2, 3])
    
    with result_col1:
        result_text = "✅ YA" if pred == 1 else "❌ TIDAK"
        result_color = "green" if pred == 1 else "red"
        st.markdown(f"### <span style='color:{result_color};font-size:2.5em;'>{result_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**Target tercapai 95%:** {'Kemungkinan Besar' if pred == 1 else 'Kemungkinan Kecil'}")
    
    with result_col2:
        st.write("### Tingkat Kepercayaan")
        prob_percentage = prob * 100
        st.metric(label="Probabilitas Positif", value=f"{prob_percentage:.1f}%")
        
        gauge_value = prob_percentage / 100
        st.progress(gauge_value, text=f"{prob_percentage:.1f}%")


if __name__ == "__main__":
    main()
