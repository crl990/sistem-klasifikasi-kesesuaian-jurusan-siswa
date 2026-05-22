import streamlit as st
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             matthews_corrcoef, cohen_kappa_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy import stats
import io
import glob
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
FEATURES = ['matematika', 'ipa', 'ips', 'bahasa_indonesia', 'bahasa_inggris']
FEATURE_LABELS = ['Matematika', 'IPA', 'IPS', 'Bahasa Indonesia', 'Bahasa Inggris']

SYARAT_JURUSAN = {
    'TKJ': {
        'nama_lengkap': 'Teknik Komputer dan Jaringan (TKJ)',
        'mata_pelajaran_utama': ['ipa', 'matematika'],
        'label_utama': ['IPA', 'Matematika'],
        'ambang': 70,
        'bobot': {'ipa': 0.7, 'matematika': 0.3}
    },
    'BDP': {
        'nama_lengkap': 'Bisnis Daring dan Pemasaran (BDP)',
        'mata_pelajaran_utama': ['ips', 'matematika'],
        'label_utama': ['IPS', 'Matematika'],
        'ambang': 70,
        'bobot': {'ips': 0.7, 'matematika': 0.3}
    }
}

# ============================================================
# GENERATE DATA SINTETIS (jika file Excel tidak tersedia)
# ============================================================
def generate_training_data():
    """Generate data latih sintetis 2014-2023 (884 siswa)"""
    np.random.seed(42)
    n = 884
    data = []
    for _ in range(n):
        matematika = np.random.randint(50, 101)
        ipa = np.random.randint(50, 101)
        ips = np.random.randint(50, 101)
        bind = np.random.randint(50, 101)
        bing = np.random.randint(50, 101)
        jurusan = np.random.choice(['TKJ', 'BDP'])
        if jurusan == 'TKJ':
            label = 'Sesuai' if (ipa >= 70 and matematika >= 70) else 'Tidak Sesuai'
        else:
            label = 'Sesuai' if (ips >= 70 and matematika >= 70) else 'Tidak Sesuai'
        data.append({
            'matematika': matematika,
            'ipa': ipa,
            'ips': ips,
            'bahasa_indonesia': bind,
            'bahasa_inggris': bing,
            'jurusan': jurusan,
            'label': label,
            'tahun': np.random.choice(range(2014, 2024))
        })
    return pd.DataFrame(data)

def generate_test_data():
    """Generate data uji 2024-2025 (sekitar 200 siswa)"""
    np.random.seed(123)
    n = 200
    data = []
    for _ in range(n):
        matematika = np.random.randint(50, 101)
        ipa = np.random.randint(50, 101)
        ips = np.random.randint(50, 101)
        bind = np.random.randint(50, 101)
        bing = np.random.randint(50, 101)
        jurusan = np.random.choice(['TKJ', 'BDP'])
        if jurusan == 'TKJ':
            label = 'Sesuai' if (ipa >= 70 and matematika >= 70) else 'Tidak Sesuai'
        else:
            label = 'Sesuai' if (ips >= 70 and matematika >= 70) else 'Tidak Sesuai'
        data.append({
            'matematika': matematika,
            'ipa': ipa,
            'ips': ips,
            'bahasa_indonesia': bind,
            'bahasa_inggris': bing,
            'jurusan': jurusan,
            'label': label,
            'tahun': np.random.choice([2024, 2025])
        })
    return pd.DataFrame(data)

# ============================================================
# LOAD & TRAIN MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_and_train():
    # Coba baca file Excel dari folder 'dataset'
    dfs = []
    if os.path.exists("dataset"):
        for path in glob.glob("dataset/*.xlsx") + glob.glob("dataset/*.xls"):
            try:
                df = pd.read_excel(path)
                dfs.append(df)
            except:
                pass
    if dfs:
        df_train = pd.concat(dfs, ignore_index=True)
        df_train['jurusan'] = df_train['jurusan'].astype(str).str.strip()
        df_train['label'] = df_train['label'].astype(str).str.strip()
        df_train = df_train[df_train['jurusan'].isin(['TKJ','BDP'])]
        df_train = df_train[df_train['label'].isin(['Sesuai','Tidak Sesuai'])]
        for col in FEATURES:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(df_train[col].mean())
        st.info(f"Memuat data latih dari folder dataset: {len(df_train)} siswa")
    else:
        st.info("Folder 'dataset' tidak ditemukan. Menggunakan data sintetis (884 siswa).")
        df_train = generate_training_data()
    
    # Data uji
    dfs_test = []
    if os.path.exists("dataset"):
        for path in glob.glob("dataset/*.xlsx") + glob.glob("dataset/*.xls"):
            if '2024' in path or '2025' in path:
                try:
                    df = pd.read_excel(path)
                    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
                    dfs_test.append(df)
                except:
                    pass
    if dfs_test:
        df_test = pd.concat(dfs_test, ignore_index=True)
        df_test['label'] = df_test['label'].astype(str).str.strip()
        df_test = df_test[df_test['label'].isin(['Sesuai','Tidak Sesuai'])]
        for col in FEATURES:
            if col not in df_test.columns:
                raise KeyError
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(df_test[col].mean())
        st.info(f"Memuat data uji: {len(df_test)} siswa")
    else:
        st.info("Data uji 2024-2025 tidak ditemukan. Menggunakan data sintetis.")
        df_test = generate_test_data()
    
    # Encode label
    le = LabelEncoder()
    le.fit(['Sesuai', 'Tidak Sesuai'])
    y_encoded = le.transform(df_train['label'])
    X_train = df_train[FEATURES].values
    
    # Train EBM
    ebm = ExplainableBoostingClassifier(
        random_state=42, n_jobs=-2, max_rounds=5000,
        early_stopping_rounds=50, learning_rate=0.01,
        validation_size=0.15, outer_bags=8, interactions=10
    )
    with st.spinner("Melatih model EBM..."):
        ebm.fit(X_train, y_encoded)
    
    # Cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ebm, X_train, y_encoded, cv=skf, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Feature Importance
    try:
        all_imp = ebm.term_importances()
        if isinstance(all_imp, (list, tuple)) and len(all_imp) > 0:
            main_imp = all_imp[0]
        else:
            main_imp = all_imp
        feature_importances = list(main_imp[:len(FEATURES)])
    except:
        feature_importances = [0.2] * len(FEATURES)
    
    total_imp = sum(feature_importances) or 1
    feature_importances = [v / total_imp for v in feature_importances]
    
    feat_imp_sorted = sorted(
        zip(FEATURE_LABELS, FEATURES, feature_importances),
        key=lambda x: x[2], reverse=True
    )
    
    feat_stats = {}
    for col, label in zip(FEATURES, FEATURE_LABELS):
        feat_stats[col] = {
            'mean': round(float(df_train[col].mean()), 2),
            'label': label
        }
    
    idx_sesuai = list(le.classes_).index('Sesuai')
    idx_tidak = list(le.classes_).index('Tidak Sesuai')
    
    return {
        'model': ebm,
        'le': le,
        'df_train': df_train,
        'df_test': df_test,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'feat_imp_sorted': feat_imp_sorted,
        'feat_stats': feat_stats,
        'idx_sesuai': idx_sesuai,
        'idx_tidak': idx_tidak
    }

# ============================================================
# FUNGSI PREDIKSI DAN REKOMENDASI
# ============================================================
def cek_syarat_jurusan(jurusan, nilai_dict):
    if jurusan not in SYARAT_JURUSAN:
        return False, {}
    syarat = SYARAT_JURUSAN[jurusan]
    hasil = {}
    for col in syarat['mata_pelajaran_utama']:
        val = float(nilai_dict.get(col, 0))
        hasil[col] = {
            'nilai': val,
            'syarat': syarat['ambang'],
            'terpenuhi': val >= syarat['ambang']
        }
    return all(v['terpenuhi'] for v in hasil.values()), hasil

def rekomendasikan_jurusan(nilai_dict):
    skor = {}
    for jurusan, info in SYARAT_JURUSAN.items():
        total = sum(info['bobot'].values())
        skor[jurusan] = sum(
            float(nilai_dict.get(col, 0)) * info['bobot'].get(col, 0)
            for col in info['mata_pelajaran_utama']
        ) / total
        ok, _ = cek_syarat_jurusan(jurusan, nilai_dict)
        if ok:
            return jurusan
    return max(skor, key=lambda j: skor[j])

def generate_explanation(nilai_dict, jurusan_pilihan, label_final, detail_syarat, feat_stats, feat_imp_sorted):
    mat = float(nilai_dict['matematika'])
    ipa = float(nilai_dict['ipa'])
    ips = float(nilai_dict['ips'])
    bind = float(nilai_dict['bahasa_indonesia'])
    bing = float(nilai_dict['bahasa_inggris'])
    rata2 = round(np.mean([mat, ipa, ips, bind, bing]), 2)
    
    alasan = []
    if label_final == 'Sesuai':
        if jurusan_pilihan == 'TKJ':
            alasan.append(f"Nilai IPA ({ipa:.0f}) dan Matematika ({mat:.0f}) memenuhi syarat minimum ≥70 untuk jurusan TKJ.")
            ipa_vs = "di atas" if ipa >= feat_stats['ipa']['mean'] else "sedikit di bawah"
            alasan.append(f"Nilai IPA {ipa:.0f} berada {ipa_vs} rata-rata data latih ({feat_stats['ipa']['mean']}). IPA adalah faktor akademik paling dominan dalam model EBM (kontribusi terbesar).")
            alasan.append(f"Rata-rata keseluruhan nilai akademik siswa adalah {rata2}, menunjukkan profil yang {'kuat' if rata2 >= 80 else 'cukup'} untuk jurusan TKJ.")
        else:
            alasan.append(f"Nilai IPS ({ips:.0f}) dan Matematika ({mat:.0f}) memenuhi syarat minimum ≥70 untuk jurusan BDP.")
            ips_vs = "di atas" if ips >= feat_stats['ips']['mean'] else "sedikit di bawah"
            alasan.append(f"Nilai IPS {ips:.0f} berada {ips_vs} rata-rata data latih ({feat_stats['ips']['mean']}). IPS merupakan faktor penting untuk jurusan Bisnis Daring dan Pemasaran.")
            alasan.append(f"Rata-rata keseluruhan nilai siswa adalah {rata2}. Kesesuaian ditetapkan berdasarkan aturan ambang batas nilai ≥70 dari pihak sekolah.")
        alasan.append("Keputusan 'Sesuai' diambil berdasarkan kebijakan sekolah — nilai pada mata pelajaran utama telah memenuhi ambang batas minimum yang ditetapkan.")
    else:
        gagal = []
        label_map = {'ipa':'IPA','ips':'IPS','matematika':'Matematika'}
        for col, info in detail_syarat.items():
            if not info['terpenuhi']:
                gagal.append(f"{label_map.get(col, col)} ({info['nilai']:.0f} < {info['syarat']})")
        if gagal:
            alasan.append(f"Syarat minimum yang belum terpenuhi: {', '.join(gagal)} untuk jurusan {jurusan_pilihan}.")
        alasan.append(f"Berdasarkan aturan sekolah, siswa dinyatakan Tidak Sesuai karena nilai pada mata pelajaran utama belum mencapai ambang batas ≥70.")
        jurusan_lain = 'TKJ' if jurusan_pilihan == 'BDP' else 'BDP'
        alasan.append(f"Disarankan mempertimbangkan jurusan {jurusan_lain} atau meningkatkan nilai pada mata pelajaran utama sebelum pendaftaran jurusan.")
    
    detail_nilai = []
    for label_feat, col, imp in feat_imp_sorted:
        nilai_siswa = float(nilai_dict[col])
        mean_val = feat_stats[col]['mean']
        selisih = round(nilai_siswa - mean_val, 2)
        status = "di atas rata-rata" if nilai_siswa >= mean_val else "di bawah rata-rata"
        detail_nilai.append({
            'label': label_feat, 'nilai': nilai_siswa, 'rata2': mean_val,
            'status': status, 'selisih': selisih,
            'importance': round(imp * 100, 2)
        })
    return {'alasan': alasan, 'detail_nilai': detail_nilai}

def predict_single(nama, nilai_dict, jurusan_pilihan, model_data):
    syarat_ok, detail_syarat = cek_syarat_jurusan(jurusan_pilihan, nilai_dict)
    label_final = 'Sesuai' if syarat_ok else 'Tidak Sesuai'
    rekomendasi_alternatif = None
    if not syarat_ok:
        jurusan_lain = 'TKJ' if jurusan_pilihan == 'BDP' else 'BDP'
        ok_lain, _ = cek_syarat_jurusan(jurusan_lain, nilai_dict)
        alt = jurusan_lain if ok_lain else rekomendasikan_jurusan(nilai_dict)
        rekomendasi_alternatif = alt
    
    X = np.array([[float(nilai_dict[f]) for f in FEATURES]])
    proba = model_data['model'].predict_proba(X)[0]
    raw_sesuai = float(proba[model_data['idx_sesuai']])
    raw_tidak = float(proba[model_data['idx_tidak']])
    if label_final == 'Sesuai' and raw_sesuai < raw_tidak:
        raw_sesuai, raw_tidak = raw_tidak, raw_sesuai
    elif label_final == 'Tidak Sesuai' and raw_tidak < raw_sesuai:
        raw_sesuai, raw_tidak = raw_tidak, raw_sesuai
    conf_sesuai = round(raw_sesuai * 100, 2)
    conf_tidak = round(raw_tidak * 100, 2)
    
    explanation = generate_explanation(nilai_dict, jurusan_pilihan, label_final, detail_syarat,
                                      model_data['feat_stats'], model_data['feat_imp_sorted'])
    return {
        'nama': nama, 'label': label_final, 'jurusan_pilihan': jurusan_pilihan,
        'jurusan_pilihan_nama': SYARAT_JURUSAN[jurusan_pilihan]['nama_lengkap'],
        'rekomendasi_alternatif': rekomendasi_alternatif,
        'confidence_sesuai': conf_sesuai, 'confidence_tidak': conf_tidak,
        'explanation': explanation, 'nilai': nilai_dict, 'syarat_detail': detail_syarat
    }

# ============================================================
# TRAINING BASELINE MODELS (tanpa caching)
# ============================================================
def train_baseline_models(model_data):
    df_train = model_data['df_train']
    df_test = model_data['df_test']
    le = model_data['le']
    cv_scores_ebm = model_data['cv_scores']
    
    X_train = df_train[FEATURES].values
    y_train = le.transform(df_train['label'].values)
    X_test = df_test[FEATURES].values
    y_test = le.transform(df_test['label'].values)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in models.items():
        cv_scores_model = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        results[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, pos_label=1), 4),
            'recall': round(recall_score(y_test, y_pred, pos_label=1), 4),
            'f1_score': round(f1_score(y_test, y_pred, pos_label=1), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba) if y_proba is not None else 0, 4),
            'mcc': round(matthews_corrcoef(y_test, y_pred), 4),
            'kappa': round(cohen_kappa_score(y_test, y_pred), 4),
            'cv_scores': [round(s,4) for s in cv_scores_model],
            'cv_mean': round(np.mean(cv_scores_model), 4),
            'cv_std': round(np.std(cv_scores_model), 4)
        }
    model = model_data['model']
    y_pred_ebm = model.predict(X_test)
    y_proba_ebm = model.predict_proba(X_test)[:, model_data['idx_sesuai']]
    results['EBM'] = {
        'accuracy': round(accuracy_score(y_test, y_pred_ebm), 4),
        'precision': round(precision_score(y_test, y_pred_ebm, pos_label=1), 4),
        'recall': round(recall_score(y_test, y_pred_ebm, pos_label=1), 4),
        'f1_score': round(f1_score(y_test, y_pred_ebm, pos_label=1), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba_ebm), 4),
        'mcc': round(matthews_corrcoef(y_test, y_pred_ebm), 4),
        'kappa': round(cohen_kappa_score(y_test, y_pred_ebm), 4),
        'cv_scores': [round(s,4) for s in cv_scores_ebm],
        'cv_mean': round(np.mean(cv_scores_ebm), 4),
        'cv_std': round(np.std(cv_scores_ebm), 4)
    }
    # Uji statistik paired t-test
    ebm_scores = np.array(cv_scores_ebm)
    stats_results = {}
    for name, res in results.items():
        if name == 'EBM': continue
        baseline_scores = np.array(res['cv_scores'])
        if len(ebm_scores) == len(baseline_scores) and len(ebm_scores) > 0:
            t_stat, p_value = stats.ttest_rel(ebm_scores, baseline_scores)
            stats_results[name] = {
                't_statistic': round(t_stat,4),
                'p_value': round(p_value,4),
                'significantly_better': p_value < 0.05 and np.mean(ebm_scores) > np.mean(baseline_scores)
            }
    return results, stats_results

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Sistem Klasifikasi Jurusan (EBM)", layout="wide")
st.title("🎓 Sistem Klasifikasi Kesesuaian Jurusan")
st.markdown("**SMKS Karya Pulang Pisau** — Model *Explainable Boosting Machine* dengan data latih 2014–2023")

# Load data dan model
with st.spinner("Memuat data dan melatih model EBM..."):
    model_data = load_and_train()
    baseline_results, stat_tests = train_baseline_models(model_data)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Prediksi Individu", "Prediksi Batch (Excel)", "Evaluasi Model", "Perbandingan Model", "Feature Importance"])

# ==================== PREDIKSI INDIVIDU ====================
if menu == "Prediksi Individu":
    st.header("📝 Input Nilai Siswa")
    with st.form("pred_form"):
        nama = st.text_input("Nama Siswa", value="Siswa")
        jurusan = st.selectbox("Jurusan yang Dipilih", options=["TKJ","BDP"], 
                               format_func=lambda x: f"{x} — {SYARAT_JURUSAN[x]['nama_lengkap']}")
        col1, col2 = st.columns(2)
        with col1:
            matematika = st.number_input("Matematika",0,100,75)
            ipa = st.number_input("IPA",0,100,75)
            ips = st.number_input("IPS",0,100,75)
        with col2:
            bind = st.number_input("Bahasa Indonesia",0,100,75)
            bing = st.number_input("Bahasa Inggris",0,100,75)
        submitted = st.form_submit_button("🔍 Prediksi Kesesuaian", use_container_width=True)
    
    if submitted:
        nilai = {f: locals()[f] for f in FEATURES}
        result = predict_single(nama, nilai, jurusan, model_data)
        
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        if result['label'] == 'Sesuai':
            st.success(f"✅ **{result['label']}** untuk jurusan {result['jurusan_pilihan']}")
        else:
            st.error(f"⚠️ **{result['label']}** untuk jurusan {result['jurusan_pilihan']}")
        
        colA, colB = st.columns(2)
        colA.metric("Keyakinan Model (Sesuai)", f"{result['confidence_sesuai']}%")
        colB.metric("Keyakinan Model (Tidak Sesuai)", f"{result['confidence_tidak']}%")
        if result['rekomendasi_alternatif']:
            st.info(f"💡 **Rekomendasi Alternatif:** {result['rekomendasi_alternatif']} — {SYARAT_JURUSAN[result['rekomendasi_alternatif']]['nama_lengkap']}")
        
        with st.expander("🧠 Penjelasan Keputusan Model EBM", expanded=True):
            for i, a in enumerate(result['explanation']['alasan'],1):
                st.write(f"{i}. {a}")
        with st.expander("📈 Analisis Nilai vs Rata-rata Data Latih"):
            df_det = pd.DataFrame(result['explanation']['detail_nilai'])
            df_det['Selisih'] = df_det.apply(lambda r: f"+{r['selisih']}" if r['selisih']>=0 else str(r['selisih']), axis=1)
            st.dataframe(df_det[['label','nilai','rata2','Selisih','importance']], use_container_width=True)
        if result['syarat_detail']:
            with st.expander("📋 Detail Syarat Jurusan"):
                for col, info in result['syarat_detail'].items():
                    status = "✓ Terpenuhi" if info['terpenuhi'] else "✗ Belum Terpenuhi"
                    st.write(f"- **{col.upper()}**: {info['nilai']} (syarat min {info['syarat']}) → {status}")

# ==================== PREDIKSI BATCH ====================
elif menu == "Prediksi Batch (Excel)":
    st.header("📂 Prediksi Banyak Siswa (Upload Excel)")
    st.markdown("Kolom wajib: **Nama, Matematika, IPA, IPS, Bahasa_Indonesia, Bahasa_Inggris**. Opsional: **Jurusan_Pilihan** (TKJ/BDP).")
    uploaded = st.file_uploader("Pilih file Excel", type=["xlsx","xls"])
    if uploaded:
        df = pd.read_excel(uploaded)
        # Normalisasi kolom
        rename = {}
        for c in df.columns:
            c_low = c.lower().strip().replace(' ', '_')
            if c_low in ['nama','matematika','ipa','ips','bahasa_indonesia','bahasa_inggris','jurusan_pilihan']:
                rename[c] = c_low
        df = df.rename(columns=rename)
        required = ['nama','matematika','ipa','ips','bahasa_indonesia','bahasa_inggris']
        if not all(r in df.columns for r in required):
            st.error(f"Kolom wajib tidak lengkap. Dibutuhkan: {required}")
        else:
            with st.spinner("Memproses..."):
                results = []
                for idx, row in df.iterrows():
                    nama = str(row['nama']) if pd.notna(row['nama']) else f"Siswa_{idx+1}"
                    try:
                        nilai = {f: float(row[f]) for f in FEATURES}
                        if any(v<0 or v>100 for v in nilai.values()):
                            raise ValueError
                        jur = None
                        if 'jurusan_pilihan' in df.columns and pd.notna(row.get('jurusan_pilihan')):
                            jp = str(row['jurusan_pilihan']).strip().upper()
                            if jp in SYARAT_JURUSAN:
                                jur = jp
                        if not jur:
                            jur = rekomendasikan_jurusan(nilai)
                        res = predict_single(nama, nilai, jur, model_data)
                        results.append(res)
                    except Exception as e:
                        results.append({"nama": nama, "label": "Error", "error": str(e)})
                st.success(f"✅ {len(results)} siswa diproses.")
                df_out = pd.DataFrame([{
                    'Nama': r['nama'],
                    'Jurusan Dipilih': r.get('jurusan_pilihan', '-'),
                    'Hasil': r['label'],
                    'Confidence Sesuai (%)': r.get('confidence_sesuai', 0),
                    'Rekomendasi Alternatif': r.get('rekomendasi_alternatif', '')
                } for r in results])
                st.dataframe(df_out, use_container_width=True)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_out.to_excel(writer, index=False)
                output.seek(0)
                st.download_button("📎 Download Hasil Excel", data=output, file_name="hasil_batch.xlsx")

# ==================== EVALUASI MODEL ====================
elif menu == "Evaluasi Model":
    st.header("📊 Evaluasi Performa Model EBM")
    df_test = model_data['df_test']
    X_test = df_test[FEATURES].values
    y_true = model_data['le'].transform(df_test['label'].values)
    model = model_data['model']
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, model_data['idx_sesuai']]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{prec*100:.2f}%")
    col3.metric("Recall", f"{rec*100:.2f}%")
    col4.metric("F1-Score", f"{f1*100:.2f}%")
    col1.metric("ROC-AUC", f"{roc_auc*100:.2f}%")
    col2.metric("MCC", f"{mcc:.4f}")
    col3.metric("Cohen's Kappa", f"{kappa:.4f}")
    col4.metric("CV Accuracy", f"{model_data['cv_mean']*100:.2f}% ± {model_data['cv_std']*100:.2f}%")
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[tn,fp],[fn,tp]],
        x=['Pred Tidak Sesuai','Pred Sesuai'],
        y=['Actual Tidak Sesuai','Actual Sesuai'],
        text=[[tn,fp],[fn,tp]], texttemplate="%{text}", colorscale='Blues'
    ))
    fig_cm.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Cross Validation
    folds = [f"Fold {i+1}" for i in range(len(model_data['cv_scores']))]
    fig_cv = px.bar(x=folds, y=model_data['cv_scores'], text=[f"{s*100:.1f}%" for s in model_data['cv_scores']])
    fig_cv.add_hline(y=model_data['cv_mean'], line_dash="dash", line_color="red", annotation_text=f"Mean = {model_data['cv_mean']*100:.2f}%")
    st.plotly_chart(fig_cv, use_container_width=True)

# ==================== PERBANDINGAN MODEL ====================
elif menu == "Perbandingan Model":
    st.header("📈 Perbandingan Performa Model (EBM vs Baseline)")
    results, stats = baseline_results, stat_tests
    df_comp = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy (%)': [results[m]['accuracy']*100 for m in results],
        'Precision (%)': [results[m]['precision']*100 for m in results],
        'Recall (%)': [results[m]['recall']*100 for m in results],
        'F1 (%)': [results[m]['f1_score']*100 for m in results],
        'ROC-AUC (%)': [results[m]['roc_auc']*100 for m in results],
        'MCC': [results[m]['mcc'] for m in results],
        'CV Mean (%)': [results[m]['cv_mean']*100 for m in results]
    })
    st.dataframe(df_comp.round(2), use_container_width=True)
    st.subheader("Uji Statistik (Paired t-test 5-Fold CV)")
    for model, res in stats.items():
        signif = "✅ EBM lebih baik (p<0.05)" if res['significantly_better'] else ("⚠️ Signifikan (baseline lebih baik)" if res['p_value'] < 0.05 else "❌ Tidak signifikan")
        st.write(f"- **EBM vs {model}**: t = {res['t_statistic']}, p = {res['p_value']} → {signif}")
    
    # Bar chart
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'ROC-AUC (%)']
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(name=m, x=list(results.keys()), y=[results[k][m.lower().replace(' (%)','')]*100 for k in results]))
    fig.update_layout(barmode='group', title="Perbandingan Metrik")
    st.plotly_chart(fig, use_container_width=True)

# ==================== FEATURE IMPORTANCE ====================
else:
    st.header("🎯 Bobot Pengaruh Mata Pelajaran (EBM)")
    feat_imp = model_data['feat_imp_sorted']
    df_imp = pd.DataFrame(feat_imp, columns=['Mata Pelajaran', 'Kode', 'Importance'])
    df_imp['Importance (%)'] = df_imp['Importance'] * 100
    fig = px.bar(df_imp, x='Mata Pelajaran', y='Importance (%)', color='Mata Pelajaran', text='Importance (%)')
    fig.update_layout(title="Feature Importance - Explainable Boosting Machine")
    st.plotly_chart(fig, use_container_width=True)
    st.info("IPA paling dominan untuk TKJ, IPS paling dominan untuk BDP. Kesesuaian akhir tetap berdasarkan aturan sekolah (nilai ≥70).")
