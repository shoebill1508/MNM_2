# ============================================================
# APP.PY - HỆ THỐNG TÍNH TOÁN ĐIỂM TÍN DỤNG (v3 - CCCD)
# ============================================================
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
st.write(os.getcwd())
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# 1. CẤU HÌNH GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="Hệ Thống Đánh Giá Điểm Tín Dụng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE   = 'credit_ledger_v2.csv'
CCCD_FILE      = 'cs-training_CCCD.xlsx'
CCCD_SHEET     = 'cs-training_cot_B'

# ============================================================
# 2. TẢI MÔ HÌNH
# ============================================================
@st.cache_resource
def load_ml_model():
    model_path = 'credit_model_v2.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_pack = load_ml_model()

if model_pack is None:
    st.error("❌ CẢNH BÁO LỖI HỆ THỐNG: Không tìm thấy tệp mô hình 'credit_model_v2.pkl'.")
    st.info("💡 Hướng dẫn khắc phục: Chạy lệnh `python train_model.py` trước khi khởi động giao diện.")
    st.stop()

rf_model     = model_pack['model']
SYS_FEATURES = model_pack['features']
INCOME_MEDIAN = model_pack.get('income_median', 5400.0)
DEP_MODE      = model_pack.get('dep_mode', 0.0)

# ============================================================
# 3. TẢI DỮ LIỆU CCCD (cache để không đọc lại mỗi lần)
# ============================================================
@st.cache_data(show_spinner="Đang tải cơ sở dữ liệu CCCD...")
def load_cccd_db():
    """Đọc file CCCD xlsx, trả về dict {cccd_str: row_dict}"""
    if not os.path.exists(CCCD_FILE):
        return {}
    df = pd.read_excel(CCCD_FILE, sheet_name=CCCD_SHEET)
    # Cột đầu là index vô nghĩa, bỏ đi
    if df.columns[0] not in ['CCCD', 'cccd']:
        df = df.iloc[:, 1:]   # bỏ cột index
    df['CCCD'] = df['CCCD'].astype(str).str.strip()
    return df.set_index('CCCD').to_dict(orient='index')

CCCD_DB = load_cccd_db()

def lookup_cccd(cccd_str: str):
    """
    Trả về (status, data_dict)
    status: 'found' | 'not_found' | 'db_missing'
    """
    if not CCCD_DB:
        return 'db_missing', {}
    key = str(cccd_str).strip()
    if key in CCCD_DB:
        return 'found', CCCD_DB[key]
    return 'not_found', {}

# Ánh xạ tên cột CCCD → tên field trong form
CCCD_TO_FORM = {
    'RevolvingUtilizationOfUnsecuredLines': 'utilization',   # 0-1 → nhân 100 để hiển thị %
    'age':                                  'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'delinq_30',
    'DebtRatio':                            'debt_ratio',    # raw, không nhân 100
    'MonthlyIncome':                        'income',        # USD gốc
    'NumberOfOpenCreditLinesAndLoans':      'lines_open',
    'NumberOfTimes90DaysLate':              'delinq_90',
    'NumberRealEstateLoansOrLines':         'mortgage',
    'NumberOfTime60-89DaysPastDueNotWorse': 'delinq_60',
    'NumberOfDependents':                   'dependents',
}

# ============================================================
# 4. TẢI DỮ LIỆU NỀN TẢNG (CHO DASHBOARD)
# ============================================================
@st.cache_data
def fetch_empirical_data():
    try:
        raw_data = pd.read_csv('cs-training.csv')
        if 'Unnamed: 0' in raw_data.columns:
            raw_data = raw_data.drop(columns=['Unnamed: 0'])
        raw_data['Phân Loại'] = raw_data['SeriousDlqin2yrs'].map(
            {0: 'Khách hàng An Toàn', 1: 'Nhóm Nợ Xấu (>90 ngày)'}
        )
        return raw_data
    except FileNotFoundError:
        return pd.DataFrame()

analytics_df = fetch_empirical_data()

# ============================================================
# 5. THANH ĐIỀU HƯỚNG (SIDEBAR)
# ============================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
st.sidebar.title("Quản Trị Chức Năng")
st.sidebar.markdown("---")

tab_dashboard = st.sidebar.checkbox("📊 Bảng Điều Khiển Thống Kê", value=True)
tab_logbook   = st.sidebar.checkbox("📜 Sổ Ghi Nhận Lịch Sử",      value=False)
tab_ai_spec   = st.sidebar.checkbox("🤖 Phân Tích Mô Hình AI",      value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Bảo Trì Hệ Thống")
if st.sidebar.button("🗑️ Xóa Nhật Ký Lịch Sử"):
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        st.sidebar.success("Đã xóa toàn bộ lịch sử.")
    else:
        st.sidebar.info("Không có lịch sử để xóa.")

# ============================================================
# 6. TIÊU ĐỀ CHÍNH
# ============================================================
st.title("🏦 NỀN TẢNG PHÂN TÍCH VÀ XẾP HẠNG TÍN DỤNG CÁ NHÂN")
st.markdown(
    f"*Hệ thống sử dụng thuật toán Random Forest với **{len(SYS_FEATURES)} đặc trưng**, "
    f"được huấn luyện trên 150.000 hồ sơ khách hàng thực tế.*"
)

# ============================================================
# 7. DASHBOARD THỐNG KÊ
# ============================================================
if tab_dashboard:
    st.divider()
    st.subheader("📊 Tổng Quan Dữ Liệu Nền Tảng")

    if analytics_df.empty:
        st.warning(
            "⚠️ **Không tải được dữ liệu nền tảng.** "
            "Tệp `cs-training.csv` không tồn tại hoặc không thể đọc. "
            "Các biểu đồ thống kê sẽ không hiển thị cho đến khi có file này."
        )
    else:
        total_records     = len(analytics_df)
        baseline_risk     = analytics_df['SeriousDlqin2yrs'].mean() * 100
        mean_age          = analytics_df['age'].mean()
        median_debt_ratio = analytics_df.loc[analytics_df['DebtRatio'] < 5, 'DebtRatio'].median()

        metrics = st.columns(4)
        metrics[0].metric("👥 Tổng Hồ Sơ",             f"{total_records:,} hồ sơ")
        metrics[1].metric("⚠️ Tỷ lệ Nợ Xấu Cơ Sở",    f"{baseline_risk:.2f}%")
        metrics[2].metric("🕰️ Độ Tuổi Trung Bình",     f"{mean_age:.0f} tuổi")
        metrics[3].metric("⚖️ Tỷ Lệ Nợ/TN (Trung Vị)", f"{median_debt_ratio*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.markdown("**🥧 Phân Bố Chất Lượng Tín Dụng**")
            safe_count = (analytics_df['SeriousDlqin2yrs'] == 0).sum()
            risk_count = (analytics_df['SeriousDlqin2yrs'] == 1).sum()
            pie_chart = go.Figure(go.Pie(
                labels=['Khách hàng An Toàn', 'Nhóm Nợ Xấu (>90 ngày)'],
                values=[safe_count, risk_count],
                hole=0.5,
                marker_colors=['#27ae60', '#c0392b'],
                textinfo='label+percent'
            ))
            pie_chart.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False)
            st.plotly_chart(pie_chart, use_container_width=True)

        with plot_col2:
            st.markdown("**📈 Tỷ Lệ Vỡ Nợ Theo Nhóm Tuổi**")
            age_bins = pd.cut(
                analytics_df['age'],
                bins=[0, 30, 45, 60, 100],
                labels=['18-30', '31-45', '46-60', '60+']
            )
            age_risk = analytics_df.groupby(age_bins, observed=True)['SeriousDlqin2yrs'].mean() * 100
            bar_chart = px.bar(
                x=age_risk.index.astype(str),
                y=age_risk.values,
                labels={'x': 'Nhóm Tuổi', 'y': 'Tỷ lệ Vỡ Nợ (%)'},
                color=age_risk.values,
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
            bar_chart.update_traces(texttemplate='%{text}%', textposition='outside')
            bar_chart.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=350,
                coloraxis_showscale=False
            )
            st.plotly_chart(bar_chart, use_container_width=True)

# ============================================================
# 8. PHÂN TÍCH MÔ HÌNH AI
# ============================================================
if tab_ai_spec:
    st.divider()
    st.subheader("🤖 Mức Độ Quan Trọng Của Các Đặc Trưng (Feature Importance)")

    mapping_dict = {
        'RevolvingUtilizationOfUnsecuredLines': '1. Tỷ lệ Lấp đầy Hạn mức Thẻ TD',
        'DebtRatio':                            '2. Hệ số Nợ / Thu nhập (DTI)',
        'MonthlyIncome':                        '3. Thu nhập Hàng tháng',
        'age':                                  '4. Tuổi tác',
        'NumberOfTimes90DaysLate':              '5. Nợ xấu (>90 ngày)',
        'NumberOfTime30-59DaysPastDueNotWorse': '6. Trễ hạn ngắn (30-59 ngày)',
        'NumberOfOpenCreditLinesAndLoans':      '7. Số Hợp đồng Vay mở',
        'NumberOfTime60-89DaysPastDueNotWorse': '8. Trễ hạn trung (60-89 ngày)',
        'NumberRealEstateLoansOrLines':         '9. Vay Bất động sản',
        'NumberOfDependents':                   '10. Số Người Phụ Thuộc'
    }

    importance_data = pd.DataFrame({
        'feature': SYS_FEATURES,
        'importance': rf_model.feature_importances_
    })
    importance_data['label'] = importance_data['feature'].map(mapping_dict)
    importance_data = importance_data.sort_values('importance', ascending=True)

    imp_chart = px.bar(
        importance_data,
        x='importance', y='label',
        orientation='h',
        color='importance',
        color_continuous_scale=['#b2bec3', '#0984e3', '#2d3436'],
        text=importance_data['importance'].apply(lambda x: f"{x:.1%}")
    )
    imp_chart.update_traces(textposition='outside')
    imp_chart.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=450,
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title=""
    )
    st.plotly_chart(imp_chart, use_container_width=True)
    st.info(
        "💡 Kết quả cho thấy **Tỷ lệ lấp đầy hạn mức thẻ tín dụng (Revolving Utilization)** "
        "là yếu tố dự báo vỡ nợ quan trọng nhất — vượt trên cả thu nhập hàng tháng."
    )

# ============================================================
# 9. TRA CỨU CCCD (TRƯỚC FORM NHẬP TAY)
# ============================================================
st.divider()
st.subheader("🪪 Tra Cứu Hồ Sơ Theo Số CCCD / Căn Cước")

cccd_col1, cccd_col2 = st.columns([3, 1])
with cccd_col1:
    cccd_input = st.text_input(
        "Nhập số CCCD (12 chữ số):",
        placeholder="VD: 475220433343",
        max_chars=20,
        help="Hệ thống sẽ tự động điền thông tin vào biểu mẫu nếu tìm thấy hồ sơ."
    )
with cccd_col2:
    st.write("")  # căn chỉnh button ngang bằng với text input
    cccd_search_btn = st.button("🔍 Tra Cứu", use_container_width=True)

# --- Session state để lưu dữ liệu prefill ---
if 'prefill' not in st.session_state:
    st.session_state['prefill'] = {}
if 'cccd_status' not in st.session_state:
    st.session_state['cccd_status'] = None
if 'cccd_queried' not in st.session_state:
    st.session_state['cccd_queried'] = ''

if cccd_search_btn:
    if not cccd_input.strip():
        st.error("🚫 **Bắt buộc nhập số CCCD.** Vui lòng nhập đầy đủ số Căn Cước Công Dân trước khi tiếp tục.")
        st.stop()
    status, row = lookup_cccd(cccd_input.strip())
    st.session_state['cccd_status']  = status
    st.session_state['cccd_queried'] = cccd_input.strip()

    if status == 'found':
        # Lưu dữ liệu để prefill form bên dưới
        st.session_state['prefill'] = row
        st.success(
            f"✅ **Tìm thấy hồ sơ CCCD `{cccd_input.strip()}`** trong cơ sở dữ liệu. "
            f"Thông tin đã được điền tự động vào biểu mẫu bên dưới. "
            f"Bạn có thể chỉnh sửa trước khi chạy phân tích."
        )
        # --- Hiển thị bảng so sánh nhanh ---
        db_income_vnd = row.get('MonthlyIncome', 0) * 25000
        db_util_pct   = row.get('RevolvingUtilizationOfUnsecuredLines', 0) * 100
        db_dti_pct    = row.get('DebtRatio', 0) * 100

        compare_cols = st.columns(4)
        compare_cols[0].metric("📅 Tuổi (DB)",           f"{row.get('age', 'N/A')} tuổi")
        compare_cols[1].metric("💵 Thu nhập (DB)",        f"{db_income_vnd:,.0f} VNĐ")
        compare_cols[2].metric("📊 Tỷ lệ dùng thẻ (DB)", f"{db_util_pct:.1f}%")
        compare_cols[3].metric("⚖️ DTI (DB)",             f"{db_dti_pct:.1f}%")

        # Cờ rủi ro từ DB label SeriousDlqin2yrs
        db_label = row.get('SeriousDlqin2yrs', None)
        if db_label == 1:
            st.warning(
                "⚠️ **Cảnh báo lịch sử:** Theo dữ liệu huấn luyện, CCCD này **đã từng có "
                "nợ quá hạn nghiêm trọng (>90 ngày)** trong quá khứ. Cần thẩm định thêm."
            )
        elif db_label == 0:
            st.info("🟢 Lịch sử tín dụng trong DB: **Không có nợ xấu nghiêm trọng** được ghi nhận.")

    elif status == 'not_found':
        st.session_state['prefill'] = {}
        st.error(
            f"❌ **Không tìm thấy CCCD `{cccd_input.strip()}`** trong hệ thống. \n\n"
            f"Khách hàng chưa có hồ sơ. Vui lòng **điền tay thông tin bên dưới** "
            f"để hệ thống tạo hồ sơ mới và lưu lại."
        )
    elif status == 'db_missing':
        st.session_state['prefill'] = {}
        st.warning(
            f"⚠️ File dữ liệu CCCD `{CCCD_FILE}` không tồn tại trên server. "
            f"Vui lòng liên hệ quản trị viên."
        )

# ============================================================
# 10. BIỂU MẪU THẨM ĐỊNH (prefill nếu có dữ liệu CCCD)
# ============================================================
st.divider()
st.subheader("📝 Bảng Khai Báo Thông Tin Thẩm Định Tín Dụng")

# ── Bắt buộc phải tra cứu CCCD trước khi hiển thị form ──
_cccd_ready = bool(st.session_state.get('cccd_queried', '').strip())
if not _cccd_ready:
    st.warning(
        "🔒 **Bạn chưa tra cứu CCCD.**\n\n"
        "Vui lòng nhập số Căn Cước Công Dân ở ô phía trên và nhấn **🔍 Tra Cứu** "
        "để hệ thống xác minh danh tính trước khi thẩm định tín dụng."
    )
    st.stop()

# Lấy giá trị prefill (nếu CCCD tìm thấy) hoặc default
pf = st.session_state.get('prefill', {})

# Thu nhập: DB lưu USD gốc, form hiển thị VNĐ (× 25000)
default_income      = int(pf.get('MonthlyIncome', 5000) * 25000) if pf else 125_000_000
default_age         = int(pf.get('age', 30)) if pf else 30
default_dependents  = int(pf.get('NumberOfDependents', 0)) if pf else 0
default_utilization = int(round(pf.get('RevolvingUtilizationOfUnsecuredLines', 0.3) * 100)) if pf else 30
default_lines_open  = int(pf.get('NumberOfOpenCreditLinesAndLoans', 5)) if pf else 5
# DTI từ DB: DebtRatio = debt/income → monthly_debt = DebtRatio * MonthlyIncome * 25000
_db_dti   = pf.get('DebtRatio', 0.24) if pf else 0.24
_db_inc   = pf.get('MonthlyIncome', 5000) * 25000 if pf else 125_000_000
default_monthly_debt = int(_db_dti * _db_inc)
default_mortgage    = int(pf.get('NumberRealEstateLoansOrLines', 0)) if pf else 0
default_delinq_30   = int(pf.get('NumberOfTime30-59DaysPastDueNotWorse', 0)) if pf else 0
default_delinq_60   = int(pf.get('NumberOfTime60-89DaysPastDueNotWorse', 0)) if pf else 0
default_delinq_90   = int(pf.get('NumberOfTimes90DaysLate', 0)) if pf else 0

# Banner thông báo nếu form đang được prefill
if pf:
    st.info(
        f"ℹ️ **Form đã được tự động điền** từ hồ sơ CCCD `{st.session_state.get('cccd_queried', '')}`. "
        f"Các giá trị bên dưới phản ánh dữ liệu trong hệ thống — bạn có thể điều chỉnh nếu cần."
    )

with st.form("credit_assessment_form"):
    st.markdown("##### 👤 1. Thông Tin Cá Nhân & Thu Nhập")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        in_age        = st.number_input("Độ tuổi:", min_value=18, max_value=99, value=default_age)
    with col_b:
        in_income     = st.number_input("Thu nhập hàng tháng (VNĐ):", value=default_income)
    with col_c:
        in_dependents = st.number_input("Số người phụ thuộc:", min_value=0, max_value=15, value=default_dependents)

    st.markdown("##### 💳 2. Thông Tin Tín Dụng & Đòn Bẩy")
    col_d, col_e = st.columns(2)
    with col_d:
        st.markdown("**Thẻ tín dụng & Vay tiêu dùng**")
        in_utilization = st.slider(
            "Tỷ lệ sử dụng hạn mức thẻ (%):", min_value=0, max_value=200,
            value=min(default_utilization, 200),
            help="Dư nợ thẻ / Tổng hạn mức × 100. >70% là nguy hiểm."
        )
        in_lines_open  = st.number_input("Số hợp đồng vay / thẻ đang mở:", min_value=0, value=default_lines_open)
    with col_e:
        st.markdown("**Nghĩa vụ nợ cố định**")
        in_monthly_debt = st.number_input(
            "Tổng tiền trả nợ hàng tháng (VNĐ):",
            min_value=0, value=default_monthly_debt, step=1_000_000,
            help="Bao gồm gốc/lãi vay ngân hàng, trả góp, nợ thẻ..."
        )
        in_mortgage = st.number_input("Số hợp đồng vay có tài sản đảm bảo:", min_value=0, value=default_mortgage)

        if in_income > 0:
            dti_value = in_monthly_debt / in_income
        else:
            dti_value = 0.0

        st.info(f"🔄 Hệ số DTI (Nợ/Thu nhập) tự động: **{dti_value*100:.1f}%** ({dti_value:.2f})")

        # --- So sánh với dữ liệu DB nếu có ---
        if pf:
            db_dti_calc = pf.get('DebtRatio', None)
            if db_dti_calc is not None:
                diff_dti = abs(dti_value - db_dti_calc)
                if diff_dti > 0.15:
                    st.warning(
                        f"⚠️ **DTI nhập ({dti_value*100:.1f}%) lệch đáng kể** so với hồ sơ DB "
                        f"({db_dti_calc*100:.1f}%). Kiểm tra lại thu nhập hoặc tổng nợ."
                    )

    st.markdown("##### ⚠️ 3. Lịch Sử Trễ Hạn (24 tháng qua)")
    col_f, col_g, col_h = st.columns(3)
    with col_f:
        in_delinq_30 = st.number_input("Số lần quá hạn 30–59 ngày:", min_value=0, value=default_delinq_30)
    with col_g:
        in_delinq_60 = st.number_input("Số lần quá hạn 60–89 ngày:", min_value=0, value=default_delinq_60)
    with col_h:
        in_delinq_90 = st.number_input("Số lần quá hạn ≥90 ngày:",   min_value=0, value=default_delinq_90)

    # --- So sánh trễ hạn với DB ---
    if pf:
        db_30 = int(pf.get('NumberOfTime30-59DaysPastDueNotWorse', 0))
        db_60 = int(pf.get('NumberOfTime60-89DaysPastDueNotWorse', 0))
        db_90 = int(pf.get('NumberOfTimes90DaysLate', 0))
        mismatch_msgs = []
        if in_delinq_30 != db_30:
            mismatch_msgs.append(f"Trễ 30-59 ngày: nhập **{in_delinq_30}** ≠ DB **{db_30}**")
        if in_delinq_60 != db_60:
            mismatch_msgs.append(f"Trễ 60-89 ngày: nhập **{in_delinq_60}** ≠ DB **{db_60}**")
        if in_delinq_90 != db_90:
            mismatch_msgs.append(f"Trễ ≥90 ngày: nhập **{in_delinq_90}** ≠ DB **{db_90}**")
        if mismatch_msgs:
            st.warning(
                "⚠️ **Phát hiện lệch dữ liệu trễ hạn so với hồ sơ DB:**\n\n"
                + "\n\n".join(f"- {m}" for m in mismatch_msgs)
                + "\n\nHệ thống sẽ **ưu tiên giá trị bạn vừa nhập** để tính điểm."
            )

    run_inference = st.form_submit_button("🚀 CHẠY PHÂN TÍCH VÀ TÍNH ĐIỂM TÍN DỤNG", use_container_width=True)

# ============================================================
# 11. ĐỘNG CƠ DỰ BÁO
# ============================================================
if run_inference:
    utilization_ratio = in_utilization / 100.0

    input_data = [[
        utilization_ratio,
        in_age,
        in_delinq_30,
        dti_value,
        in_income,
        in_lines_open,
        in_delinq_90,
        in_mortgage,
        in_delinq_60,
        in_dependents
    ]]

    inference_vector = pd.DataFrame(input_data, columns=SYS_FEATURES)

    pd_prob_arr  = rf_model.predict_proba(inference_vector)
    pd_prob_risk = float(pd_prob_arr[0][1])

    credit_score = int(np.clip(150 + (1 - pd_prob_risk) * 600, 150, 750))

    st.divider()
    st.subheader(f"🎯 KẾT QUẢ THẨM ĐỊNH: **{credit_score} ĐIỂM CIC**")

    # --- Banner so sánh kết quả vs DB label ---
    if pf:
        db_label = pf.get('SeriousDlqin2yrs', None)
        if db_label == 1 and credit_score >= 570:
            st.warning(
                "⚠️ **Lưu ý không nhất quán:** Mô hình cho điểm khá cao nhưng hồ sơ DB "
                "ghi nhận lịch sử **nợ xấu nghiêm trọng**. Khuyến nghị thẩm định bổ sung."
            )
        elif db_label == 0 and credit_score < 430:
            st.info(
                "ℹ️ Mô hình cho điểm thấp dù DB không ghi nhận nợ xấu. "
                "Có thể do thông số nhập hiện tại khác với thời điểm lịch sử."
            )

    if credit_score >= 680:
        st.success(
            "✅ **HẠNG A – RỦI RO CỰC THẤP (680–750 điểm)**\n\n"
            "Hồ sơ sở hữu lịch sử thanh toán hoàn hảo và đòn bẩy tài chính kiểm soát tốt. "
            "→ Đề xuất: **Phê duyệt ngay** với ưu đãi hạn mức và giảm lãi suất."
        )
        status_log = "Duyệt – Ưu Đãi Hạng A"
    elif credit_score >= 570:
        st.info(
            "🟢 **HẠNG B – RỦI RO THẤP (570–679 điểm)**\n\n"
            "Sức khỏe tài chính tổng thể ở mức tốt, tỷ lệ đòn bẩy trong giới hạn kiểm soát. "
            "→ Đề xuất: **Phê duyệt theo quy trình thông thường**."
        )
        status_log = "Duyệt – Tiêu Chuẩn Hạng B"
    elif credit_score >= 430:
        st.warning(
            "⚠️ **HẠNG C – RỦI RO TRUNG BÌNH CAO (430–569 điểm)**\n\n"
            "Mô hình nhận diện tín hiệu căng thẳng dòng tiền hoặc dư nợ thẻ tiệm cận hạn mức. "
            "→ Đề xuất: **Treo hồ sơ**, yêu cầu bổ sung tài sản thế chấp hoặc người bảo lãnh."
        )
        status_log = "Đình Chỉ – Cần Bổ Sung"
    else:
        st.error(
            "🚫 **HẠNG D – RỦI RO TỐI ĐA (dưới 430 điểm)**\n\n"
            "Hồ sơ trùng khớp với mẫu vỡ nợ hệ thống (lịch sử trễ hạn dày đặc hoặc DTI mất kiểm soát). "
            "→ Đề xuất: **Từ chối giải ngân** để bảo toàn vốn."
        )
        status_log = "Từ Chối Giải Ngân"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📉 Xác suất Vỡ Nợ",  f"{pd_prob_risk*100:.1f}%")
    m2.metric("💳 Tỷ lệ Dùng Thẻ",  f"{utilization_ratio*100:.0f}%")
    m3.metric("⚖️ Hệ số DTI",        f"{dti_value*100:.0f}%")
    m4.metric("🚨 Tổng Lần Trễ Hạn", f"{in_delinq_30 + in_delinq_60 + in_delinq_90} lần")

    # ── Ghi vào Ledger (Đọc → Nối → Ghi đè an toàn) ──
    now_stamp  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ledger_row = pd.DataFrame([{
        'Thời gian':        now_stamp,
        'CCCD':             st.session_state.get('cccd_queried', '') or 'Nhập tay',
        'Tuổi':             in_age,
        'Thu nhập (VNĐ)':   in_income,
        'DTI':              round(dti_value, 4),
        'Tỷ lệ dùng thẻ':  round(utilization_ratio, 4),
        'Trễ hạn 30-59':   in_delinq_30,
        'Trễ hạn 60-89':   in_delinq_60,
        'Trễ hạn ≥90':     in_delinq_90,
        'Người phụ thuộc': in_dependents,
        'Vay BĐS':         in_mortgage,
        'Xác suất vỡ nợ':  round(pd_prob_risk, 4),
        'Điểm CIC':        credit_score,
        'Quyết định':      status_log,
    }])

    try:
        if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 5:
            existing = pd.read_csv(HISTORY_FILE, encoding='utf-8-sig', on_bad_lines='skip')
            existing.columns = existing.columns.str.strip()
            # Chỉ nối nếu cấu trúc cột hợp lệ
            if 'Quyết định' in existing.columns:
                updated_data = pd.concat([existing, ledger_row], ignore_index=True)
            else:
                updated_data = ledger_row
        else:
            updated_data = ledger_row

        updated_data.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
        st.caption(f"💾 *Hồ sơ đã được lưu vào nhật ký hệ thống (Timestamp: {now_stamp}).*")
    except Exception as e:
        st.warning(f"⚠️ Không thể lưu lịch sử: {e}")

# ============================================================
# 12. SỔ LỊCH SỬ
# ============================================================
if tab_logbook:
    st.divider()
    st.subheader("📜 Lịch Sử Thẩm Định")

    df_columns = [
        "Thời gian", "CCCD", "Tuổi", "Thu nhập (VNĐ)", "DTI", "Tỷ lệ dùng thẻ",
        "Trễ hạn 30-59", "Trễ hạn 60-89", "Trễ hạn ≥90", "Người phụ thuộc",
        "Vay BĐS", "Xác suất vỡ nợ", "Điểm CIC", "Quyết định"
    ]
    ledger_data = pd.DataFrame(columns=df_columns)

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        try:
            temp_data = pd.read_csv(HISTORY_FILE, on_bad_lines='skip', encoding='utf-8-sig')
            temp_data.columns = temp_data.columns.str.strip()
            if 'Quyết định' in temp_data.columns:
                ledger_data = temp_data
        except Exception as e:
            st.error(f"Lỗi đọc file lịch sử: {e}")

    if not ledger_data.empty:
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Tổng số truy vấn", len(ledger_data))
        duyet_count   = ledger_data['Quyết định'].astype(str).str.contains('Duyệt', na=False).sum()
        tu_choi_count = ledger_data['Quyết định'].astype(str).str.contains('Từ Chối|Đình Chỉ', na=False).sum()
        kpi2.metric("Số hồ sơ được duyệt", duyet_count)
        kpi3.metric("Đình chỉ / Từ chối",  tu_choi_count)

        # Tạo bản sao để hiển thị với định dạng thân thiện hơn
        display_df = ledger_data.copy()
        # Format số thực thành phần trăm dễ đọc cho cột DTI và Tỷ lệ dùng thẻ
        for col in ['DTI', 'Tỷ lệ dùng thẻ']:
            if col in display_df.columns:
                try:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').apply(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else ''
                    )
                except Exception:
                    pass
        # Format xác suất vỡ nợ
        if 'Xác suất vỡ nợ' in display_df.columns:
            try:
                display_df['Xác suất vỡ nợ'] = pd.to_numeric(display_df['Xác suất vỡ nợ'], errors='coerce').apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else ''
                )
            except Exception:
                pass
        # Format thu nhập
        if 'Thu nhập (VNĐ)' in display_df.columns:
            try:
                display_df['Thu nhập (VNĐ)'] = pd.to_numeric(display_df['Thu nhập (VNĐ)'], errors='coerce').apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else ''
                )
            except Exception:
                pass

        csv_export = ledger_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải Báo Cáo Lịch Sử (CSV)",
            data=csv_export,
            file_name=f"lich_su_tham_dinh_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        st.dataframe(display_df.tail(20), use_container_width=True)
    else:
        st.info("Chưa có hồ sơ nào được thẩm định hoặc file dữ liệu đang trống. Hãy sử dụng biểu mẫu bên trên để tạo dữ liệu mới.")
