import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. THU THẬP DỮ LIỆU
try:
    df = pd.read_csv('cs-training.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
except FileNotFoundError:
    print("❌ LỖI: Tệp 'cs-training.csv' không tồn tại.")
    exit()

# 2. ĐỊNH NGHĨA ĐẶC TRƯNG (Cần khai báo danh sách này)
FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
            'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
TARGET = 'SeriousDlqin2yrs'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# 3. TIỀN XỬ LÝ (Sửa lỗi imputer)
income_imputer = SimpleImputer(strategy='median')
X['MonthlyIncome'] = income_imputer.fit_transform(X[['MonthlyIncome']])

# Chuyển từ df sang numpy array sau khi fill missing cho cột phụ thuộc
dep_imputer = SimpleImputer(strategy='most_frequent')
X['NumberOfDependents'] = dep_imputer.fit_transform(X[['NumberOfDependents']])

# 4. CHUYỂN ĐỔI TIỀN TỆ
X['MonthlyIncome'] = X['MonthlyIncome'] * 25000

# 5. HUẤN LUYỆN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. ĐÁNH GIÁ
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"🏆 ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['An Toàn', 'Nợ Xấu']))

cm = confusion_matrix(y_test, y_pred)
print(f"📊 Ma trận: \n{cm}")

# 7. LƯU MÔ HÌNH
model_package = {'model': model, 'features': FEATURES}
with open('credit_model_v2.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✅ Hoàn tất! Tệp 'credit_model_v2.pkl' đã sẵn sàng.")