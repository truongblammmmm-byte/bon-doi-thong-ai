import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify

print("===========================================")
print("⏳ BƯỚC 1: ĐANG HUẤN LUYỆN LẠI AI CHUẨN 6 BIẾN...")

# 1. Đọc và dọn dẹp dữ liệu (Đảm bảo file CSV nằm cùng thư mục với file app.py)
df = pd.read_csv('dataadidas.xlsx - Data Sales Adidas.csv')

for col in ['Units Sold', 'Price per Unit']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
df['Month'] = df['Invoice Date'].dt.month
df = df.sort_values(by=['Product', 'Invoice Date'])
df['Product_ID'] = df['Product'].astype('category').cat.codes
df = df.rename(columns={'Price per Unit': 'Avg_Price'})

df['Lag_1'] = df.groupby('Product_ID')['Units Sold'].shift(1)
df['Lag_2'] = df.groupby('Product_ID')['Units Sold'].shift(2)
df['Lag_3'] = df.groupby('Product_ID')['Units Sold'].shift(3)
df = df.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3', 'Avg_Price', 'Units Sold'])

features = ['Product_ID', 'Month', 'Lag_1', 'Lag_2', 'Lag_3', 'Avg_Price']
X = df[features]
y = df['Units Sold']

# 2. Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ ĐÃ TRAIN XONG! NÃO BỘ SẴN SÀNG TRÊN CLOUD.")

print("===========================================")
print("🚀 BƯỚC 2: KHỞI ĐỘNG MÁY CHỦ API...")

# 3. Khởi tạo Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Lấy đúng 6 biến từ n8n
        row = [
            data.get('Product_ID', 1),
            data.get('Month', 1),
            data.get('Lag_1', 0),
            data.get('Lag_2', 0),
            data.get('Lag_3', 0),
            data.get('Avg_Price', 0)
        ]
        # Bọc vào bảng để AI tính toán
        df_input = pd.DataFrame([row], columns=features)
        prediction = model.predict(df_input)
        
        return jsonify({"AI_DU_BAO": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# Lệnh chạy server cho môi trường Cloud
if __name__ == '__main__':
    # host='0.0.0.0' giúp máy chủ Cloud mở luồng nhận dữ liệu từ bên ngoài
    app.run(host='0.0.0.0', port=5000)