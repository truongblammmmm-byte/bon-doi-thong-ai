import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify

print("===========================================")
print("⏳ BƯỚC 1: ĐANG HUẤN LUYỆN LẠI AI DỰA TRÊN DỮ LIỆU SẠCH...")

# 1. Đọc dữ liệu trực tiếp từ file Excel 
df = pd.read_excel('du_lieu_sach_de_chay_AI.xlsx')

# 2. Định nghĩa chính xác 6 biến (features) và biến mục tiêu (target)
features = ['Product_ID', 'Month', 'Avg_Price', 'Lag_1', 'Lag_2', 'Lag_3']
target = 'Units Sold'

# 3. Loại bỏ các dòng chứa giá trị null
print(f"Số lượng dòng trước khi drop null: {len(df)}")
df = df.dropna(subset=features + [target])
print(f"Số lượng dòng sau khi drop null: {len(df)}")

# 4. Huấn luyện mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[features], df[target])
print(" ĐÃ TRAIN XONG! NÃO BỘ SẴN SÀNG TRÊN CLOUD.")

print("===========================================")
print(" BƯỚC 2: KHỞI ĐỘNG MÁY CHỦ API...")

# 5. Khởi tạo Flask App 
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
       
        row = [
            data.get('Product_ID', 1),
            data.get('Month', 11),
            data.get('Avg_Price', 110),
            data.get('Lag_1', 36000),
            data.get('Lag_2', 30000),
            data.get('Lag_3', 20000)
        ]
        # Bọc vào bảng để AI tính toán
        df_input = pd.DataFrame([row], columns=features)
        prediction = model.predict(df_input)
        
        return jsonify({"AI_DU_BAO": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# Lệnh chạy server cho môi trường Cloud
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
