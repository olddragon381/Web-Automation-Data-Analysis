import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

def random_forest_model():
    st.header("Random Forest")

    df = st.session_state.get("df")


    if df is not None:
        st.write("### Dataset Preview")
        st.write(df.head())
        
        st.subheader("Thống kê mô tả")
        st.write(df.describe())

        st.subheader("Kiểm tra dữ liệu thiếu")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0])

        if missing_data.sum() == 0:
            st.write("Không có dữ liệu thiếu.")

        # Phân tích dữ liệu cột
        st.subheader("Phân tích cột")
        column_to_analyze = st.selectbox("Chọn cột để phân tích", df.columns)
        # Kiểm tra và xử lý dữ liệu DateTime
        st.subheader("Phân tích Dữ liệu DateTime")
        datetime_columns = []

        # Xác định cột DateTime
        for col in df.columns:
            try:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    pd.to_datetime(df[col], errors='raise')  # Thử chuyển đổi sang DateTime
                    datetime_columns.append(col)
            except Exception:
                pass

        if datetime_columns:
            st.write("Các cột được nhận diện là DateTime:", datetime_columns)
            datetime_col = st.selectbox("Chọn cột DateTime để phân tích", datetime_columns)
            
            # Chuyển đổi cột sang kiểu DateTime
            df[datetime_col] = pd.to_datetime(df[datetime_col])

            # Tạo các đặc trưng mới từ cột DateTime
            df['Year'] = df[datetime_col].dt.year
            df['Month'] = df[datetime_col].dt.month
            df['Day'] = df[datetime_col].dt.day
            df['DayOfWeek'] = df[datetime_col].dt.dayofweek
            df['Hour'] = df[datetime_col].dt.hour

            st.write("Dữ liệu sau khi thêm các đặc trưng từ DateTime:")
            st.dataframe(df[[datetime_col, 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour']].head())

            # Phân tích theo thời gian
            st.subheader("Phân tích dữ liệu thời gian")
            time_aggregation = st.selectbox("Chọn cách tổng hợp", ["Theo năm", "Theo tháng", "Theo ngày trong tuần"])
            
            if time_aggregation == "Theo năm":
                st.bar_chart(df.groupby('Year').size())
            elif time_aggregation == "Theo tháng":
                st.bar_chart(df.groupby('Month').size())
            elif time_aggregation == "Theo ngày trong tuần":
                st.bar_chart(df.groupby('DayOfWeek').size())
        else:
            st.write("Không tìm thấy cột nào có kiểu DateTime.")



        if datetime_columns:
            st.write("Các cột được nhận diện là DateTime:", datetime_columns)
            datetime_col = st.selectbox("Chọn cột DateTime để phân tích", datetime_columns)

            # Chuyển đổi cột sang kiểu DateTime
            df[datetime_col] = pd.to_datetime(df[datetime_col])

            # Chọn cột khác để vẽ biểu đồ theo DateTime
            st.subheader("Vẽ biểu đồ theo DateTime")
            other_columns = [col for col in df.columns if col != datetime_col]
            selected_col = st.selectbox("Chọn cột để vẽ", other_columns)

            if df[selected_col].dtype in ["float64", "int64"]:
                fig, ax = plt.subplots()
                df.sort_values(datetime_col).plot(x=datetime_col, y=selected_col, ax=ax)
                plt.xlabel("Thời gian")
                plt.ylabel(selected_col)
                plt.title(f"Biểu đồ {selected_col} theo {datetime_col}")
                st.pyplot(fig)
            else:
                st.write("Cột được chọn không phù hợp để vẽ biểu đồ.")
        else:
            st.write("Không tìm thấy cột nào có kiểu DateTime.")

        if pd.api.types.is_datetime64_any_dtype(df[column_to_analyze]):
            st.write("Phân tích dữ liệu thời gian:")
            st.line_chart(df[column_to_analyze].value_counts().sort_index())

        if df[column_to_analyze].dtype in ["float64", "int64"]:
                st.write("Phân phối giá trị trong cột:")
                fig, ax = plt.subplots()
                sns.histplot(df[column_to_analyze], kde=True, ax=ax)
                st.pyplot(fig)

        elif df[column_to_analyze].dtype == "object":
            st.write("Tần suất các giá trị trong cột:")
            st.bar_chart(df[column_to_analyze].value_counts())



     
        # Hiển thị heatmap nếu dữ liệu đủ nhỏ
        if df.shape[1] <= 20:
            st.subheader("Heatmap Tương quan")
            numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
            selected_cols = st.multiselect("Chọn các cột để vẽ heatmap", numerical_cols, default=numerical_cols)

            if len(selected_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.write("Vui lòng chọn ít nhất hai cột số để vẽ heatmap.")

        
        



        
    if df is not None:
        st.write("### Dataset Preview")
        st.write(df.head())

        # Kiểm tra các cột
        numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        st.write(f"Numerical Columns: {list(numerical_cols)}")
        st.write(f"Categorical Columns: {list(categorical_cols)}")

        # Chọn feature và target
        features = st.multiselect("Select Features", options=numerical_cols)
        target = st.selectbox("Select Target", options=numerical_cols)

        if features and target:
            X = df[features]
            y = df[target]
            
            # Chia tập dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Huấn luyện mô hình
         

            # Huấn luyện mô hình
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Đánh giá mô hình
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.write("### Model Performance")
            st.write(f"Mean Squared Error: {mse}")

            # Biểu đồ Actual vs Predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            start = st.slider("Chọn vị trí bắt đầu", 0, len(y_test) - 1, 0)  # Thanh trượt để chọn vị trí bắt đầu
            end = st.slider("Chọn vị trí kết thúc", start + 1, len(y_test), min(len(y_test), start + 50))  # Kết thúc không quá xa

            # Cắt đoạn dữ liệu
            y_test_subset = y_test[start:end]
            y_pred_subset = y_pred[start:end]

            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            ax.plot(range(start, end), y_test_subset, label="Actual", color="blue")  # Dòng thực tế
            ax.plot(range(start, end), y_pred_subset, label="Predicted", color="orange")  # Dòng dự đoán

            ax.set_xlabel("Index")  # Nhãn trục x
            ax.set_ylabel("Value")  # Nhãn trục y
            ax.set_title("Actual vs Predicted (Subset)")  # Tiêu đề biểu đồ
            ax.legend(loc="best")  # Hiển thị chú thích (Actual, Predicted)

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

            # Lưu mô hình
            if st.button("Save Model"):
                joblib.dump(model, "Random_Forest_model.pkl")
                st.success("Model saved as 'Random_Forest_model.pkl'.")
        else:
            if not features:
                st.warning("Please select at least one feature.")
            if not target:
                st.warning("Please select a target variable.")
    else:
        st.info("Please upload a dataset first.")

def main():
    st.title("Random Forest with Streamlit")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.success("Dataset has been uploaded successfully!")

    random_forest_model()

if __name__ == "__main__":
    main()
