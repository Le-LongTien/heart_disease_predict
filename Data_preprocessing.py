# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix


# Đọc dữ liệu từ tệp CSV 
df = pd.read_csv('heart_2020.csv')
columns_to_convert = ['HeartDisease','BMI', 'Smoking','AlcoholDrinking', 'Stroke','PhysicalHealth','MentalHealth', 'DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer']
# Chỉnh sửa dữ liệu trong cột cụ thể và chuyển đổi "yes" thành 1 và "no" thành 0
for column in columns_to_convert:
    df[column] = df[column].replace({'Yes': 1, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3})

df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})
df['GenHealth'] = df['GenHealth'].replace({'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent':4})
df['Race'] = df['Race'].replace({'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3, 'American Indian/Alaskan Native': 4, 'Other': 5})
df['AgeCategory'] = df['AgeCategory'].replace({'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12})

# Thay thế bằng tên các cột bạn muốn xóa
columns_to_drop = ['PhysicalHealth', 'MentalHealth', 'DiffWalking']
# Xóa các cột không mong muốn
df.drop(columns=columns_to_drop, inplace=True)

# Đổi tên các cột
rename_columns = {
    'SleepTime':'Thoi gian ngu',
    'Smoking': 'Hut thuoc',
    'Stroke': 'Dot quy',
    'PhysicalActivity': 'Hoat dong the chat',
    'Asthma': 'Hen suyen',
    'KidneyDisease': 'Benh than',
    'SkinCancer': 'Ung thu da',
    'AlcoholDrinking': 'Ruou bia',
    'Diabetic': 'Tieu duong',
    'GenHealth': 'Suc khoe tong quat',
    'Race': 'Chung toc',
    'AgeCategory': 'Do tuoi',
    'Sex': 'Gioi tinh'    
}

df.rename(columns=rename_columns, inplace=True)
# Đổi thứ tự các cột theo thứ tự mong muốn
new_column_order = [
    'HeartDisease', 'Gioi tinh', 'Do tuoi', 'Chung toc', 'BMI',
    'Suc khoe tong quat', 'Hoat dong the chat', 'Thoi gian ngu', 'Hut thuoc',
    'Ruou bia', 'Dot quy', 'Tieu duong', 'Hen suyen', 'Benh than', 'Ung thu da'
]
df = df[new_column_order]
df.to_csv('heart_2020_cleaned1.csv', index=False)
# Lấy mẫu ngẫu nhiên 30.000 dữ liệu từ phần người không mắc bệnh tim
df_heart_disease = df[df['HeartDisease'] == 1]
df_no_heart_disease = df[df['HeartDisease'] == 0]
df_no_heart_disease_sampled = df_no_heart_disease.sample(n=30000, random_state=42)
# Kết hợp lại dữ liệu đã xử lý
df_balanced = pd.concat([df_heart_disease, df_no_heart_disease_sampled])
df_balanced.to_csv('heart_2020_cleaned1.csv', index=False)


