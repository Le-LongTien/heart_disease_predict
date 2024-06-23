import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tkinter as tk
from tkinter import ttk

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('heart_2020_cleaned_balanced 6.csv')

# Tách tập dữ liệu
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Tạo và huấn luyện VotingClassifier với tham số tối ưu
log_reg = LogisticRegression(C=0.03359818286283781, solver='liblinear', max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=28, weights='uniform')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('knn', knn),
        ('rf', rf)
    ],
    voting='soft'  # 'hard' để dùng majority voting, 'soft' để dùng probability voting
)

voting_clf.fit(X_train, y_train)

# Đánh giá VotingClassifier với Cross-validation
voting_score = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"Điểm VotingClassifier (Cross-Validation): {voting_score}")

# Hàm dự đoán
def predict_Heart_Disease(voting_clf, X):
    return voting_clf.predict(X)[0]

# Hàm hiển thị biểu đồ
def show_plots():
    print("Báo cáo phân loại cho VotingClassifier:")
    print(classification_report(y_test, voting_clf.predict(X_test)))
    #confusion_matrix
    voting_cm = confusion_matrix(y_test, voting_clf.predict(X_test))
    plt.figure(figsize=(10, 6))
    sns.heatmap(voting_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix for VotingClassifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    #ROC&AUC
    y_prob = voting_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'VotingClassifier (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    #learning_curve
    train_sizes, train_scores, valid_scores = learning_curve(
        voting_clf, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, valid_scores_mean, label='Validation score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

#show_plots()

# Tạo giao diện người dùng
class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự Đoán Bệnh Tim")
        
        # Danh sách các nhãn, giá trị mặc định và chú thích
        label_entries = [
            ('Gioi tinh', 0, '0: Nữ, 1: Nam'),
            ('Do tuoi', 0, '0: 18-24, 1: 25-29, 2: 30-34, 3: 35-39, 4: 40-44, 5: 45-49, 6: 50-54, 7: 55-59, 8: 60-64, 9: 65-69, 10: 70-74, 11: 75-79, 12: ngoài 80'),
            ('Chung toc', 0, '0: Trắng, 1: Đen, 2: Châu Á, 3: Hispanic, 4: Người bản địa, 5: chủng tộc khác'),
            ('BMI', 30, 'Chỉ số BMI'),
            ('Suc khoe tong quat', 2, '0: Kém, 1: Hơi kém , 2: Trung bình, 3: Khá, 4: Tốt'),
            ('Hoat dong the chat', 1, '0: Không, 1: Có'),
            ('Thoi gian ngu', 8, 'Thời gian ngủ trung bình mỗi đêm'),
            ('Hut thuoc', 1, '0: Không, 1: Có'),
            ('Ruou bia', 1, '0: Không, 1: Có'),
            ('Dot quy', 1, '0: Không, 1: Có'),
            ('Tieu duong', 1, '0: Không, 1: Có'),
            ('Hen suyen', 0, '0: Không, 1: Có'),
            ('Benh than', 0, '0: Không, 1: Có'),
            ('Ung thu da', 0, '0: Không, 1: Có')
        ]

        self.entries = {}
        for idx, (label, default_value, tooltip) in enumerate(label_entries):
            tk.Label(root, text=label).grid(row=idx, column=0, padx=10, pady=5, sticky=tk.W)
            entry = tk.Entry(root)
            entry.insert(0, str(default_value))
            entry.grid(row=idx, column=1, padx=10, pady=5)
            self.entries[label] = entry
            tk.Label(root, text=tooltip, fg="gray").grid(row=idx, column=2, padx=10, pady=5, sticky=tk.W)
        
        predict_button = tk.Button(root, text="Dự đoán", command=self.predict)
        predict_button.grid(row=len(label_entries), column=0, columnspan=3, pady=10)
        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=len(label_entries)+1, column=0, columnspan=3, pady=10)
        
    def predict(self):
        data = {label: float(entry.get()) for label, entry in self.entries.items()}
        data_for_prediction = pd.DataFrame([data])
        prediction_voting = predict_Heart_Disease(voting_clf, data_for_prediction)
        result_text = f"Dự đoán: {'Có bệnh tim' if prediction_voting == 1 else 'Không có bệnh tim'}"
        self.result_label.config(text=result_text)
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()