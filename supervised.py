
Missing value count plotting for all the features in the dataset (marks : 5) 


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset into a pandas DataFrame
# df = pd.read_csv("train.csv")

# # Calculate the number of missing values for each feature
# missing_values = df.isnull().sum()

# # Create a bar plot of the missing values
# missing_values.plot(kind="bar")
# plt.title("Missing Value Count Plot")
# plt.xlabel("Feature")
# plt.ylabel("Number of Missing Values")
# plt.show()




Find the measure of central tedency of ApplicantIncome,CoapplicantIncome,LoanAmount and its value
# import pandas as pd

# # Load the dataset into a pandas DataFrame
# df = pd.read_csv("train.csv")

# # Calculate the mean, median, and mode for ApplicantIncome
# applicant_income_mean = df["ApplicantIncome"].mean()
# applicant_income_median = df["ApplicantIncome"].median()
# applicant_income_mode = df["ApplicantIncome"].mode()[0]

# print(f"ApplicantIncome Mean: {applicant_income_mean}")
# print(f"ApplicantIncome Median: {applicant_income_median}")
# print(f"ApplicantIncome Mode: {applicant_income_mode}")

# # Calculate the mean, median, and mode for CoapplicantIncome
# coapplicant_income_mean = df["CoapplicantIncome"].mean()
# coapplicant_income_median = df["CoapplicantIncome"].median()
# coapplicant_income_mode = df["CoapplicantIncome"].mode()[0]

# print(f"CoapplicantIncome Mean: {coapplicant_income_mean}")
# print(f"CoapplicantIncome Median: {coapplicant_income_median}")
# print(f"CoapplicantIncome Mode: {coapplicant_income_mode}")

# # Calculate the mean, median, and mode for LoanAmount
# loan_amount_mean = df["LoanAmount"].mean()
# loan_amount_median = df["LoanAmount"].median()
# loan_amount_mode = df["LoanAmount"].mode()[0]

# print(f"LoanAmount Mean: {loan_amount_mean}")
# print(f"LoanAmount Median: {loan_amount_median}")
# print(f"LoanAmount Mode: {loan_amount_mode}")





Find the % of married men and women who are graduate of total men and total women
# import pandas as pd


# df = pd.read_csv("train.csv")


# total_men = df[df["Gender"] == "Male"].shape[0]
# total_women = df[df["Gender"] == "Female"].shape[0]


# married_graduate_men = df[(df["Gender"] == "Male") & (df["Married"] == "Yes") & (df["Education"] == "Graduate")].shape[0]
# married_graduate_women = df[(df["Gender"] == "Female") & (df["Married"] == "Yes") & (df["Education"] == "Graduate")].shape[0]


# percentage_married_graduate_men = (married_graduate_men / total_men) * 100
# percentage_married_graduate_women = (married_graduate_women / total_women) * 100

# print(f"Percentage of married men who are graduates: {percentage_married_graduate_men}%")
# print(f"Percentage of married women who are graduates: {percentage_married_graduate_women}%")








For which category of property area, loan is approved most of the time
# import pandas as pd


# df = pd.read_csv("train.csv")

# print(df.isnull().sum())

# approved_loans_by_area = df[df["Loan_Status"] == "Y"]["Property_Area"].value_counts()


# print(approved_loans_by_area)


# most_approved_area = approved_loans_by_area.index[0]

# print(f"The category of property area with the most approved loans is: {most_approved_area}")








Plot the  relation b/w married and unmarried man against the loan status

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("train.csv")


# df["Married"] = df["Married"].apply(lambda x: "Married" if x == "Yes" else "Unmarried")


# pivot_table = df.pivot_table(values="Loan_ID", index="Married", columns="Loan_Status", aggfunc="count")


# pivot_table.plot(kind="bar", stacked=True)
# plt.title("Relationship between Married and Unmarried Men and Loan Status")
# plt.xlabel("Marital Status")
# plt.ylabel("Number of Loans")
# plt.show()



How to handle the inbalance data set, write at least three approaches    -> in theory.txt




 What are top 3 most important features for loan prediction and find their feature importance score 

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# # Load the dataset
# df = pd.read_csv("train.csv")

# # Handle missing values
# mode = df['Gender'].value_counts().index[0]
# df['Gender'] = df['Gender'].fillna(mode)

# # Encode categorical variables
# le = LabelEncoder()
# for column in df.select_dtypes(include='object'):
#     df[column] = le.fit_transform(df[column])


# df.dropna(subset=['LoanAmount','Loan_Amount_Term','Credit_History'], inplace=True)


# print(df.isnull().sum())







# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)



# lr_y_pred = lr_model.predict(X_test)
# lr_accuracy = accuracy_score(y_test, lr_y_pred)
# lr_precision = precision_score(y_test, lr_y_pred)
# lr_recall = recall_score(y_test, lr_y_pred)
# lr_f1 = f1_score(y_test, lr_y_pred)
# lr_report = classification_report(y_test, lr_y_pred)
# lr_confusion_matrix = confusion_matrix(y_test, lr_y_pred)

# print("Logistic Regression:")
# print("Accuracy:", lr_accuracy)
# print("Precision:", lr_precision)
# print("Recall:", lr_recall)
# print("F1 Score:", lr_f1)
# print("Classification Report:\n", lr_report)
# print("Confusion Matrix:\n", lr_confusion_matrix)



## the three important feature is 
# loan amount
# Credit_History 
# Self_Employed







 Train the model with Logistic Regression, Decision Tree and Knn and find which model has any of overfitting and biasing problem for major class 

# # 10  marks
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# # Load the dataset
# df = pd.read_csv("train.csv")

# # Handle missing values
# mode = df['Gender'].value_counts().index[0]
# df['Gender'] = df['Gender'].fillna(mode)

# # Encode categorical variables
# le = LabelEncoder()
# for column in df.select_dtypes(include='object'):
#     df[column] = le.fit_transform(df[column])


# df.dropna(subset=['LoanAmount','Loan_Amount_Term','Credit_History'], inplace=True)


# print(df.isnull().sum())







# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)


# dt_model = DecisionTreeClassifier()
# dt_model.fit(X_train, y_train)


# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train, y_train)


# lr_y_pred = lr_model.predict(X_test)
# lr_accuracy = accuracy_score(y_test, lr_y_pred)
# lr_precision = precision_score(y_test, lr_y_pred)
# lr_recall = recall_score(y_test, lr_y_pred)
# lr_f1 = f1_score(y_test, lr_y_pred)
# lr_report = classification_report(y_test, lr_y_pred)
# lr_confusion_matrix = confusion_matrix(y_test, lr_y_pred)

# print("Logistic Regression:")
# print("Accuracy:", lr_accuracy)
# print("Precision:", lr_precision)
# print("Recall:", lr_recall)
# print("F1 Score:", lr_f1)
# print("Classification Report:\n", lr_report)
# print("Confusion Matrix:\n", lr_confusion_matrix)

# dt_y_pred = dt_model.predict(X_test)
# dt_accuracy = accuracy_score(y_test, dt_y_pred)
# dt_precision = precision_score(y_test, dt_y_pred)
# dt_recall = recall_score(y_test, dt_y_pred)
# dt_f1 = f1_score(y_test, dt_y_pred)
# dt_report = classification_report(y_test, dt_y_pred)
# dt_confusion_matrix = confusion_matrix(y_test, dt_y_pred)

# print("\nDecision Tree:")
# print("Accuracy:", dt_accuracy)
# print("Precision:", dt_precision)
# print("Recall:", dt_recall)
# print("F1 Score:", dt_f1)
# print("Classification Report:\n", dt_report)
# print("Confusion Matrix:\n", dt_confusion_matrix)

# knn_y_pred = knn_model.predict(X_test)
# knn_accuracy = accuracy_score(y_test, knn_y_pred)
# knn_precision = precision_score(y_test, knn_y_pred)
# knn_recall = recall_score(y_test, knn_y_pred)
# knn_f1 = f1_score(y_test, knn_y_pred)
# knn_report = classification_report(y_test, knn_y_pred)
# knn_confusion_matrix = confusion_matrix(y_test, knn_y_pred)

# print("\nKNN:")
# print("Accuracy:", knn_accuracy)
# print("Precision:", knn_precision)
# print("Recall:", knn_recall)
# print("F1 Score:", knn_f1)
# print("Classification Report:\n", knn_report)
# print("Confusion Matrix:\n", knn_confusion_matrix)





Please explain the reason which model performed well and why  -> theory.csv


What are the possible techniqes available to improve the accuracy and implement one of those technique -> theory.csv


