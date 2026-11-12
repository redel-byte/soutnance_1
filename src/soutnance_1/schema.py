TARGET = "Mental_Health_Risk"
CLASSES = ("High", "Low", "Medium")

ORDINAL_CATEGORIES = {
    "Year_of_Study": [1, 2, 3, 4, 5],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}
NOMINAL_CATEGORIES = {
    "Gender": ["Female", "Male", "Non-binary"],
    "Part_Time_Job": ["No", "Yes"],
    "Previous_Mental_Health_History": ["No", "Yes"],
}
NUMERIC_COLUMNS = (
    "Age", "Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours",
    "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate",
)
FEATURE_COLUMNS = (*NUMERIC_COLUMNS, *ORDINAL_CATEGORIES, *NOMINAL_CATEGORIES)
