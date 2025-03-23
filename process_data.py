import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jdatetime  # For Persian date conversion in BenefactorsData

# Load datasets
benefactors_df = pd.read_csv('gp-datamining/datasets/BenefactorsData.csv')
transactional_df = pd.read_csv('gp-datamining/datasets/TransactionalData.csv')

# Convert Persian dates to Gregorian (only for BenefactorsData)
def persian_to_gregorian(persian_date):
    if pd.isna(persian_date) or persian_date == '':
        return None
    try:
        year, month, day = map(int, persian_date.split('-'))
        j_date = jdatetime.date(year, month, day)
        g_date = j_date.togregorian()
        return g_date.strftime('%Y-%m-%d')
    except:
        return None

benefactors_df['BirthDate_Gregorian'] = benefactors_df['BirthDate'].apply(persian_to_gregorian)

# State translation mapping (Persian to English)
state_mapping = {
    'تهران': 'Tehran',
    'البرز': 'Alborz',
    'اصفهان': 'Isfahan',
    'فارس': 'Fars',
    'خوزستان': 'Khuzestan',
    'خراسان رضوی': 'Razavi Khorasan',
    'آذربایجان شرقی': 'East Azerbaijan',
    'مازندران': 'Mazandaran',
    'گیلان': 'Gilan',
    'کرمان': 'Kerman',
    'سیستان و بلوچستان': 'Sistan and Baluchestan',
    'هرمزگان': 'Hormozgan',
    'کرمانشاه': 'Kermanshah',
    'گلستان': 'Golestan',
    'لرستان': 'Lorestan',
    'همدان': 'Hamadan',
    'قم': 'Qom',
    'مرکزی': 'Markazi',
    'یزد': 'Yazd',
    'بوشهر': 'Bushehr',
    'اردبیل': 'Ardabil',
    'قزوین': 'Qazvin',
    'زنجان': 'Zanjan',
    'چهارمحال و بختیاری': 'Chaharmahal and Bakhtiari',
    'آذربایجان غربی': 'West Azerbaijan',
    'کردستان': 'Kurdistan',
    'ایلام': 'Ilam',
    'کهگیلویه و بویراحمد': 'Kohgiluyeh and Boyer-Ahmad',
    'خراسان شمالی': 'North Khorasan',
    'خراسان جنوبی': 'South Khorasan',
    'سمنان': 'Semnan'
}

# --- Step 1: General Dataset Evaluation ---
def evaluate_dataset(df, name):
    print(f"\n=== {name} Overview ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Numerical columns:", df.select_dtypes(include=['int64', 'float64']).columns.tolist())
    print("Categorical columns:", df.select_dtypes(include=['object']).columns.tolist())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))
    
    # Clean Gender before plotting
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'زن': 'Female', 'مرد': 'Male'})
    
    # Translate State before plotting
    if 'State' in df.columns:
        df['State'] = df['State'].map(state_mapping).fillna(df['State'])
    
    # Visualize
    if 'PaymentAmount' in df.columns:
        sns.histplot(df['PaymentAmount'].dropna(), bins=20)
        plt.title(f"{name} - Payment Amount Distribution")
        plt.xlabel("Payment Amount")
        plt.ylabel("Count")
        plt.show()
    
    for col in ['Gender', 'ReferralSource', 'SupportType']:
        if col in df.columns:
            sns.countplot(x=col, data=df)
            plt.title(f"{name} - Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()
    
    # Special handling for State: Show Tehran, Alborz, and "Other"
    if 'State' in df.columns:
        top_2_states = ['Tehran', 'Alborz']
        df['State_Grouped'] = df['State'].apply(lambda x: x if x in top_2_states else 'Other')
        state_order = ['Tehran', 'Alborz', 'Other']
        sns.countplot(x='State_Grouped', data=df, order=state_order)
        plt.title(f"{name} - Distribution of States (Tehran, Alborz, Other)")
        plt.xlabel("State")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

evaluate_dataset(benefactors_df, "Benefactors Data")
evaluate_dataset(transactional_df, "Transactional Data")

# --- Data Cleaning ---
def clean_data(df, name):
    print(f"\n=== Cleaning {name} ===")
    print("Missing Values Before:")
    print(df.isnull().sum())
    
    if 'BirthDate' in df.columns:
        df['BirthDate'].fillna('Unknown', inplace=True)
        df['BirthDate_Gregorian'].fillna('Unknown', inplace=True)
        df['Gender'] = df['Gender'].replace({'زن': 'Female', 'مرد': 'Male'})
        df['State'] = df['State'].map(state_mapping).fillna(df['State'])
        df['BirthDate_Gregorian'] = pd.to_datetime(df['BirthDate_Gregorian'], errors='coerce')
        df['Age'] = (pd.to_datetime('2025-03-22') - df['BirthDate_Gregorian']).dt.days // 365
        # Replace BirthDate with the Gregorian version
        df = df.drop(columns=['BirthDate'])  # Drop the original Persian BirthDate
        df = df.rename(columns={'BirthDate_Gregorian': 'BirthDate'})  # Rename the Gregorian column to BirthDate
    if 'PaymentDate' in df.columns:
        df['PaymentDate'] = pd.to_datetime(df['PaymentDate'], errors='coerce')
        df['PaymentDate'].fillna('Unknown', inplace=True)
    
    print("Missing Values After:")
    print(df.isnull().sum())

    return df

benefactors_df = clean_data(benefactors_df, "Benefactors Data")
transactional_df = clean_data(transactional_df, "Transactional Data")

# --- Identifying Data Distribution ---
def check_distribution(df, name):
    print(f"\n=== {name} Distribution ===")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        if col in ['Age', 'PaymentAmount']:
            sns.histplot(df[col].dropna(), bins=20)
            plt.title(f"{name} - {col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.show()
            z_scores = stats.zscore(df[col].dropna())
            outliers = len([x for x in z_scores if abs(x) > 3])
            print(f"{col} - Outliers (Z-score > 3): {outliers}")

check_distribution(benefactors_df, "Benefactors Data")
check_distribution(transactional_df, "Transactional Data")

# --- Relationship Between Features ---
def feature_relationship(df, name):
    print(f"\n=== {name} Feature Relationships ===")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f"{name} - Numerical Feature Correlation")
    plt.show()
    
    if 'Gender' in df.columns:
        sns.catplot(x='Gender', hue='ReferralSource', kind='count', data=df)
        plt.title(f"{name} - Gender vs Referral Source")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.show()
    if 'SupportType' in df.columns:
        sns.catplot(x='SupportType', y='PaymentAmount', kind='box', data=df)
        plt.title(f"{name} - Support Type vs Payment Amount")
        plt.xlabel("Support Type")
        plt.ylabel("Payment Amount")
        plt.show()

feature_relationship(benefactors_df, "Benefactors Data")
feature_relationship(transactional_df, "Transactional Data")

# --- TransactionalData Specific Steps ---
filtered_transactions = transactional_df[transactional_df['PaymentAmount'] > 100000]
print("\nFiltered Transactions (PaymentAmount > 100,000):", filtered_transactions.shape)

agg_data = transactional_df.groupby('UserID').agg({
    'PaymentAmount': 'sum',
    'TransID': 'count',
    'PaymentDate': 'max'
}).rename(columns={'TransID': 'Frequency', 'PaymentAmount': 'TotalAmount'})
print("\nAggregated Transactional Data:")
print(agg_data.head())

# Categorize R, F, M with dynamic binning
agg_data['Recency'] = (pd.to_datetime('2025-03-22') - agg_data['PaymentDate']).dt.days
for col in ['Recency', 'Frequency', 'TotalAmount']:
    unique_vals = agg_data[col].nunique()
    if unique_vals > 1:
        n_bins = min(4, unique_vals)
        try:
            agg_data[f"{col}_cat"] = pd.qcut(agg_data[col], n_bins, labels=range(1, n_bins + 1), duplicates='drop')
        except ValueError as e:
            print(f"Could not categorize {col}: {e}")
            agg_data[f"{col}_cat"] = pd.NA
    else:
        print(f"Skipping {col} categorization: only {unique_vals} unique value(s)")
        agg_data[f"{col}_cat"] = 1
print("\nCategorized R, F, M:")
print(agg_data.head())

# --- BenefactorsData Exploration ---
print("\n=== Benefactors Data Exploration ===")
print("Unique States:", benefactors_df['State'].nunique())
print("Gender Distribution:", benefactors_df['Gender'].value_counts())
print("Most Common Referral Source:", benefactors_df['ReferralSource'].mode()[0])
print("Average Age by Gender:", benefactors_df.groupby('Gender')['Age'].mean())

# --- Combined Analysis ---
combined_df = pd.merge(benefactors_df, agg_data, on='UserID', how='left')
print("\n=== Combined Data Sample ===")
print(combined_df.head())

sns.boxplot(x='ReferralSource', y='TotalAmount', data=combined_df)
plt.title("Referral Source vs Total Payment Amount")
plt.xlabel("Referral Source")
plt.ylabel("Total Payment Amount")
plt.xticks(rotation=45)
plt.show()