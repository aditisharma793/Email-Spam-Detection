import pandas as pd
import re

# Step 1: Load the original dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection.tsv",
    sep="\t",
    header=None,
    names=["Category", "Message"]
)

print("Original Dataset:")
print(data.head())

# Step 2: Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Remove rows with missing values
data = data.dropna(subset=["Category", "Message"])

# Step 4: Remove duplicate records
data = data.drop_duplicates(subset=["Category", "Message"])

# Step 5: Convert category into lowercase
data["Category"] = data["Category"].str.lower().str.strip()

# Step 6: Create a function to clean messages
def clean_text(message):

    # Convert text into lowercase
    message = message.lower()

    # Remove special characters and punctuation
    message = re.sub(r"[^a-z0-9\s]", " ", message)

    # Remove extra spaces
    message = re.sub(r"\s+", " ", message).strip()

    return message

# Step 7: Apply cleaning to all messages
data["Cleaned_Message"] = data["Message"].apply(clean_text)

# Step 8: Remove empty messages
data = data[data["Cleaned_Message"] != ""]

# Step 9: Display cleaned dataset
print("\nCleaned Dataset:")
print(data[["Category", "Cleaned_Message"]].head())

# Step 10: Display final information
print("\nFinal Dataset Shape:")
print(data.shape)

print("\nFinal Category Counts:")
print(data["Category"].value_counts())

# Step 11: Save cleaned dataset
data.to_csv("dataset/cleaned_sms.csv", index=False)

print("\nData preprocessing completed successfully!")