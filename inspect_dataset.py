import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection.tsv",
    sep="\t",
    header=None,
    names=["Category", "Message"]
)

# Display first five records
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Count spam and ham messages
print("\nMessage Categories:")
print(data["Category"].value_counts())

# Check dataset size
print("\nDataset Shape:")
print(data.shape)

# Display Spam vs Ham graph
data["Category"].value_counts().plot(
    kind="bar",
    title="Spam vs Ham Messages"
)

plt.xlabel("Message Category")
plt.ylabel("Number of Messages")
plt.show()

# Check missing values
print("Missing values:")
print(data.isnull().sum())

# Check duplicate records
print("Duplicate records:")
print(data.duplicated().sum())