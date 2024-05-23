import random

# List of domain names
domains = ['google.com', 'yahoo.com', 'tcs.com', 'abs.com','gmail']

# Function to generate random email addresses
def generate_email(correct_format=True):
    if correct_format:
        username = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(random.randint(5, 10)))
        domain = random.choice(domains)
        return f"{username}@{domain}", 1
    else:
        email = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(random.randint(5, 10)))
        return email, 0

# Generate dataset
dataset = []
for _ in range(100):
    correct_email, _ = generate_email(correct_format=True)
    dataset.append((correct_email, 1))
    wrong_email, _ = generate_email(correct_format=False)
    dataset.append((wrong_email, 0))

# Shuffle the dataset
random.shuffle(dataset)

# Save dataset to CSV file
with open('email_dataset.csv', 'w') as f:
    f.write("text,label\n")
    for email, label in dataset:
        f.write(f"{email},{label}\n")

print("Dataset created successfully!")
