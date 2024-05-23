import random

# Function to generate random credit card numbers
def generate_credit_card(correct_format=True):
    if correct_format:
        numbers = ''.join(random.choice('0123456789') for _ in range(16))
        cc_number = '-'.join(numbers[i:i+4] for i in range(0, len(numbers), 4))
        return cc_number, 1
    else:
        cc_number = ''.join(random.choice('0123456789') for _ in range(random.randint(10, 20)))
        return cc_number, 0

# Generate dataset
dataset = []
for _ in range(100):
    correct_cc, _ = generate_credit_card(correct_format=True)
    dataset.append((correct_cc, 1))
    wrong_cc, _ = generate_credit_card(correct_format=False)
    dataset.append((wrong_cc, 0))

# Shuffle the dataset
random.shuffle(dataset)

# Save dataset to CSV file
with open('credit_card_dataset.csv', 'w') as f:
    f.write("text,label\n")
    for cc_number, label in dataset:
        f.write(f"{cc_number},{label}\n")

print("Dataset created successfully!")
