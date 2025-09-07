import re
import os

# Define the activities and their corresponding one-hot encoded labels.
ACTIVITY_LABELS = {
    "Jogging": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Walking": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "Downstairs": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "Upstairs": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Sitting": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Standing": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

# The number of client files you have.
CLIENT_COUNT = 9


def parse_and_merge_data_to_single_file():
    """
    Parses sensor data from all client files, merges all records into a
    single list, appends the correct label, and writes everything
    to a single output CSV file.
    """
    print("Starting data merging process...")

    # A single list to hold all records from all activities and clients.
    all_records = []

    # Loop through each client file from Client 1.txt to Client 9.txt
    for i in range(1, CLIENT_COUNT + 1):
        filename = f"Client {i}.txt"

        # Check if the file exists before trying to open it.
        if not os.path.exists(filename):
            print(f"⚠️  Warning: File '{filename}' not found. Skipping.")
            continue

        print(f"Processing '{filename}'...")

        with open(filename, 'r') as file:
            content = file.read()

        # Split the file content by activity sections.
        # The pattern looks for '//// ActivityName' and uses it as a delimiter.
        parts = re.split(r'////\s*(\w+)', content)

        # Iterate over the parts, two at a time (activity name, then its data block)
        for j in range(1, len(parts), 2):
            activity_name = parts[j].strip()
            activity_data_block = parts[j + 1]

            if activity_name in ACTIVITY_LABELS:
                # Find all data records enclosed in curly braces {}
                records = re.findall(r'\{(.*?)\}', activity_data_block, re.DOTALL)

                label = ACTIVITY_LABELS[activity_name]

                for record_str in records:
                    # Clean up the string: remove newlines and leading/trailing spaces.
                    cleaned_str = record_str.replace('\n', '').strip()
                    if not cleaned_str:
                        continue

                    # Split the string of numbers by commas and convert them to floats.
                    try:
                        data_points = [float(p) for p in cleaned_str.split(',') if p.strip()]
                        # Append the list of data points and its corresponding label to the master list.
                        all_records.append((data_points, label))
                    except ValueError as e:
                        print(
                            f"  ⚠️  Warning: Could not parse a record in '{filename}' for activity '{activity_name}'. Error: {e}")
                        print(f"     Problematic data snippet: '{cleaned_str[:50]}...'")

    # --- Write the merged data to a SINGLE output file ---
    print("\nWriting all merged data to a single file...")
    if not all_records:
        print("No data was found in any client file. Output file will not be created.")
        return

    output_filename = "all_activities_merged.csv"
    with open(output_filename, 'w') as f:
        # Iterate through the master list and write each record to the file
        for data_points, label in all_records:
            # Create comma-separated strings for the data and the label
            data_str = ",".join(map(str, data_points))
            label_str = ",".join(map(str, label))
            # Write a single line with all data points followed by the label components
            f.write(f"{data_str},{label_str}\n")

    print(f"✅ Successfully created '{output_filename}' with {len(all_records)} total records.")


# This makes the script executable
if __name__ == "__main__":
    parse_and_merge_data_to_single_file()
    print("\nScript finished successfully! 🎉")