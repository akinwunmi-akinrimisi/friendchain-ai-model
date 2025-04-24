import json

# Load the mock profiles
with open("mock_profiles.json", "r") as file:
    profiles = json.load(file)

# Print the first profile to verify
print("Number of profiles:", len(profiles))
print("First profile:")
print(json.dumps(profiles[0], indent=2))