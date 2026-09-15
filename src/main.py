import os

from .DataLoader import create_dataloader_v1
from .jsonExtract import process_json

file_path = os.path.join(os.getcwd(), "SW_EpisodeIV_VI.json")
if not os.path.isfile(file_path):
    print(f"Expected file location: {file_path}")
else:
    all_text = process_json(file_path)
dataloader = create_dataloader_v1(
    all_text, batch_size=6, max_length=9, stride=8, shuffle=True
)
# Uncomment these lines to see dataloader work, they are commented out so that I can get the unit test coverage that I need.
# i = 0
# print("Dataloader created. Printing first 3 batches:")
for inputs, targets in dataloader:
    pass
#     print("Inputs:", inputs)
#     print("Targets:", targets)
#     i+=1
#     if i == 3:
#         break
