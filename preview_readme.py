# #!/usr/bin/env python3
# import re

# # Read the auto‐generated structure
# with open("STRUCTURE.md", encoding="utf8") as f:
#     structure = f.read().rstrip()

# # Read your original README
# with open("README.md", encoding="utf8") as f:
#     readme = f.read()

# # Replace the section between the markers
# new_readme = re.sub(
#     r"<!-- STRUCTURE_START -->.*?<!-- STRUCTURE_END -->",
#     f"<!-- STRUCTURE_START -->\n{structure}\n<!-- STRUCTURE_END -->",
#     readme,
#     flags=re.DOTALL,
# )

# # Write to preview file
# with open("README.preview.md", "w", encoding="utf8") as f:
#     f.write(new_readme)

# print("✅ Generated README.preview.md")
