"""
This is for automatic logging from the local current.txt file.
Make sure to add any changes to the current.txt before committing to this.
"""

import sys
import os
import re

def parse_file(path):
    """
    Read 'current.txt' file and extract the date, version, title, and description.
    Expects lines in the form:
        date: 2025-XX-XX
        version: X.X.X
        title: XX
        author: XX
        description: | (then multiline)
            - Something
            - Something else
    """
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        sys.exit(1)
    
    # placeholders
    date = None
    version = None
    title = None
    author = None
    description = []

    # to track description
    in_description = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            match = re.match(r"^(\w+):\s*(.*)", line)
            if match and not in_description:
                key, value = match.groups()
                key = key.lower()

                if key == "date":
                    date = value.strip()
                elif key == "version":
                    version = value.strip()
                elif key == "title":
                    title = value.strip()
                elif key == "author":
                    author = value.strip()
                elif key == "description":
                    in_description = True
            else:
                if in_description:
                    # note for later: might want to skip blank lines or any weird formatting
                    description.append(line)

        description = "\n".join(description).strip()

        if not (date and version and title and author):
            print("Error: 'date', 'version', or 'title' is missing in 'current.txt'")
            sys.exit(1)

        return date, version, title, author, description
    
def format_for_markdown(date, version, title, author, description):
    """
    Returns a markdown-formatted string given the date, version, title, author, and description.
    """
    md_entry = []
    md_entry.append(f"## **Version [{version}]** - {date}")
    md_entry.append(f"**Change:** {title}")
    md_entry.append(f"**Author:** {author}")
    md_entry.append("")
    if description:
        md_entry.append(description)

    # blank line for spacing
    md_entry.append("")
    return "\n".join(md_entry)

def format_for_text(date, version, title, author, description):
    """
    Returns a plain text-formatted string given the date, version, title, author, and description.
    """
    txt_entry = []
    txt_entry.append(f"Version: {version}  |  Date: {date}")
    txt_entry.append(f"Change: {title}")
    txt_entry.append(f"Author: {author}")
    if description:
        txt_entry.append(description)
    txt_entry.append("-" * 60)
    # separator / blank line
    txt_entry.append("")
    return "\n".join(txt_entry)

def append_to_files(md_entry, txt_entry, md_file="changelog/README.md", txt_file="changelog/changelog.txt"):
    """
    Appends entries to Markdown and text files.
    If the files don't exist, create one.
    """
    # Append to changelog.md
    with open(md_file, "a", encoding="utf-8") as f_md:
        f_md.write(md_entry)

    # Append to changelog.txt
    with open(txt_file, "a", encoding="utf-8") as f_txt:
        f_txt.write(txt_entry)

def run_log():
    local_path = "changelog/current.txt"
    date, version, title, author, description = parse_file(local_path)
    md_entry = format_for_markdown(date, version, title, author, description)
    txt_entry = format_for_text(date, version, title, author, description)
    append_to_files(md_entry, txt_entry)
    print(f"Appended changes for version {version} to changelog/README.md and changelog/changelog.txt")