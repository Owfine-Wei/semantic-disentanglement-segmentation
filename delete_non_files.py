import os

def delete_specific_files(root_dir, keywords):
    """
    Traverse directories and delete files containing specific keywords.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found {root_dir}")
        return

    print(f"Start scanning directory: {root_dir}")
    deleted_count = 0

    # Traverse directory using os.walk
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if filename contains any of the keywords
            if any(keyword in file for keyword in keywords):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"Operation completed. Total files deleted: {deleted_count}")

if __name__ == "__main__":
    # 1. Specify root directory (use r prefix for raw string to avoid backslash escape issues)
    target_directory = r"/root/autodl-tmp/data/Cityscapes/gtFine"
    
    # 2. Specify keywords contained in the filenames to be deleted
    target_keywords = ['color', 'instanceIds', 'labelIds', 'polygons']

    # Execute delete function
    # For safety, it is recommended to run once to check print output before actual deletion.
    # But according to your requirements, the deletion logic is executed directly here.
    delete_specific_files(target_directory, target_keywords)
