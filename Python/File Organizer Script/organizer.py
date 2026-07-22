from pathlib import Path
import shutil

FILE_TYPES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "Documents": [".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx", ".pptx"],
    "Videos": [".mp4", ".mkv", ".avi", ".mov"],
    "Music": [".mp3", ".wav", ".aac"],
    "Archives": [".zip", ".rar", ".7z"],
    "Code": [".py", ".java", ".cpp", ".c", ".html", ".css", ".js"]
}


def get_category(extension):
    extension = extension.lower()

    for category, extensions in FILE_TYPES.items():
        if extension in extensions:
            return category

    return "Others"


def move_file(file_path, destination_folder):

    destination_folder.mkdir(exist_ok=True)

    destination = destination_folder / file_path.name

    count = 1

    while destination.exists():

        destination = destination_folder / f"{file_path.stem}_{count}{file_path.suffix}"

        count += 1

    shutil.move(str(file_path), str(destination))


def organize_folder(folder_path):

    folder = Path(folder_path)

    moved = 0
    skipped = 0
    errors = 0

    print("\n========== FILE ORGANIZER ==========\n")

    for item in folder.iterdir():

        if item.is_dir():
            continue

        if item.name.startswith("."):
            skipped += 1
            print("Skipped Hidden File:", item.name)
            continue

        category = get_category(item.suffix)

        destination = folder / category

        try:

            move_file(item, destination)

            print(f"Moved: {item.name}  -->  {category}")

            moved += 1

        except Exception as e:

            print(f"Error moving {item.name}")

            print(e)

            errors += 1

    print("\n========== SUMMARY ==========")

    print("Moved Files   :", moved)

    print("Skipped Files :", skipped)

    print("Errors        :", errors)

    print("\nOrganization Complete!")


def main():

    print("=" * 45)

    print("FILE ORGANIZER SCRIPT")

    print("=" * 45)

    folder = input("\nEnter folder path: ").strip()

    folder_path = Path(folder)

    if not folder_path.exists():

        print("\nFolder does not exist.")

        return

    organize_folder(folder_path)


if __name__ == "__main__":
    main()
