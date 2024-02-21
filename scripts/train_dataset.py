import os
import shutil

def organize_images(input_dir, output_dir):
    with os.scandir(input_dir) as entries:
        for entriy in entries:
            filename = entriy.name
            if filename.endswith('.jpg'):
                # Extract the prefix (e.g., 'GCC_train_') and the number part
                _, _, number = filename.split('_')
                # Create the target directory path
                target_dir = os.path.join(output_dir, number[:5])
                # Create the target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                # Move the file to the target directory
                shutil.move(os.path.join(input_dir, filename), os.path.join(target_dir, number))

if __name__ == "__main__":
    dir = "./playground/data/LLaVA-Pretrain/images"

    organize_images(dir, dir)
    print("Images organized successfully!")
