from GONet_Wizard.GONet_utils import GONetFile
from PIL import Image

def main():
    go = GONetFile.from_file(r"Testing Images\202_250628_063009_1751092241.jpg")
    meta = go.meta
    #print(meta)

    with Image.open(r"Testing Images\202_250628_063009_1751092241.jpg") as img:
        exif_data = img.getexif()
    print(exif_data)

    #go2 = GONetFile.from_file(r"Testing Images\exiftesting3.jpg")
    #meta2 = go2.meta

    with Image.open(r"Testing Images\exiftesting3.jpg") as img2:
        exif_data2 = img2.getexif()
    print(exif_data2)

    #print("go.meta Comparison")
    #print(meta==meta2)

    print("exif_data Comparison")
    print(exif_data==exif_data2)

if __name__ == "__main__":
    main()