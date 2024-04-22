import os
import fitz  # pip install --upgrade pip; pip install --upgrade pymupdf
from tqdm import tqdm  # pip install tqdm
import argparse


def extract_imgs_from_path(workdir):
    for each_path in os.listdir(workdir):
        if ".pdf" in each_path:
            doc = fitz.Document((os.path.join(workdir, each_path)))

            for i in tqdm(range(len(doc)), desc="pages"):
                for img in tqdm(doc.get_page_images(i), desc="page_images"):
                    xref = img[0]
                    image = doc.extract_image(xref)
                    pix = fitz.Pixmap(doc, xref)
                    pix.save(os.path.join(workdir,"/images/", "%s_p%s-%s.png" % (each_path[:-4], i, xref)))

    print("Done!")


if __name__ == '__main__':

    cl_argparser = argparse.ArgumentParser(
        description="This extracts all images from all pdf files for a given working directory"
    )
    cl_argparser.add_argument("-d", "--directory", help="the path of the directory the image-containing"
                                                        "pdf-files are in.")

    args = cl_argparser.parse_args()
    workdir = args.directory
    extract_imgs_from_path(workdir)