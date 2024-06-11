import os
import fitz  # pip install --upgrade pip; pip install --upgrade pymupdf
from tqdm import tqdm  # pip install tqdm
import argparse


def extract_imgs_from_path(workdir):
    """
    extracts all images from all pdf files in a given directory
    :param workdir: path to the directory containing the pdfs
    :return:
    """
    for filename in os.listdir(workdir):
        if ".pdf" in filename:
            doc = fitz.Document((os.path.join(workdir, filename)))

            for i in tqdm(range(len(doc)), desc="pages"):
                for img in tqdm(doc.get_page_images(i), desc="page_images"):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    pix.save(os.path.join(workdir, "%s_p%s-%s.png" % (filename[:-4], i, xref)))

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