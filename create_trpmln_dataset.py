#!/usr/bin/env python3
# This script generates the Dataset TRPMLN for lung nodules from LIDC-IDRI.

import os
import sys
import argparse


import gc # clean ram used by garbage collecrtore
import random

import cv2 # for normalise image and save in specific format
import pandas as pd # for save some information to scv
import pylidc as pl # we need pylidc to query lcidi-idri datasete
from tqdm.auto import tqdm # progress bar


class ScanData:
    def __init__(self, path=None):
        if path is None:
            raise KeyError("please provied path of LIDCI-IDRI")
        self.scans = self.create_pylidcrc(path)
        self.extract_data()

    def create_pylidcrc(self, path):
        config_file = "/root/.pylidcrc"
        config = f"[dicom]\npath={path}"    
        with open(config_file, "w") as f:
            f.write(config)
        # scans += pl.query(pl.Scan).all() # for query all slices
        return pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 3, pl.Scan.pixel_spacing <= 1)

    def extract_data(self):
        self.data = []
        total_scans = self.scans.count()
        for i, scan in tqdm(enumerate(self.scans), total=total_scans):
            # if i > 4: # deactivate the test for the 5 first items.
            #     break
            nodules = scan.cluster_annotations()
            # Note: for each scan.id we have many nodules, each nodules
            # has many anns from diffrent experts.
            for anns in nodules:
                malignancies = 0
                for ann in anns:
                    malignancies += ann.malignancy
                avg_malignancy = malignancies / len(anns)
                cancer = 1 if avg_malignancy >= 3 else 0
                cancer_name = "cancer" if cancer else "normal"

                #ann = random.choice(anns) # ROI extracting depend the ann celected.
                ann = anns[0]
                roi_name = f"{cancer_name}_{scan.patient_id}_{scan.id}_{ann.id}.tiff"

                row = {
                    "roi_name": roi_name,
                    "ann": ann,
                    # "scan_id": scan.id,
                    # "rand_nodule_id": ann.id,
                    "cancer": cancer,
                }
                self.data.append(row)

        return self

    def write_to_csv(self, filename):
        if filename is None or filename == "":
            raise KeyError("you miss name of csv file to store into data info")
        df = pd.DataFrame(self.data, columns=["roi_name", "cancer"])
        df.to_csv(filename, index=False)
        return self

    def save_roi_to_tiff(self, dir=None):
        if dir is None or dir == "":
            raise KeyError("you miss name of dir to store images")
        # padding = [(0, 0), (0, 0), (0, 0)] # for no padding
        padding = [(30, 10), (10, 25), (0, 0)]
        for i, row in tqdm(enumerate(self.data), total=len(self.data)):
            vol, roi, bbox, ann = None, None, None, None
            ann = row["ann"]
            bbox = ann.bbox(pad=padding)
            try:
                vol = ann.scan.to_volume()
            except Exception as e:
                print(f'Warning: {e}')
                continue

            for region in range(vol[bbox].shape[2]):
                roi = vol[bbox][:, :, region]
                # Rescale the ROI image to the range of 0 to 255 for 8-bit images
                roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Save the image as a TIFF file in the patient directory
                filename = row["roi_name"]
                cv2.imwrite(f"{dir}/{filename}", roi)
            if i % 10: # clean some ram usage, use more cpu and time.
                gc.collect()
        return self

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate the Dataset TRPMLN for lung nodules from LIDC-IDRI.")
    parser.add_argument("-d", "--dataset", type=str, help="The path for Dataset LIDC-IDRI")
    parser.add_argument("-r", "--roi", type=str, help="The path for the ROI directory extracted.")
    parser.add_argument("-c", "--csv", type=str, help="The path for the csv file generated.")

    args = parser.parse_args(sys.argv[1:])


    if args.dataset is None:
        raise ValueError("Please provide the path for LIDC-IDRI Dataset.")

    if not os.path.exists(args.dataset):
        raise ValueError(f"Dir {args.dataset} does not exist.")

    if args.roi is None:
        raise ValueError("Please provide the path for ROI Directory output.")

    if not os.path.exists(args.roi):
        raise ValueError(f"Dir {args.roi} does not exist.")

    if args.csv is None:
        raise ValueError("Please provide the path to store the csv filename generated.")

    scan_data = ScanData(path=args.dataset)
    scan_data.write_to_csv(filename=args.csv)
    scan_data.save_roi_to_tiff(args.roi)
